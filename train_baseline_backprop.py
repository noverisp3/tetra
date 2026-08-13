"""Backprop baseline for the discrete-learning research.

Trains the SAME architecture as `train_discrete.py` (stochastic bit-flip
ternary model, tiny preset) but with ordinary backpropagation + AdamW, on the
same data and at the same step/token budget. Reports held-out cross-entropy on
the same eval slice used by the C++ `selflearn --eval` measurements, so the
gradient-free rules can be compared apples-to-apples against a real optimizer.

Usage:
    python train_baseline_backprop.py --steps 300 --save-dir checkpoints_bp
    python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
        --steps 100 --data-cache data_teacher   # backprop fine-tune (Exp 2, Method 1)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from ternary_llm.data import create_dataloaders, create_multi_source_dataloaders
from ternary_llm.transformer import StochasticTransformerModel
from ternary_llm.arg_utils import DeprecatedFlag, warn_deprecated
from ternary_llm.cli import add_data_args, add_flip_args, configure_cpu_runtime

# Flags from closed experiments, retained verbatim for reproducing the numbers
# in EXPERIMENTS.md. Setting one prints a warning but still works.
DEPRECATED_FLAGS = [
    DeprecatedFlag("--soft-flip-temp", "soft_flip_temp", "rejected (Exp 14)",
                   "smoothing the flip decision adds no transferable gain; the flip "
                   "budget (set by --adaptive-thr) is what the mechanism trades in."),
]


def load_eval_tokens(path: str, n: int) -> np.ndarray:
    """Read up to n uint16 tokens from a raw token slice (mirrors --eval)."""
    data = np.fromfile(path, dtype=np.uint16)
    return data[:n]


@torch.no_grad()
def eval_ce(model, tokens: np.ndarray, block_size: int = 128,
            max_positions: int = 20000, scale: float = 1.0) -> float:
    """Average next-token CE over the slice, causal full-context within chunk.

    Processes the slice in block_size chunks (like the C++ runtime); logits
    may be optionally scaled (scale=1.0 = natural regime for backprop models).
    """
    model.eval()
    total = 0.0
    n = 0
    limit = min(len(tokens) - 1, max_positions)
    for start in range(0, limit, block_size):
        end = min(limit, start + block_size)
        x = torch.tensor(tokens[start:end].astype(np.int64)[None], dtype=torch.long)
        y = torch.tensor(tokens[start + 1:end + 1].astype(np.int64)[None], dtype=torch.long)
        logits, _, _ = model(x, None)
        if scale != 1.0:
            logits = logits * scale
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="mean")
        total += loss.item() * (end - start)
        n += end - start
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Backprop baseline for discrete learning")
    add_data_args(
        parser,
        steps=300, batch_size=16, block_size=128,
        val_split=0.05, data_cache="tinydata",
        device="cpu",
    )
    parser.add_argument("--eval-slice", type=str,
                        default="examples/discrete/sliceEval100k.bin")
    parser.add_argument("--eval-positions", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    add_flip_args(
        parser,
        threshold=20.0, acc_decay=0.99,
        adaptive_thr=None, acc_energy=True,
    )
    parser.add_argument("--no-flips", action="store_true",
                        help="Freeze ternary weights (skip apply_bit_flips) — "
                             "ablation: backprop only trains embedding/lm_head")
    parser.add_argument("--soft-flip-temp", type=float, default=None,
                        help="Exp 14: annealed soft flips — initial relative band "
                             "half-width s (fraction of tau). Flip probability "
                             "sigma(margin/s) inside the band, hard 0/1 outside. "
                             "Cosine-annealed s -> 0.08*s over training (converges "
                             "to the deterministic rule). None = deterministic")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="checkpoints_bp")
    parser.add_argument("--resume", type=str, default=None,
                        help="Phase-1 checkpoint to continue from (backprop "
                             "fine-tuning / continual learning, Exp 2 Method 1)")
    parser.add_argument("--lr-domain", type=float, default=None,
                        help="LR for fine-tuning on a new domain (default: --lr)")
    parser.add_argument("--domain-eval", type=str, default=None,
                        help="Raw uint16 .bin slice of the NEW domain; reports "
                             "its CE alongside the TinyStories slice (adaptation metric)")
    args = parser.parse_args()

    warn_deprecated(parser, args, DEPRECATED_FLAGS)

    if args.device != "cuda" or not torch.cuda.is_available():
        n_threads = configure_cpu_runtime(args.device)
        print(f"CPU runtime: {n_threads} intra-op threads")

    data_cache = Path(args.data_cache)
    meta_path = data_cache / "metadata.json"
    bin_path = data_cache / "tinystories.bin"
    manifest_path = data_cache / "manifest.json"
    tokens = None
    if manifest_path.exists():
        from ternary_llm.data import create_multi_source_dataloaders
        with open(manifest_path) as f:
            manifest = json.load(f)
        vocab_size = manifest["vocab_size"]
        print(f"Sources: {list(manifest['sources'].keys())} | "
              f"Total tokens: {manifest['total_tokens']:,}")
        train_loader, val_loader = create_multi_source_dataloaders(
            data_cache, block_size=args.block_size, batch_size=args.batch_size,
            val_split=args.val_split, num_workers=0,
        )
    else:
        with open(meta_path) as f:
            meta = json.load(f)
        vocab_size = meta["vocab_size"]
        tokens = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        print(f"Tokens: {len(tokens):,} | Vocab: {vocab_size}")
        train_loader, val_loader = create_dataloaders(
            tokens, block_size=args.block_size, batch_size=args.batch_size,
            val_split=args.val_split, num_workers=0,
        )

    model = StochasticTransformerModel(
        vocab_size=vocab_size, hidden_dim=256, num_layers=6, num_heads=8,
        ffn_dim=1024, max_seq_len=2048, threshold=args.threshold,
    )
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_cfg = ckpt["config"]
        # Override architecture from the Phase-1 checkpoint if it differs.
        model = StochasticTransformerModel(
            vocab_size=ckpt_cfg.get("vocab_size", vocab_size),
            hidden_dim=ckpt_cfg.get("hidden_dim", 256),
            num_layers=ckpt_cfg.get("num_layers", 6),
            num_heads=ckpt_cfg.get("num_heads", 8),
            ffn_dim=ckpt_cfg.get("ffn_dim", 1024),
            max_seq_len=ckpt_cfg.get("max_seq_len", 2048),
            threshold=args.threshold,
        )
        sd = {}
        for k, v in ckpt["model_state_dict"].items():
            sd[k] = v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Zero accumulators: replaying the Phase-1 accumulator state on a new
        # domain is unsafe (findings #10/#12 — the old-domain mass kick
        # inverts the matrix). Reset so the first flips reflect new-domain grads.
        for name, buf in model.named_buffers():
            if name.endswith(".accumulator"):
                buf.zero_()
        print(f"Resumed from {args.resume} (step {ckpt.get('step')}); "
              f"missing {len(missing)} / unexpected {len(unexpected)} keys"
              " - accumulators reset")
    n_ternary = 0
    for name, buf in model.named_buffers():
        if name.endswith(".packed_weights"):
            n_ternary += buf.numel() * 4
    print(f"Model: 6L/256/8H/1024FFN, ternary bits: {n_ternary:,}")
    if args.acc_energy or args.adaptive_thr is not None or args.soft_flip_temp is not None:
        model.set_flip_config(acc_decay=args.acc_decay, energy=args.acc_energy,
                              adaptive_thr=args.adaptive_thr,
                              soft_temp=args.soft_flip_temp)
        print(f"Exp 3 flip mechanics: energy acc={args.acc_energy} "
              f"(decay {args.acc_decay}) | adaptive thr k={args.adaptive_thr}"
              + (f" | Exp 14 soft flips T0={args.soft_flip_temp} "
                 f"(anneal -> {0.08 * args.soft_flip_temp:.4f})"
                 if args.soft_flip_temp is not None else ""))

    eval_tokens = load_eval_tokens(args.eval_slice, args.eval_positions)
    print(f"Eval slice: {args.eval_slice} ({len(eval_tokens)} tokens, "
          f"{args.eval_positions} positions)")
    if args.domain_eval:
        domain_tokens = load_eval_tokens(args.domain_eval, args.eval_positions)
        print(f"Domain eval slice: {args.domain_eval} ({len(domain_tokens)} tokens)")
    else:
        domain_tokens = None
    print(f"Mode: ternary flips {'DISABLED (frozen)' if args.no_flips else 'ENABLED'}")
    if args.resume:
        print(f"Fine-tune LR: {args.lr_domain if args.lr_domain else args.lr} "
              f"(was {args.lr})")

    # Held-out (last 5%) reference at init (single-source caches only).
    ce_init_slice = eval_ce(model, eval_tokens, args.block_size,
                            args.eval_positions)
    if tokens is not None:
        val_tokens = tokens[int(len(tokens) * (1 - args.val_split)):]
        ce_init_val = eval_ce(model, np.asarray(val_tokens), args.block_size,
                              args.eval_positions)
        print(f"[step 0] heldout(last5%) CE {ce_init_val:.4f} | "
              f"slice CE {ce_init_slice:.4f}")
    else:
        print(f"[step 0] slice CE {ce_init_slice:.4f}")
    ce_init_domain = (eval_ce(model, domain_tokens, args.block_size, args.eval_positions)
                      if domain_tokens is not None else None)
    if ce_init_domain is not None:
        print(f"[step 0] domain CE {ce_init_domain:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_domain if args.lr_domain else args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: _schedule(s, args.steps, args.warmup,
                                           args.lr_domain if args.lr_domain else args.lr,
                                           args.min_lr))
    torch.manual_seed(0)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_iters = iter(train_loader)
    losses = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iters)
        except StopIteration:
            train_iters = iter(train_loader)
            batch = next(train_iters)
        x, y = batch
        x = x.to(args.device)
        y = y.to(args.device)
        opt.zero_grad(set_to_none=True)
        logits, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        # SBF: flip ternary bits where the accumulated gradient exceeds threshold.
        if not args.no_flips:
            if args.soft_flip_temp is not None:
                # Exp 14: cosine-anneal the soft band down to 8% of T0
                # (nearly deterministic) over the run.
                frac = 0.5 * (1.0 + np.cos(np.pi * step / args.steps))
                model.set_flip_config(
                    soft_temp=args.soft_flip_temp * (0.08 + 0.92 * frac))
            model.apply_bit_flips()
        losses.append(loss.item())

        if step % 10 == 0 or step == args.steps:
            print(f"[step {step}] train CE {loss.item():.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | "
                  f"avg10 {np.mean(losses[-10:]):.4f}")

        if step % args.eval_every == 0 or step == args.steps:
            ce_slice = eval_ce(model, eval_tokens, args.block_size,
                               args.eval_positions)
            if tokens is not None:
                ce_val = eval_ce(model, np.asarray(val_tokens), args.block_size,
                                 args.eval_positions)
                line = (f"  eval[step {step}] heldout(last5%) CE {ce_val:.4f} | "
                        f"slice CE {ce_slice:.4f}")
            else:
                line = f"  eval[step {step}] slice CE {ce_slice:.4f}"
            if domain_tokens is not None:
                ce_dom = eval_ce(model, domain_tokens, args.block_size,
                                 args.eval_positions)
                line += f" | domain CE {ce_dom:.4f}"
            print(line)

        if step % 100 == 0 or step == args.steps:
            ckpt_path = save_dir / f"checkpoint_{step:06d}.pt"
            torch.save({
                "config": {
                    "vocab_size": vocab_size, "hidden_dim": 256,
                    "num_layers": 6, "num_heads": 8, "ffn_dim": 1024,
                    "max_seq_len": 2048, "threshold": args.threshold,
                },
                "model_state_dict": model.state_dict(),
                "step": step,
            }, ckpt_path)
            print(f"  saved {ckpt_path}")

    ce_final = eval_ce(model, eval_tokens, args.block_size, args.eval_positions)
    print(f"\nFINAL slice CE: {ce_final:.4f} "
          f"(init {ce_init_slice:.4f}, baseline random {np.log(vocab_size):.4f})")
    if domain_tokens is not None:
        ce_dom_final = eval_ce(model, domain_tokens, args.block_size,
                               args.eval_positions)
        print(f"FINAL domain CE: {ce_dom_final:.4f} "
              f"(init {ce_init_domain:.4f})")


def _schedule(step: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return 0.1 + 0.9 * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return max(min_lr / lr, 0.5 * (1.0 + np.cos(np.pi * t)))


if __name__ == "__main__":
    main()
