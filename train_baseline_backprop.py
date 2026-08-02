"""Backprop baseline for the discrete-learning research.

Trains the SAME architecture as `train_discrete.py` (stochastic bit-flip
ternary model, tiny preset) but with ordinary backpropagation + AdamW, on the
same data and at the same step/token budget. Reports held-out cross-entropy on
the same eval slice used by the C++ `selflearn --eval` measurements, so the
gradient-free rules can be compared apples-to-apples against a real optimizer.

Usage:
    python train_baseline_backprop.py --steps 300 --save-dir checkpoints_bp
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from ternary_llm.data import create_dataloaders
from ternary_llm.transformer import StochasticTransformerModel


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
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--data-cache", type=str, default="tinydata")
    parser.add_argument("--eval-slice", type=str,
                        default="checkpoints_discrete_c2/sliceEval100k.bin")
    parser.add_argument("--eval-positions", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--no-flips", action="store_true",
                        help="Freeze ternary weights (skip apply_bit_flips) — "
                             "ablation: backprop only trains embedding/lm_head")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints_bp")
    args = parser.parse_args()

    data_cache = Path(args.data_cache)
    meta_path = data_cache / "metadata.json"
    bin_path = data_cache / "tinystories.bin"
    with open(meta_path) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    tokens = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
    print(f"Tokens: {len(tokens):,} | Vocab: {vocab_size}")

    model = StochasticTransformerModel(
        vocab_size=vocab_size, hidden_dim=256, num_layers=6, num_heads=8,
        ffn_dim=1024, max_seq_len=2048, threshold=args.threshold,
    )
    n_ternary = 0
    for name, buf in model.named_buffers():
        if name.endswith(".packed_weights"):
            n_ternary += buf.numel() * 4
    print(f"Model: 6L/256/8H/1024FFN, ternary bits: {n_ternary:,}")

    train_loader, val_loader = create_dataloaders(
        tokens, block_size=args.block_size, batch_size=args.batch_size,
        val_split=args.val_split, num_workers=0,
    )

    eval_tokens = load_eval_tokens(args.eval_slice, args.eval_positions)
    print(f"Eval slice: {args.eval_slice} ({len(eval_tokens)} tokens, "
          f"{args.eval_positions} positions)")
    print(f"Mode: ternary flips {'DISABLED (frozen)' if args.no_flips else 'ENABLED'}")

    # Held-out (last 5%) reference at init.
    val_tokens = tokens[int(len(tokens) * (1 - args.val_split)):]
    ce_init_val = eval_ce(model, np.asarray(val_tokens), args.block_size,
                          args.eval_positions)
    ce_init_slice = eval_ce(model, eval_tokens, args.block_size,
                            args.eval_positions)
    print(f"[step 0] heldout(last5%) CE {ce_init_val:.4f} | slice CE {ce_init_slice:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: _schedule(s, args.steps, args.warmup,
                                           args.lr, args.min_lr))
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
            model.apply_bit_flips()
        losses.append(loss.item())

        if step % 10 == 0 or step == args.steps:
            print(f"[step {step}] train CE {loss.item():.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | "
                  f"avg10 {np.mean(losses[-10:]):.4f}")

        if step % args.eval_every == 0 or step == args.steps:
            ce_val = eval_ce(model, np.asarray(val_tokens), args.block_size,
                             args.eval_positions)
            ce_slice = eval_ce(model, eval_tokens, args.block_size,
                               args.eval_positions)
            print(f"  eval[step {step}] heldout(last5%) CE {ce_val:.4f} | "
                  f"slice CE {ce_slice:.4f}")

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


def _schedule(step: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return 0.1 + 0.9 * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return max(min_lr / lr, 0.5 * (1.0 + np.cos(np.pi * t)))


if __name__ == "__main__":
    main()
