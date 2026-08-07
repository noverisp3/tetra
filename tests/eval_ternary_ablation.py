"""Cut-the-tail ablation: does learned ternary weight structure add anything?

With the SAME frozen (trained) embedding + tied lm_head, evaluate forward pass
under two ternary weight sets:
  1. ternary weights trained 250 steps by rule 'c' (from checkpoint)
  2. ternary weights re-initialized randomly (same init distribution)

If CE(learned) ~= CE(random): the ternary core is effectively decorative —
the model's capability lives in the FP32 embedding + lm_head over a fixed
random projection. If CE(learned) << CE(random): the ternary weights carry
real learned structure that the embedding depends on.

Usage:
    python tests/eval_ternary_ablation.py [--checkpoint ...] [--slice ...] [--positions N]
"""
import argparse
import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from ternary_llm.transformer import StochasticTransformerModel
from ternary_llm.layers import StochasticTernaryLinear, init_ternary_weight
from ternary_llm.quantization import unpack_ternary_tensor, pack_ternary_tensor


def load_eval_tokens(path: str, n: int) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16)[:n]


def ternary_distribution(model) -> dict:
    """Count {-1,0,+1} (and any outliers) across all ternary weights."""
    counts = {}
    per_matrix = []
    for m in model.modules():
        if isinstance(m, StochasticTernaryLinear):
            w = unpack_ternary_tensor(m.packed_weights, (m.out_features, m.in_features))
            c = {}
            for v, n in zip(*(torch.unique(w, return_counts=True))):
                c[int(v)] = int(n)
            per_matrix.append(c)
            for v, n in c.items():
                counts[v] = counts.get(v, 0) + n
    return counts, per_matrix


@torch.no_grad()
def eval_ce(model, tokens: np.ndarray, block_size: int = 128,
            max_positions: int = 20000, scale: float = 1.0) -> float:
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
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="mean")
        total += loss.item() * (end - start)
        n += end - start
    return total / max(n, 1)


def randomize_ternary(model, seed: int = 0):
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, StochasticTernaryLinear):
            m.packed_weights.copy_(
                init_ternary_weight(m.out_features, m.in_features, sparsity=0.5))
            m._w_raw_cache = None


def histmatch_ternary(model, seed: int = 0):
    """Histogram-matched random: shuffle ternary positions within each matrix,
    preserving the exact per-matrix {-1,0,+1} counts of the learned weights.
    """
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, StochasticTernaryLinear):
            w = unpack_ternary_tensor(m.packed_weights, (m.out_features, m.in_features))
            wf = w.flatten()
            perm = torch.randperm(wf.numel())
            shuffled = wf[perm].reshape_as(w)
            m.packed_weights.copy_(pack_ternary_tensor(shuffled))
            m._w_raw_cache = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_discrete_c2/checkpoint_000250.pt")
    parser.add_argument("--slice", type=str,
                        default="examples/discrete/sliceEval100k.bin")
    parser.add_argument("--positions", type=int, default=20000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ternary-mode", type=str, default="histmatch",
                        choices=["learned", "random", "histmatch"],
                        help="Ternary weights to evaluate: learned (checkpoint), "
                             "random (fresh init), histmatch (per-matrix counts preserved, "
                             "positions shuffled)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Logit scale (default: 1/sqrt(hidden_dim) for discrete checkpoints; "
                             "use 1.0 for stochastic/backprop checkpoints)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sd = ckpt["model_state_dict"]

    model = StochasticTransformerModel(
        vocab_size=cfg["vocab_size"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"], max_seq_len=cfg["max_seq_len"],
        threshold=cfg["threshold"],
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    if args.scale is not None:
        scale = args.scale
    else:
        scale = 1.0 / math.sqrt(cfg["hidden_dim"])

    tokens = load_eval_tokens(args.slice, args.positions)
    rule = cfg.get("rule", cfg.get("mode", "?"))
    print(f"Checkpoint step {ckpt.get('step')}: rule={rule} "
          f"| scale={scale:.5f} | slice {args.slice} | {len(tokens)} tokens")

    counts, _ = ternary_distribution(model)
    total = sum(counts.values())
    c_n1 = counts.get(-1, 0); c_0 = counts.get(0, 0); c_p1 = counts.get(1, 0)
    print(f"Learned ternary distribution: "
          f"-1={c_n1/total*100:.1f}% 0={c_0/total*100:.1f}% "
          f"+1={c_p1/total*100:.1f}% (sparsity={c_0/total*100:.1f}%)"
          + (f" outliers(+-2)={counts.get(2, 0)/total*100:.1f}%" if counts.get(2) else ""))

    ce_learned = eval_ce(model, tokens, args.block_size, args.positions, scale)

    if args.ternary_mode == "learned":
        model_target = copy.deepcopy(model)
        mode_name = "LEARNED"
    elif args.ternary_mode == "random":
        model_target = copy.deepcopy(model)
        randomize_ternary(model_target, seed=args.seed)
        mode_name = "RANDOM"
    else:
        model_target = copy.deepcopy(model)
        histmatch_ternary(model_target, seed=args.seed)
        mode_name = "HISTMATCH-RANDOM"
    ce_target = eval_ce(model_target, tokens, args.block_size, args.positions, scale)

    delta = ce_target - ce_learned
    print(f"\nTernary {mode_name} : CE {ce_target:.4f}")
    print(f"Ternary LEARNED : CE {ce_learned:.4f}")
    print(f"Delta ({mode_name.lower()} - learned): {delta:+.4f} nats")


if __name__ == "__main__":
    main()
