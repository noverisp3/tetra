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
    python eval_ternary_ablation.py [--checkpoint ...] [--slice ...] [--positions N]
"""
import argparse
import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F

from ternary_llm.transformer import StochasticTransformerModel
from ternary_llm.layers import StochasticTernaryLinear, init_ternary_weight


def load_eval_tokens(path: str, n: int) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16)[:n]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_discrete_c2/checkpoint_000250.pt")
    parser.add_argument("--slice", type=str,
                        default="checkpoints_discrete_c2/sliceEval100k.bin")
    parser.add_argument("--positions", type=int, default=20000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
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
    model.load_state_dict(sd)
    model.eval()
    scale = 1.0 / math.sqrt(cfg["hidden_dim"])

    tokens = load_eval_tokens(args.slice, args.positions)
    print(f"Checkpoint step {ckpt.get('step')}: rule={cfg['rule']} "
          f"| scale={scale:.5f} | slice {args.slice} | {len(tokens)} tokens")

    ce_learned = eval_ce(model, tokens, args.block_size, args.positions, scale)

    model_random = copy.deepcopy(model)
    randomize_ternary(model_random, seed=args.seed)
    ce_random = eval_ce(model_random, tokens, args.block_size, args.positions, scale)

    delta = ce_random - ce_learned
    print(f"\nTernary LEARNED : CE {ce_learned:.4f}")
    print(f"Ternary RANDOM  : CE {ce_random:.4f}")
    print(f"Delta (random - learned): {delta:+.4f} nats "
          f"({'RANDOM BETTER' if delta < 0 else 'LEARNED BETTER'})")
    verdict = ("ternary is decorative (no structural benefit)" if abs(delta) < 0.05
               else "ternary carries learned structure")
    print(f"Interpretation: {verdict}")


if __name__ == "__main__":
    main()
