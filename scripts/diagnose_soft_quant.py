"""Exp 8 diagnostic: dead-zone / plateau / outlier occupancy per layer.

Loads two STE checkpoints (baseline vs soft-quant hybrid) and reports the
fraction of latent weights in each ternary quantization region:
  |x_n| < 0.5      -> dead zone (forward quantizes to 0)
  0.5 <= |x_n| < 1.5 -> plateau (forward value +-1)
  |x_n| >= 1.5     -> outlier (forward value +-2)

Usage: python scripts/diagnose_soft_quant.py <ckpt_a> <ckpt_b>
"""
import sys
from pathlib import Path

import torch

TENSOR_NAMES = {
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
    "ffn.up_proj", "ffn.down_proj",
}


def analyze(path: str, label: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    s = ck["model_state_dict"]
    scale = ck["config"]["ternary_scale"]
    per_channel = ck["config"]["per_channel"]

    rows = []
    totals = {"dz": 0, "p1": 0, "p2": 0}
    for k, v in s.items():
        if "latent_weights" not in k:
            continue
        w = v.float()
        if per_channel:
            delta = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-6) * scale
        else:
            delta = w.abs().mean().clamp(min=1e-6) * scale
        xn = (w / delta).abs()
        n = xn.numel()
        dz = (xn < 0.5).float().mean().item()
        p1 = ((xn >= 0.5) & (xn < 1.5)).float().mean().item()
        p2 = (xn >= 1.5).float().mean().item()
        name = k.replace(".latent_weights", "").replace("layers.", "L")
        rows.append((name, dz, p1, p2))
        totals["dz"] += n * dz
        totals["p1"] += n * p1
        totals["p2"] += n * p2

    total = totals["dz"] + totals["p1"] + totals["p2"]
    print(f"\n=== {label} ({path}) ===")
    print(f"{'tensor':<18} {'dead<0.5':>9} {'plateau 0.5-1.5':>16} {'outlier>=1.5':>12}")
    for name, dz, p1, p2 in rows:
        print(f"{name:<18} {dz*100:>8.1f}% {p1*100:>15.1f}% {p2*100:>11.1f}%")
    print(f"{'TOTAL':<18} {totals['dz']/total*100:>8.1f}% {totals['p1']/total*100:>15.1f}% {totals['p2']/total*100:>11.1f}%")

    # per-row spread: fraction of rows that are pure dead zone (rank-collapse symptom)
    dead_rows = {}
    for k, v in s.items():
        if "latent_weights" not in k:
            continue
        w = v.float()
        delta = w.abs().mean().clamp(min=1e-6) * scale
        xn = (w / delta).abs()
        row_dz = (xn < 0.5).float().mean(dim=1)
        name = k.replace(".latent_weights", "").replace("layers.", "L")
        dead_rows[name] = (row_dz > 0.95).float().mean().item()
    print("\n  rows with >95% dead-zone weights (near-zero rows):")
    for name, frac in dead_rows.items():
        if frac > 0:
            print(f"    {name:<18} {frac*100:.1f}% of rows")


if __name__ == "__main__":
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
    if len(sys.argv) > 3:
        analyze(sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else sys.argv[3])
