"""Exp 11 diagnostic: outlier magnitude distribution across v8-forward checkpoints.

For every latent_weights matrix: x_n = W/Delta (scale x mean|W|, per-tensor),
then report over the |x_n| > 1.5 outlier set:
  - outlier share (count / total weights)
  - mean |x_n| of outliers
  - linger share: |x_n| in (1.5, 2.0)  (the band v8-reg attacks)
  - true-magnitude share: |x_n| >= 2.0
  - max |x_n|

Usage: python scripts/diagnose_v8_reg.py ck1.pt [ck2.pt ...]
"""
import sys
import torch


def stats(path: str) -> dict:
    ck = torch.load(path, map_location="cpu")
    config = ck.get("config", {})
    scale = config.get("ternary_scale", 1.0)
    per_channel = config.get("per_channel", False)
    n_out = n_linger = n_true = 0
    s_abs = 0.0
    s_max = 0.0
    for name, p in ck["model_state_dict"].items():
        if "latent_weights" not in name or p.ndim != 2:
            continue
        p = p.float()
        if per_channel:
            delta = p.abs().mean(dim=1, keepdim=True).clamp(min=1e-6) * scale
        else:
            delta = p.abs().mean().clamp(min=1e-6) * scale
        x_n = p / delta
        mag = x_n.abs()
        mask = mag > 1.5
        n_out += mask.sum().item()
        s_abs += mag[mask].sum().item()
        s_max = max(s_max, mag.max().item())
        n_linger += ((mag > 1.5) & (mag < 2.0)).sum().item()
        n_true += (mag >= 2.0).sum().item()
    total_w = sum(p.numel() for name, p in ck["model_state_dict"].items()
                  if "latent_weights" in name and p.ndim == 2)
    return {
        "outlier_share": n_out / max(total_w, 1),
        "mean_abs": s_abs / max(n_out, 1),
        "linger_15_20": n_linger / max(n_out, 1),
        "true_ge_20": n_true / max(n_out, 1),
        "max_abs": s_max,
    }


if __name__ == "__main__":
    print(f"{'checkpoint':<52} {'outlier%':>9} {'mean|x|':>8} {'15-2.0':>7} {'>=2.0':>7} {'max':>6}")
    for path in sys.argv[1:]:
        s = stats(path)
        print(f"{path:<52} {s['outlier_share']*100:8.2f}% {s['mean_abs']:8.3f} "
              f"{s['linger_15_20']*100:6.1f}% {s['true_ge_20']*100:6.1f}% {s['max_abs']:6.3f}")
