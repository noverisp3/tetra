"""Speed benchmark for Tetra model scaling.

Tests training speed (forward + backward + optimizer step) for
different model sizes on real data.

Usage:
    python benchmark_speed.py
    python benchmark_speed.py --steps 10 --batch-size 8
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from ternary_llm.transformer import TernaryTransformerModel
from ternary_llm.data import ChunkedDataset


CONFIGS = {
    "tiny": dict(hidden_dim=128, num_layers=4, num_heads=4, ffn_dim=512),
    "medium": dict(hidden_dim=512, num_layers=12, num_heads=8, ffn_dim=2048),
    "large": dict(hidden_dim=768, num_layers=12, num_heads=12, ffn_dim=2048),
}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    ternary = sum(p.numel() for n, p in model.named_parameters() if "latent_weights" in n)
    fp32 = total - ternary
    return total, ternary, fp32


def resolve_device():
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def benchmark_config(name, cfg, tokens, block_size, batch_size, steps, device):
    print(f"\n{'='*60}")
    print(f"  {name.upper()} MODEL")
    print(f"{'='*60}")

    model = TernaryTransformerModel(
        vocab_size=8192,
        max_seq_len=block_size,
        **cfg,
    )
    total, ternary, fp32 = count_params(model)
    print(f"  Params: {total:,} (ternary: {ternary:,} / {ternary*2/8/1024:.0f} KB packed, fp32: {fp32:,} / {fp32*4/1024:.0f} KB)")

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    ds = ChunkedDataset(tokens, block_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    data_iter = iter(loader)

    # Warmup (1 step)
    model.train()
    x, y = next(data_iter)
    x, y = x.to(device), y.to(device)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Benchmark
    step_times = []
    losses = []

    for step in range(steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        t0 = time.perf_counter()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type != "privateuseone" and str(device) != "directml":
            if hasattr(torch, "cuda") and torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        step_times.append(elapsed)
        losses.append(loss.item())
        print(f"  Step {step+1:2d}/{steps}: {elapsed*1000:7.1f} ms | loss {loss.item():.4f}")

    avg_ms = sum(step_times) / len(step_times)
    tokens_per_sec = (batch_size * block_size) / avg_ms * 1000

    print(f"\n  Average: {avg_ms:.1f} ms/step | {tokens_per_sec:,.0f} tokens/sec")
    print(f"  Loss range: {min(losses):.4f} - {max(losses):.4f}")

    return {
        "name": name,
        "params": total,
        "ternary": ternary,
        "fp32": fp32,
        "avg_ms": avg_ms,
        "tokens_per_sec": tokens_per_sec,
        "loss_last": losses[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Tetra Speed Benchmark")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--configs", nargs="+", default=["tiny", "medium", "large"])
    args = parser.parse_args()

    print("=" * 60)
    print("  Tetra Training Speed Benchmark")
    print("=" * 60)
    print(f"  Steps: {args.steps} | Batch: {args.batch_size} | Block: {args.block_size}")

    device = resolve_device()
    print(f"  Device: {device}")

    bin_path = Path("data") / "tinystories.bin"
    if not bin_path.exists():
        print(f"  ERROR: {bin_path} not found")
        return

    tokens = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
    print(f"  Tokens: {len(tokens):,}")

    results = []
    for name in args.configs:
        if name not in CONFIGS:
            print(f"  Unknown config: {name}")
            continue
        r = benchmark_config(
            name, CONFIGS[name], tokens,
            args.block_size, args.batch_size, args.steps, device,
        )
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<10} {'Params':>12} {'ms/step':>10} {'tok/s':>10} {'Loss':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        print(f"  {r['name']:<10} {r['params']:>12,} {r['avg_ms']:>9.1f}ms {r['tokens_per_sec']:>10,.0f} {r['loss_last']:>8.4f}")

    # Extrapolate full training time
    print(f"\n  Estimated time for 10K steps:")
    for r in results:
        hours = r['avg_ms'] * 10000 / 1000 / 3600
        print(f"    {r['name']:<10}: {hours:.1f} hours")


if __name__ == "__main__":
    main()
