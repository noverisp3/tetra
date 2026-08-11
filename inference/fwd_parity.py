"""Forward logits parity check: PyTorch model vs C++ inference binary.

Catches the bug class that silently inflated/deflated eval numbers before
(attention-score clamp, stale logit-scale metadata, quantizer drift, cache
layout bugs): every exported model must reproduce the checkpoint's logits.

Usage:
    python inference/fwd_parity.py checkpoints/checkpoint_010000.pt
    python inference/fwd_parity.py examples/tiny/8.5M_15K_STE.pt --samples 16
    python inference/fwd_parity.py checkpoints/exp15_mla.bin --ckpt <mla.pt>

Exit code 0 iff all samples pass max-diff and cosine thresholds.
"""
import sys
import math
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ternary_llm.transformer import (
    TernaryTransformerModel, StochasticTransformerModel, StochasticMLAModel,
)
from inference.export_model import export_model


def build_model(ckpt: dict) -> torch.nn.Module:
    config = ckpt["config"]
    sd = ckpt["model_state_dict"]
    mode = config.get("mode", "ste")
    mla = config.get("mla", False) or any("kv_down_proj" in k for k in sd)
    if mode == "stochastic" and mla:
        kv_latent_dim = config.get("kv_latent_dim", None)
        rope_per_head = config.get("rope_per_head", None)
        hidden_dim = config.get("hidden_dim", 256)
        num_heads = config.get("num_heads", 4)
        for k, v in sd.items():
            if k.endswith("kv_down_proj.packed_weights") and kv_latent_dim is None:
                kv_latent_dim = v.numel() * 4 // hidden_dim
            if k.endswith("q_rope_proj.packed_weights") and rope_per_head is None:
                rope_dim = v.numel() * 4 // hidden_dim
                rope_per_head = rope_dim // num_heads
        model = StochasticMLAModel(
            vocab_size=config["vocab_size"], hidden_dim=hidden_dim,
            num_layers=config["num_layers"], num_heads=num_heads,
            ffn_dim=config["ffn_dim"], max_seq_len=config["max_seq_len"],
            scale=config.get("ternary_scale", 1.0),
            threshold=config.get("threshold", None),
            int8=config.get("int8", False),
            topk=config.get("topk", 1.0),
            group_size=config.get("group_size", 0),
            kv_latent_dim=kv_latent_dim,
            rope_per_head=rope_per_head,
        )
    elif mode == "stochastic":
        model = StochasticTransformerModel(
            vocab_size=config["vocab_size"], hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"], num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"], max_seq_len=config["max_seq_len"],
            scale=config.get("ternary_scale", 1.0),
            threshold=config.get("threshold", None),
            int8=config.get("int8", False),
            topk=config.get("topk", 1.0),
            group_size=config.get("group_size", 0),
        )
    else:
        model = TernaryTransformerModel(
            vocab_size=config["vocab_size"], hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"], num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"], max_seq_len=config["max_seq_len"],
        )
    model.load_state_dict(sd, strict=False)
    model.eval()
    if mode != "stochastic":
        for m in model.modules():
            if hasattr(m, "set_v8_forward"):
                m.set_v8_forward(True)
    return model


def torch_logits(model: torch.nn.Module, tokens: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    x = torch.tensor(tokens[None, :], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, _ = model(x, None)
    return logits[0, -1, :].cpu().numpy().astype(np.float64)


def cpp_logits(binary: str, model_bin: str, tokens: np.ndarray) -> np.ndarray:
    token_str = ",".join(str(int(t)) for t in tokens)
    result = subprocess.run(
        [binary, model_bin, token_str, "0"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"C++ binary failed: {result.stderr.strip()}")
    parts = result.stdout.strip().split()
    return np.array([float(p) for p in parts], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description="Forward logits parity: torch vs C++")
    ap.add_argument("checkpoint", help=".pt checkpoint (architecture + weights source)")
    ap.add_argument("--bin", default=None, help="existing export; skip re-exporting")
    ap.add_argument("--binary", default=str(Path(__file__).parent / "tetra_avx2.exe"))
    ap.add_argument("--tokens", default="examples/discrete/sliceEval100k.bin")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--len", type=int, default=32)
    ap.add_argument("--tol", type=float, default=0.25, help="max |logit diff| threshold")
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        print(f"ERROR: binary not found: {binary}")
        return 1

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    mode = config.get("mode", "ste")
    print(f"Checkpoint: {args.checkpoint} (mode={mode})")

    model = build_model(ckpt)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total:,}")

    if args.bin is not None:
        bin_path = args.bin
        print(f"Using existing export: {bin_path}")
    else:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            bin_path = f.name
        if mode == "stochastic":
            export_model(model, bin_path, mode=mode, v7=True)
        else:
            export_model(model, bin_path, mode=mode, v8=True, v8_k=0)
        print(f"Exported to temp: {bin_path}")

    tokens = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    max_offset = len(tokens) - args.len - 1
    rng = np.random.RandomState(args.seed)
    offsets = rng.randint(0, max_offset, size=args.samples)

    print(f"\n{'sample':>6} {'off':>8} {'max|diff|':>10} {'mean|diff|':>10} {'cos':>8} {'top1':>6}")
    overall_max = 0.0
    all_cos = []
    top1_ok = 0
    for sidx, off in enumerate(offsets):
        seq = tokens[off:off + args.len]
        py = torch_logits(model, seq)
        cpp = cpp_logits(str(binary), bin_path, seq)
        if len(cpp) != len(py):
            print(f"[{sidx + 1}] C++ returned {len(cpp)} logits, torch {len(py)}")
            return 1
        diff = np.abs(py - cpp)
        mx = float(diff.max())
        mn = float(diff.mean())
        denom = max(float(np.linalg.norm(py)), 1e-30)
        cos = float(np.dot(py, cpp) / (denom * max(float(np.linalg.norm(cpp)), 1e-30)))
        t1 = int(np.argmax(py) == np.argmax(cpp))
        all_cos.append(cos)
        top1_ok += t1
        overall_max = max(overall_max, mx)
        print(f"{sidx + 1:>6} {off:>8} {mx:>10.6f} {mn:>10.6f} {cos:>8.6f} {t1:>6}")

    mean_cos = float(np.mean(all_cos))
    print(f"\nOverall: max|diff|={overall_max:.6f}  mean cos={mean_cos:.6f}  "
          f"top1 agreement {top1_ok}/{args.samples}")
    ok = overall_max <= args.tol and mean_cos >= args.min_cos
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())