"""Perplexity benchmark for Tetra — compare Python vs C++ inference.

Computes perplexity on TinyStories validation set:
    PPL = exp(average cross-entropy loss)

Usage:
    python inference/benchmark_ppl.py
    python inference/benchmark_ppl.py --checkpoint checkpoints/checkpoint_010000.pt
    python inference/benchmark_ppl.py --binary tetra_model.bin --compare-cpp --samples 50
"""
import sys
import math
import subprocess
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from ternary_llm.transformer import TernaryTransformerModel
from ternary_llm.data import get_tokenizer_compat, ChunkedDataset
from inference.export_model import export_model


def compute_ppl_pytorch(model, tokens, block_size=128, max_batches=200):
    """Compute perplexity using PyTorch model on token chunks."""
    model.eval()
    device = next(model.parameters()).device

    ds = ChunkedDataset(tokens, block_size)
    n_batches = min(len(ds) // 1, max_batches)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, min(n_batches * block_size, len(tokens) - block_size), block_size):
            chunk = tokens[i:i + block_size + 1]
            x = torch.tensor(np.array([chunk[:-1]]), dtype=torch.long, device=device)
            y = torch.tensor(np.array([chunk[1:]]), dtype=torch.long, device=device)

            logits, loss, _ = model(x, y)
            # loss is already cross-entropy averaged over seq_len
            total_loss += loss.item() * block_size
            total_tokens += block_size

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return ppl, avg_loss, total_tokens


def get_pytorch_logits(model, tokens):
    """Get last-position logits from PyTorch for a 1-D token array."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor(np.array([tokens]), dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, _ = model(x, None)
    return logits[0, -1, :].cpu().numpy().astype(np.float64)


def get_cpp_logits(binary_path, model_path, tokens):
    """Get last-position logits from C++ inference for a token list."""
    token_str = ",".join(str(t) for t in tokens)
    result = subprocess.run(
        [binary_path, model_path, token_str, "0"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"C++ binary failed: {result.stderr}")
    parts = result.stdout.strip().split()
    logits = np.array([float(p) for p in parts], dtype=np.float64)
    return logits


def main():
    parser = argparse.ArgumentParser(description="Tetra Perplexity Benchmark")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--binary", type=str, default=None, help="Path to tetra.exe for C++ comparison")
    parser.add_argument("--compare-cpp", action="store_true", help="Compare C++ vs Python logits")
    parser.add_argument("--samples", type=int, default=50, help="Number of logit comparison samples")
    args = parser.parse_args()

    print("Tetra Perplexity Benchmark")

    # Load tokenizer
    enc = get_tokenizer_compat(args.tokenizer_dir)
    print(f"Tokenizer vocab: {enc.n_vocab}")

    # Load validation tokens
    bin_path = Path(args.data_dir) / "tinystories.bin"
    if not bin_path.exists():
        print(f"ERROR: {bin_path} not found")
        return

    tokens = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
    # Use last 5% as validation (matches training val_split)
    val_start = int(len(tokens) * 0.95)
    val_tokens = tokens[val_start:]
    print(f"Validation tokens: {len(val_tokens):,}")

    # Load PyTorch model
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if ckpt_path is None:
        # Auto-find latest checkpoint
        ckpt_dir = Path("checkpoints")
        ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            print("ERROR: No checkpoint found. Use --checkpoint to specify one.")
            return
        ckpt_path = ckpts[-1]

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    model = TernaryTransformerModel(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        max_seq_len=config["max_seq_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = sum(
        p.numel() for name, p in model.named_parameters()
        if any(t in name for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"])
    )
    print(f"Model params: {total_params:,} (ternary: {ternary_params:,})")

    # Compute PyTorch perplexity
    print(f"Computing PyTorch perplexity (block_size={args.block_size}, max_batches={args.max_batches})...")
    ppl, avg_loss, n_tokens = compute_ppl_pytorch(
        model, val_tokens, args.block_size, args.max_batches
    )
    print(f"\nPyTorch PPL:     {ppl:.4f}")
    print(f"PyTorch Loss:    {avg_loss:.4f}")
    print(f"Tokens evaluated: {n_tokens:,}")

    # C++ comparison
    if args.compare_cpp:
        if args.binary is None:
            print("ERROR: --binary required with --compare-cpp")
            return
        binary_path = Path(args.binary)
        if not binary_path.exists():
            print(f"ERROR: Binary not found: {binary_path}")
            return

        print(f"\n--- C++ vs PyTorch Logit Comparison ({args.samples} samples) ---")

        # Export model to temp binary
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            model_bin_path = f.name
        print(f"Exporting model to {model_bin_path}...")
        export_model(model, model_bin_path)

        # Collect logit pairs from random chunks
        bs = args.block_size
        max_offset = len(val_tokens) - bs - 1
        rng = np.random.RandomState(42)
        offsets = rng.randint(0, max_offset, size=args.samples)

        diffs = []
        cos_sims = []
        for sidx, offset in enumerate(offsets):
            chunk = val_tokens[offset:offset + bs + 1]
            input_tokens = chunk[:-1].tolist()

            # PyTorch
            py_logits = get_pytorch_logits(model, input_tokens)

            # C++
            try:
                cpp_logits = get_cpp_logits(str(binary_path), model_bin_path, input_tokens)
            except Exception as e:
                print(f"  [{sidx + 1}/{args.samples}] C++ error at offset {offset}: {e}")
                continue

            if len(py_logits) != len(cpp_logits):
                print(f"  [{sidx + 1}/{args.samples}] Size mismatch: Py={len(py_logits)} vs C++={len(cpp_logits)}")
                continue

            # Compare
            diff = np.abs(py_logits - cpp_logits)
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())
            rmse = float(np.sqrt((diff ** 2).mean()))

            py_mean = float(py_logits.mean())
            py_std = float(py_logits.std())
            cpp_mean = float(cpp_logits.mean())
            cpp_std = float(cpp_logits.std())

            cos_sim = float(np.dot(py_logits, cpp_logits) /
                          (np.linalg.norm(py_logits) * np.linalg.norm(cpp_logits) + 1e-30))
            cos_sims.append(cos_sim)
            diffs.append((max_diff, mean_diff, rmse))

            if (sidx + 1) % 10 == 0 or sidx == 0:
                print(f"  [{sidx + 1}/{args.samples}] max_diff={max_diff:.3e} mean_diff={mean_diff:.3e} cos_sim={cos_sim:.6f}")

        # Summary
        if diffs:
            max_diffs = [d[0] for d in diffs]
            mean_diffs = [d[1] for d in diffs]
            rmses = [d[2] for d in diffs]
            print(f"\n  Summary across {len(diffs)} samples:")
            print(f"    Max difference  - avg: {np.mean(max_diffs):.3e}  max: {np.max(max_diffs):.3e}")
            print(f"    Mean difference - avg: {np.mean(mean_diffs):.3e}  max: {np.max(mean_diffs):.3e}")
            print(f"    RMSE           - avg: {np.mean(rmses):.3e}  max: {np.max(rmses):.3e}")
            print(f"    Cosine similarity - min: {np.min(cos_sims):.6f}  avg: {np.mean(cos_sims):.6f}")

            # If C++ and PyTorch produce the same logits, perplexity should match
            all_close = np.max(max_diffs) < 1e-3
            if all_close:
                print(f"\n  [OK] C++ and PyTorch logits match (max_diff < 1e-3)")
                print(f"  => C++ perplexity approx {ppl:.4f}")
            else:
                print(f"\n  [WARN] C++ and PyTorch logits differ")
                print(f"  => PyTorch PPL = {ppl:.4f}, C++ PPL may differ")
        else:
            print("\n  No valid comparisons completed")

        # Cleanup
        Path(model_bin_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
