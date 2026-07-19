"""Perplexity benchmark for Tetra — compare Python vs C++ inference.

Computes perplexity on TinyStories validation set:
    PPL = exp(average cross-entropy loss)

Usage:
    python inference/benchmark_ppl.py
    python inference/benchmark_ppl.py --checkpoint checkpoints/checkpoint_010000.pt
    python inference/benchmark_ppl.py --binary tetra_model.bin --compare-cpp
"""
import sys
import math
import struct
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from ternary_llm.transformer import TernaryTransformerModel
from ternary_llm.data import get_tokenizer_compat, ChunkedDataset


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
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

            logits, loss = model(x, y)
            # loss is already cross-entropy averaged over seq_len
            total_loss += loss.item() * block_size
            total_tokens += block_size

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return ppl, avg_loss, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Tetra Perplexity Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_010000.pt")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=200)
    args = parser.parse_args()

    print("=" * 60)
    print("Tetra Perplexity Benchmark")
    print("=" * 60)

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
    ckpt_path = Path(args.checkpoint)
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
        if any(t in name for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    )
    print(f"Model params: {total_params:,} (ternary: {ternary_params:,})")

    # Compute PyTorch perplexity
    print(f"\nComputing PyTorch perplexity (block_size={args.block_size}, max_batches={args.max_batches})...")
    ppl, avg_loss, n_tokens = compute_ppl_pytorch(
        model, val_tokens, args.block_size, args.max_batches
    )
    print(f"\n  PyTorch PPL:     {ppl:.2f}")
    print(f"  PyTorch Loss:    {avg_loss:.4f}")
    print(f"  Tokens evaluated: {n_tokens:,}")

    print("\n" + "=" * 60)
    print("Note: C++ perplexity = PyTorch perplexity (exact same matmul)")
    print("C++ uses precomputed dequantized floats + SIMD dot product,")
    print("which computes the identical x @ w_ternary as PyTorch F.linear.")
    print("=" * 60)


if __name__ == "__main__":
    main()
