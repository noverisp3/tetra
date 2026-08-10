"""Evaluate CE of an STE checkpoint on a token window (torch, same path as train.py).

Usage: python tests/eval_ckpt_ce.py <checkpoint.pt> --tokens <file.bin> [--offset N]
                                     [--positions N] [--block-size 128]
"""
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--tokens", type=str, default="examples/discrete/sliceEval100k.bin")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--positions", type=int, default=40000)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    from ternary_llm.transformer import TernaryTransformerModel

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    sd = ckpt["model_state_dict"]

    model = TernaryTransformerModel(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        max_seq_len=config["max_seq_len"],
        ternary_scale=config.get("ternary_scale", 1.0),
        per_channel=config.get("per_channel", False),
        group_size=config.get("group_size", 0),
        init_mode=config.get("init_mode", "kaiming"),
    )
    if config.get("lq", False):
        n = 0
        for m in model.modules():
            if hasattr(m, "set_lq"):
                m.set_lq(True)
                n += 1
        print(f"LQ model: {n} codebook layers")
    model.load_state_dict(sd, strict=False)
    model.eval()

    tokens = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    n_blocks = min((len(tokens) - args.offset) // args.block_size,
                   args.positions // args.block_size)
    total = 0.0
    count = 0
    with torch.no_grad():
        for i in range(n_blocks):
            start = args.offset + i * args.block_size
            x = torch.tensor(tokens[start:start + args.block_size], dtype=torch.long).unsqueeze(0)
            y = torch.tensor(tokens[start + 1:start + args.block_size + 1], dtype=torch.long).unsqueeze(0)
            _, loss, _ = model(x, y)
            total += loss.item() * (args.block_size - 1)
            count += args.block_size - 1
    print(f"mean CE: {total / count:.4f} over {count} positions")


if __name__ == "__main__":
    main()
