"""Export Tetra model weights to binary format for C++ inference.

Weight encoding (2 bits per ternary weight):
    00 = -1
    01 =  0
    10 = +1

Pack 4 weights per byte, MSB first.

Binary file format (v2):
    Header (64 bytes):
        [0:4]   magic: "TETR" (4 bytes)
        [4:8]   version: uint32 = 2
        [8:12]  vocab_size: uint32
        [12:16] hidden_dim: uint32
        [16:20] num_layers: uint32
        [20:24] num_heads: uint32
        [24:28] ffn_dim: uint32
        [28:32] max_seq_len: uint32
        [32:40] num_ternary_params: uint64
        [40:48] num_fp32_params: uint64 (embeddings + lm_head + norms)
        [48:64] reserved

    Sections (sequential):
        For each ternary weight:
            [4 bytes] name_length (uint32)
            [N bytes] name (utf-8)
            [4 bytes] rows (uint32)
            [4 bytes] cols (uint32)
            [4 bytes] alpha (float32) — absmean scale factor α = mean(|W_latent|)
            [ceil(rows*cols/4) bytes] packed ternary weights

        For each fp32 weight:
            [4 bytes] name_length (uint32)
            [N bytes] name (utf-8)
            [4 bytes] rows (uint32)
            [4 bytes] cols (uint32)
            [rows*cols*4 bytes] fp32 weights (little-endian)

Usage:
    python inference/export_model.py checkpoints_local/checkpoint_000200.pt
"""
import sys
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ternary_llm.transformer import TernaryTransformerModel


TERNARY_ENCODING = {-1: 0b00, 0: 0b01, 1: 0b10}

# Which parameters are ternary vs fp32
TERNARY_PARAM_NAMES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # ffn
]

FP32_PARAM_NAMES = [
    "token_embedding", "pos_embedding",  # embeddings
    "attn_norm", "ffn_norm", "norm",     # norms
    "lm_head",                            # lm head (tied to embedding)
]


def pack_ternary(weights: torch.Tensor) -> bytes:
    """Pack ternary weights {-1, 0, 1} into 2-bit encoding."""
    flat = weights.cpu().numpy().astype(np.int8).flatten()
    n = len(flat)
    packed_len = (n + 3) // 4
    packed = np.zeros(packed_len, dtype=np.uint8)

    for i in range(n):
        val = int(flat[i])
        if val not in TERNARY_ENCODING:
            # Clamp to nearest ternary value
            if val < 0:
                val = -1
            elif val > 0:
                val = 1
            else:
                val = 0
        encoded = TERNARY_ENCODING[val]
        byte_idx = i // 4
        bit_shift = 6 - (i % 4) * 2  # MSB first
        packed[byte_idx] |= encoded << bit_shift

    return packed.tobytes()


def count_params(model: TernaryTransformerModel) -> tuple[int, int]:
    """Count ternary vs fp32 parameters."""
    ternary_count = 0
    fp32_count = 0

    for name, param in model.named_parameters():
        is_ternary = any(t in name for t in TERNARY_PARAM_NAMES)
        if is_ternary:
            ternary_count += param.numel()
        else:
            fp32_count += param.numel()

    return ternary_count, fp32_count


def export_model(model: TernaryTransformerModel, output_path: str):
    """Export model weights to binary format."""
    model.eval()

    ternary_count, fp32_count = count_params(model)
    print(f"Ternary params: {ternary_count:,} ({ternary_count * 2 / 8 / 1024:.1f} KB packed)")
    print(f"FP32 params: {fp32_count:,} ({fp32_count * 4 / 1024:.1f} KB)")
    print(f"Total: {ternary_count + fp32_count:,}")

    with open(output_path, "wb") as f:
        # Header (64 bytes)
        vocab_size = model.token_embedding.num_embeddings
        hidden_dim = model.hidden_dim
        num_layers = len(model.layers)
        num_heads = model.layers[0].attn.num_heads
        ffn_dim = model.layers[0].ffn.gate_proj.out_features
        max_seq_len = model.max_seq_len

        header = struct.pack(
            "<4sIIIIIIIQQ16s",
            b"TETR",              # magic
            2,                     # version (v2: adds per-matrix alpha scale)
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            ffn_dim,
            max_seq_len,
            ternary_count,
            fp32_count,
            b"\x00" * 16,         # reserved
        )
        f.write(header)

        # Write ternary weights
        for name, param in model.named_parameters():
            is_ternary = any(t in name for t in TERNARY_PARAM_NAMES)
            if not is_ternary:
                continue

            # Quantize to ternary
            from ternary_llm.quantization import TernaryQuantizer
            w_ternary = TernaryQuantizer.apply(param.data)
            w_ternary = w_ternary.to(torch.int8)

            # Compute alpha = mean(|W_latent|) — absmean scale factor
            alpha = param.data.abs().mean().item()

            shape = list(w_ternary.shape)
            name_bytes = name.encode("utf-8")

            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<HH", shape[0], shape[1]))
            f.write(struct.pack("<f", alpha))  # v2: per-matrix alpha
            f.write(pack_ternary(w_ternary))

        # Write fp32 weights (skip lm_head — tied to token_embedding)
        for name, param in model.named_parameters():
            is_ternary = any(t in name for t in TERNARY_PARAM_NAMES)
            if is_ternary:
                continue
            # Skip lm_head — weight is tied to token_embedding
            if name == "lm_head.weight":
                continue

            w_fp32 = param.data.float().cpu().numpy().flatten()
            ndim = len(param.shape)
            shape = list(param.shape)
            # Pad shape to 4 dims for uniform loading
            while len(shape) < 4:
                shape.append(1)

            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            # ndim + shape (up to 4 dims, padded with 1s)
            padded = shape + [1] * (4 - len(shape))
            f.write(struct.pack("<B", ndim))
            f.write(struct.pack("<4I", *padded))
            f.write(w_fp32.tobytes())

    file_size = Path(output_path).stat().st_size
    print(f"\nExported to {output_path} ({file_size / 1024:.1f} KB)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_model.py <checkpoint.pt> [output.bin]")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "tetra_model.bin"

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

    export_model(model, output_path)


if __name__ == "__main__":
    main()
