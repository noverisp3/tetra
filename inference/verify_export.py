"""Verify exported Tetra binary matches PyTorch model output."""
import sys
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ternary_llm.transformer import TernaryTransformerModel
from ternary_llm.quantization import TernaryQuantizer


TERNARY_DECODING = {-1: 0b00, 0: 0b01, 1: 0b10}  # reverse of encoding
TERNARY_REVERSE = {0b00: -1, 0b01: 0, 0b10: 1, 0b11: 0}


def unpack_ternary(data: bytes, n: int) -> np.ndarray:
    """Unpack 2-bit ternary encoding to int8 array."""
    raw = np.frombuffer(data, dtype=np.uint8)
    result = np.zeros(n, dtype=np.int8)
    for i in range(n):
        byte_idx = i // 4
        bit_shift = 6 - (i % 4) * 2
        encoded = (raw[byte_idx] >> bit_shift) & 0b11
        result[i] = TERNARY_REVERSE.get(encoded, 0)
    return result


def load_binary_model(path: str) -> dict:
    """Load exported Tetra binary model into a dict of numpy arrays."""
    weights = {}
    with open(path, "rb") as f:
        # Header (64 bytes)
        header = f.read(64)
        magic, version, vocab_size, hidden_dim, num_layers, num_heads, ffn_dim, \
        max_seq_len, ternary_count, fp32_count = \
            struct.unpack("<4sIIIIIIIQQ", header[:48])
        assert magic == b"TETR", f"Bad magic: {magic}"
        print(f"Header: v{version} vocab={vocab_size} hidden={hidden_dim} "
          f"layers={num_layers} heads={num_heads} ffn={ffn_dim} seq={max_seq_len}")

        # Ternary weights
        for _ in range(6 * num_layers):  # q,k,v,o,gate_up,down per layer
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            rows, cols = struct.unpack("<HH", f.read(4))
            packed_size = (rows * cols + 3) // 4
            packed_data = f.read(packed_size)
            arr = unpack_ternary(packed_data, rows * cols).reshape(rows, cols)
            weights[name] = arr

        # FP32 weights
        total_fp32_tensors = 0
        while True:
            name_len_data = f.read(4)
            if len(name_len_data) < 4:
                break
            name_len = struct.unpack("<I", name_len_data)[0]
            name = f.read(name_len).decode("utf-8")
            ndim = struct.unpack("<B", f.read(1))[0]
            padded = struct.unpack("<4I", f.read(16))
            shape = list(padded[:ndim])
            n_elements = 1
            for s in shape:
                n_elements *= s
            raw_data = f.read(n_elements * 4)
            arr = np.frombuffer(raw_data, dtype=np.float32).reshape(shape)
            weights[name] = arr
            total_fp32_tensors += 1

        print(f"Loaded {len(weights)} tensors ({6*num_layers} ternary, {total_fp32_tensors} fp32)")
    return weights


def compare(pytorch_model, binary_weights):
    """Compare PyTorch model weights with binary-loaded weights."""
    print("\n--- Weight comparison ---")
    max_diff = 0.0
    for name, param in pytorch_model.named_parameters():
        is_ternary = any(t in name for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"])
        if name == "lm_head.weight":
            continue

        if name not in binary_weights:
            print(f"  MISSING: {name}")
            continue

        if is_ternary:
            pytorch_w = TernaryQuantizer.apply(param.data).cpu().numpy()
        else:
            pytorch_w = param.data.float().cpu().numpy()

        bin_w = binary_weights[name]
        if pytorch_w.shape != bin_w.shape:
            print(f"  SHAPE MISMATCH: {name} pytorch={pytorch_w.shape} binary={bin_w.shape}")
            continue

        diff = np.abs(pytorch_w - bin_w).max()
        max_diff = max(max_diff, diff)
        if diff > 1e-5:
            print(f"  {name}: max_diff={diff:.6f}")
        else:
            print(f"  {name}: OK (exact match)")

    print(f"\nOverall max diff: {max_diff:.6f}")


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None
    bin_path = sys.argv[2] if len(sys.argv) > 2 else "tetra_model.bin"

    print("Loading PyTorch model...")
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

    print("Loading binary model...")
    binary_weights = load_binary_model(bin_path)

    compare(model, binary_weights)


if __name__ == "__main__":
    main()
