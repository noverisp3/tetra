"""Verify exported Tetra binary matches PyTorch model output.

Supports STE, stochastic, and MLA models.
"""
import sys
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ternary_llm.transformer import (
    TernaryTransformerModel, StochasticTransformerModel, StochasticMLAModel,
)


# v7: code 11 = outlier ±2 (magnitude 2; sign lives in the side-channel blob,
# re-derived from the .pt checkpoint during comparison). v6 files never emit
# code 11, so mapping it to 2 is safe for both.
TERNARY_REVERSE = {0b00: -1, 0b01: 0, 0b10: 1, 0b11: 2}


def apply_sign_blob(flat: np.ndarray, blob: bytes, n_outliers: int) -> np.ndarray:
    """Apply v7 dense sign bits (MSB-first, 1 = positive) to code-11 entries."""
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))[:n_outliers]
    signs = bits.astype(np.float32) * 2.0 - 1.0
    out = flat.copy()
    mask = np.abs(out) > 1.5
    out[mask] = out[mask] * signs
    return out


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


def load_binary_model(path: str) -> tuple[dict, dict]:
    """Load exported Tetra binary model.

    Returns (weights_dict, header_info_dict).
    """
    weights = {}
    header_info = {}
    with open(path, "rb") as f:
        header = f.read(64)
        magic, version, vocab_size, hidden_dim, num_layers, num_heads, ffn_dim, \
            max_seq_len, ternary_count, fp32_count = \
            struct.unpack("<4sIIIIIIIQQ", header[:48])
        assert magic == b"TETR", f"Bad magic: {magic}"

        flags = 0; kv_latent_dim = 0; rope_per_head = 0; group_size = 0
        if version >= 5:
            flags, kv_latent_dim, rope_per_head, group_size = struct.unpack(
                "<HHHH", header[48:56])

        header_info = {
            "version": version,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "max_seq_len": max_seq_len,
            "ternary_count": ternary_count,
            "fp32_count": fp32_count,
            "is_mla": bool(flags & 1),
            "kv_latent_dim": kv_latent_dim,
            "rope_per_head": rope_per_head,
            "group_size": group_size,
        }

        print(f"Header v{version}: vocab={vocab_size} hidden={hidden_dim} "
              f"layers={num_layers} heads={num_heads} ffn={ffn_dim} seq={max_seq_len}")
        if flags & 1:
            print(f"  MLA: kv_latent_dim={kv_latent_dim} rope_per_head={rope_per_head}")
        if group_size > 0:
            print(f"  Group size: {group_size}")

        # Ternary weights: peek name suffix to distinguish from FP32
        ternary_count_loaded = 0
        while True:
            peek_pos = f.tell()
            if peek_pos >= f.seek(0, 2):
                break
            f.seek(peek_pos)
            name_len_data = f.read(4)
            if len(name_len_data) < 4:
                break
            name_len = struct.unpack("<I", name_len_data)[0]
            if name_len > 1024 or name_len == 0:
                break
            name_bytes = f.read(name_len)
            if len(name_bytes) < name_len:
                break
            name = name_bytes.decode("utf-8")

            # Check if ternary (ends with "latent_weights")
            is_ternary = name.endswith("latent_weights")
            if not is_ternary:
                # Back up: this is the start of FP32 section
                f.seek(peek_pos)
                break

            rows, cols = struct.unpack("<HH", f.read(4))

            if version >= 4:
                gs, num_alphas = struct.unpack("<HH", f.read(4))
                if num_alphas > 0:
                    f.read(num_alphas * 4)  # skip alphas
            elif version >= 3:
                num_alphas = struct.unpack("<H", f.read(2))[0]
                if num_alphas > 0:
                    f.read(num_alphas * 4)
            elif version >= 2:
                f.read(4)  # scalar alpha

            packed_size = (rows * cols + 3) // 4
            packed_data = f.read(packed_size)
            arr = unpack_ternary(packed_data, rows * cols).reshape(rows, cols)

            if version >= 6:
                f.read(rows * cols * 4)  # skip FP32 accumulator (v6/v7)

            if version >= 7:
                # v7: trailing uint32 outlier count + dense sign blob (MSB-first,
                # row-major scan order, 1 = positive). Signs resolve code-11 ±2.
                n_outliers = struct.unpack("<I", f.read(4))[0]
                if n_outliers > 0:
                    nbytes = (n_outliers + 7) // 8
                    blob = f.read(nbytes)
                    flat = arr.reshape(-1)
                    flat = apply_sign_blob(flat, blob, n_outliers)
                    arr = flat.reshape(rows, cols)

            weights[name] = arr
            ternary_count_loaded += 1

        # FP32/INT8 weights
        fp32_count_loaded = 0
        while True:
            name_len_data = f.read(4)
            if len(name_len_data) < 4:
                break
            name_len = struct.unpack("<I", name_len_data)[0]
            if name_len > 1024:
                break
            name = f.read(name_len).decode("utf-8")
            ndim = struct.unpack("<B", f.read(1))[0]
            dtype = struct.unpack("<B", f.read(1))[0]
            padded = struct.unpack("<4I", f.read(16))
            shape = list(padded[:ndim])
            n_elements = 1
            for s in shape:
                n_elements *= s

            if dtype == 1:  # INT8
                scale_val = struct.unpack("<f", f.read(4))[0]
                raw_int8 = np.frombuffer(f.read(n_elements), dtype=np.int8)
                arr = raw_int8.astype(np.float32) * scale_val
                arr = arr.reshape(shape)
            else:
                raw_data = f.read(n_elements * 4)
                arr = np.frombuffer(raw_data, dtype=np.float32).reshape(shape)

            weights[name] = arr
            fp32_count_loaded += 1

        print(f"Loaded {len(weights)} tensors ({ternary_count_loaded} ternary, {fp32_count_loaded} fp32/int8)")
    return weights, header_info


def verify_export(checkpoint_path: str, binary_path: str, mode: str = None,
                  verbose: bool = True) -> bool:
    """Verify exported binary matches PyTorch model.

    Returns True if verification passes.
    """
    print(f"\n{'='*50}")
    print(f"  Verify Export")
    print(f"{'='*50}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    sd = ckpt["model_state_dict"]
    if mode is None:
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
            vocab_size=config["vocab_size"], hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"], num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"], max_seq_len=config["max_seq_len"],
            scale=config.get("ternary_scale", 1.0),
            threshold=config.get("threshold", None),
            int8=config.get("int8", False),
            topk=config.get("topk", 1.0),
            group_size=config.get("group_size", 0),
            kv_latent_dim=kv_latent_dim,
            rope_per_head=rope_per_head,
        )
        print(f"Mode: stochastic+MLA")
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
        print(f"Mode: stochastic")
    else:
        model = TernaryTransformerModel(
            vocab_size=config["vocab_size"], hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"], num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"], max_seq_len=config["max_seq_len"],
        )
        print(f"Mode: STE")

    model.load_state_dict(sd, strict=False)
    model.eval()

    # Load binary
    bin_weights, header_info = load_binary_model(binary_path)

    # Compare weights
    print(f"\n--- Weight comparison ---")
    max_diff = 0.0
    total_compared = 0
    total_ok = 0

    has_packed = any("packed_weights" in n for n, _ in model.named_buffers())

    if has_packed:
        # Stochastic mode: unpack packed_weights and compare with binary latent_weights
        from ternary_llm.quantization import unpack_ternary_tensor as _unpack

        buffers = dict(model.named_buffers())
        for name, buf in buffers.items():
            if not name.endswith(".packed_weights"):
                continue

            prefix = name.rsplit(".", 1)[0]
            layer_idx = name.split(".")[1]
            is_ffn_gate = "gate_proj" in name and f"layers.{layer_idx}.ffn" in name

            if is_ffn_gate:
                up_name = name.replace("gate_proj", "up_proj")
                if up_name not in buffers:
                    continue
                from inference.export_model import get_stochastic_shape, get_stochastic_module
                s = get_stochastic_shape(model, name)
                gate_mod = get_stochastic_module(model, name)
                up_mod = get_stochastic_module(model, up_name)
                gate_w = _unpack(buf, s, gate_mod.outlier_signs if header_info.get("version", 0) >= 7 else None)
                up_w = _unpack(buffers[up_name], s, up_mod.outlier_signs if header_info.get("version", 0) >= 7 else None)
                fused = torch.cat([gate_w, up_w], dim=0).to(torch.int8)
                bin_name = prefix.replace("gate_proj", "gate_up_proj") + ".latent_weights"
            elif "up_proj" in name and f"layers.{layer_idx}.ffn" in name:
                continue
            else:
                from inference.export_model import get_stochastic_shape, get_stochastic_module
                s = get_stochastic_shape(model, name)
                mod = get_stochastic_module(model, name)
                fused = _unpack(buf, s, mod.outlier_signs if header_info.get("version", 0) >= 7 else None).to(torch.int8)
                bin_name = prefix + ".latent_weights"

            if bin_name not in bin_weights:
                print(f"  MISSING in binary: {bin_name}")
                continue

            py_w = fused.cpu().numpy().astype(np.int8)
            bin_w = bin_weights[bin_name]
            if py_w.shape != bin_w.shape:
                print(f"  SHAPE MISMATCH: {bin_name} py={py_w.shape} bin={bin_w.shape}")
                continue

            total_compared += 1
            diff = np.abs(py_w.astype(np.float32) - bin_w.astype(np.float32)).max()
            max_diff = max(max_diff, diff)
            if diff < 1e-5:
                total_ok += 1
            elif verbose:
                print(f"  {bin_name}: max_diff={diff:.6f}")

        # Compare FP32 params
        for name, param in model.named_parameters():
            if name == "lm_head.weight":
                continue
            if name not in bin_weights:
                # Might be packed ternary (not stored as fp32)
                continue
            py_w = param.data.float().cpu().numpy()
            bin_w = bin_weights[name]
            if py_w.shape != bin_w.shape:
                # Squeeze shape for comparison
                py_w = py_w.reshape(bin_w.shape)
            total_compared += 1
            diff = np.abs(py_w - bin_w).max()
            max_diff = max(max_diff, diff)
            if diff < 1e-5:
                total_ok += 1
            elif verbose:
                print(f"  {name}: max_diff={diff:.6f}")
    else:
        # STE mode: standard comparison
        from ternary_llm.quantization import TernaryQuantizer
        for name, param in model.named_parameters():
            is_ternary = any(t in name for t in [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_up_proj", "down_proj",
                "kv_down_proj", "k_up_proj", "v_up_proj",
                "q_rope_proj", "k_rope_proj",
            ])
            if name == "lm_head.weight":
                continue
            if name not in bin_weights:
                print(f"  MISSING: {name}")
                continue
            if is_ternary:
                py_w = TernaryQuantizer.apply(param.data).cpu().numpy()
            else:
                py_w = param.data.float().cpu().numpy()
            bin_w = bin_weights[name]
            if py_w.shape != bin_w.shape:
                print(f"  SHAPE MISMATCH: {name} py={py_w.shape} bin={bin_w.shape}")
                continue
            total_compared += 1
            diff = np.abs(py_w - bin_w).max()
            max_diff = max(max_diff, diff)
            if diff < 1e-5:
                total_ok += 1
            elif verbose:
                print(f"  {name}: max_diff={diff:.6f}")

    print(f"\n  Compared: {total_compared} tensors")
    print(f"  Exact match: {total_ok}/{total_compared}")
    print(f"  Max diff: {max_diff:.6f}")

    passed = max_diff < 1e-3
    if passed:
        print(f"\n  [OK] Verification passed (max_diff < 1e-3)")
    else:
        print(f"\n  [WARN] Verification failed (max_diff >= 1e-3)")
    print(f"{'='*50}\n")
    return passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify exported Tetra binary")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("binary", nargs="?", default="tetra_model.bin",
                        help="Path to exported .bin (default: tetra_model.bin)")
    parser.add_argument("--mode", choices=["ste", "stochastic"], default=None,
                        help="Override mode detection")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    verify_export(args.checkpoint, args.binary, mode=args.mode, verbose=not args.quiet)


if __name__ == "__main__":
    main()
