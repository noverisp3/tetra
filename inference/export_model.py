"""Export model weights to binary format for C++ inference.

2 bits per ternary weight, 4 weights per byte (MSB first).
Binary format v4: header (64 bytes) + ternary sections + fp32 sections.
"""
import sys
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ternary_llm.transformer import TernaryTransformerModel, StochasticTransformerModel


TERNARY_ENCODING = {-1: 0b00, 0: 0b01, 1: 0b10}

TERNARY_PARAM_NAMES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_up_proj", "down_proj",               # ffn
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
            if val < 0:
                val = -1
            elif val > 0:
                val = 1
            else:
                val = 0
        encoded = TERNARY_ENCODING[val]
        byte_idx = i // 4
        bit_shift = 6 - (i % 4) * 2
        packed[byte_idx] |= encoded << bit_shift

    return packed.tobytes()


def count_params(model) -> tuple[int, int]:
    """Count ternary vs fp32 parameters."""
    ternary_count = 0
    fp32_count = 0

    # Check if model is stochastic (has packed_weights buffers)
    has_packed = any("packed_weights" in n for n, _ in model.named_buffers())

    if has_packed:
        # Stochastic mode: count packed_weights buffers as ternary
        for name, buf in model.named_buffers():
            if "packed_weights" in name:
                # Each packed byte holds 4 ternary weights
                ternary_count += buf.numel() * 4
        # Count remaining parameters (norms, embeddings, bias) as fp32
        for name, param in model.named_parameters():
            fp32_count += param.numel()
        # Subtract accumulator buffers (not exported)
    else:
        for name, param in model.named_parameters():
            is_ternary = any(t in name for t in TERNARY_PARAM_NAMES)
            if is_ternary:
                ternary_count += param.numel()
            else:
                fp32_count += param.numel()

    return ternary_count, fp32_count


def get_stochastic_module(model, buf_name: str):
    """Get the StochasticTernaryLinear module from a packed_weights buffer name."""
    name = buf_name.replace(".packed_weights", "")
    parts = name.split(".")
    layer = model.layers[int(parts[1])]
    sub = layer
    for p in parts[2:]:
        sub = getattr(sub, p)
    return sub


def get_stochastic_shape(model, buf_name: str) -> tuple:
    """Get (out_features, in_features) from the StochasticTernaryLinear module."""
    sub = get_stochastic_module(model, buf_name)
    return (sub.out_features, sub.in_features)


def export_model(model, output_path, mode="ste", scale=1.0):
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

        # Get FFN dim from first layer
        ffn_layer = model.layers[0].ffn
        if hasattr(ffn_layer, 'gate_up_proj'):
            ffn_dim = ffn_layer.gate_up_proj.out_features // 2
        else:
            ffn_dim = ffn_layer.gate_proj.out_features
        max_seq_len = model.max_seq_len

        header = struct.pack(
            "<4sIIIIIIIQQ16s",
            b"TETR",
            4,
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            ffn_dim,
            max_seq_len,
            ternary_count,
            fp32_count,
            b"\x00" * 16,
        )
        f.write(header)

        def write_ternary_entry(tensor, new_name, mod=None, alphas=None):
            """Write a ternary weight with v4 per-group alpha support."""
            nb = new_name.encode("utf-8")
            f.write(struct.pack("<I", len(nb))); f.write(nb)
            f.write(struct.pack("<HH", tensor.shape[0], tensor.shape[1]))
            group_size = getattr(mod, 'group_size', 0) if mod is not None else 0
            f.write(struct.pack("<H", group_size))
            if alphas is not None:
                a = alphas.detach().float().cpu().numpy() if hasattr(alphas, 'cpu') else np.array(alphas, dtype=np.float32)
                a = a.flatten()
                f.write(struct.pack("<H", len(a)))
                f.write(a.tobytes())
            elif mod is not None and hasattr(mod, 'alphas') and mod.alphas is not None:
                a = mod.alphas.detach().float().cpu().numpy().flatten()
                f.write(struct.pack("<H", len(a)))
                f.write(a.tobytes())
            else:
                f.write(struct.pack("<H", 0))
            f.write(pack_ternary(tensor))

        if mode == "stochastic":
            from ternary_llm.quantization import unpack_ternary_tensor as _unpack
            buffers = dict(model.named_buffers())
            written = set()

            for name, buf in buffers.items():
                if not name.endswith(".packed_weights"):
                    continue
                if name in written:
                    continue
                written.add(name)

                prefix = name.rsplit(".", 1)[0]
                layer_idx = name.split(".")[1]  # get layer index for gate_proj fusion check
                is_ffn_gate = "gate_proj" in name and f"layers.{layer_idx}.ffn" in name

                if is_ffn_gate:
                    up_name = name.replace("gate_proj", "up_proj")
                    if up_name not in buffers:
                        continue
                    written.add(up_name)
                    s = get_stochastic_shape(model, name)
                    gate_w = _unpack(buf, s)
                    up_w = _unpack(buffers[up_name], s)
                    fused = torch.cat([gate_w, up_w], dim=0).to(torch.int8)
                    new_name = prefix.replace("gate_proj", "gate_up_proj") + ".latent_weights"
                    gate_mod = get_stochastic_module(model, name)
                    up_mod = get_stochastic_module(model, up_name)
                    if gate_mod.alphas is not None:
                        combined = torch.cat([gate_mod.alphas, up_mod.alphas])
                        write_ternary_entry(fused, new_name, alphas=combined)
                    else:
                        write_ternary_entry(fused, new_name)
                elif "up_proj" in name and f"layers.{layer_idx}.ffn" in name:
                    continue
                else:
                    s = get_stochastic_shape(model, name)
                    w = _unpack(buf, s).to(torch.int8)
                    new_name = prefix + ".latent_weights"
                    mod = get_stochastic_module(model, name)
                    write_ternary_entry(w, new_name, mod)

            # --- Write fp32/int8 weights (stochastic) ---
            for name, param in model.named_parameters():
                if name == "lm_head.weight":
                    continue
                ndim = len(param.shape)
                shape = list(param.shape)
                while len(shape) < 4:
                    shape.append(1)
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<B", ndim))

                is_int8 = name in ("token_embedding.weight", "pos_embedding.weight")
                if is_int8:
                    f.write(struct.pack("<B", 1))
                    w = param.data.float().cpu().numpy().flatten()
                    scale = np.max(np.abs(w)) / 127.0
                    w_int8 = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
                    f.write(struct.pack("<4I", *shape))
                    f.write(struct.pack("<f", scale))
                    f.write(w_int8.tobytes())
                else:
                    f.write(struct.pack("<B", 0))
                    w_fp32 = param.data.float().cpu().numpy().flatten()
                    f.write(struct.pack("<4I", *shape))
                    f.write(w_fp32.tobytes())

        else:
            # --- STE mode: original export logic ---
            for name, param in model.named_parameters():
                is_ternary = any(t in name for t in TERNARY_PARAM_NAMES)
                if not is_ternary:
                    continue

                from ternary_llm.quantization import TernaryQuantizer
                w_ternary = TernaryQuantizer.apply(param.data)
                w_ternary = w_ternary.to(torch.int8)
                write_ternary_entry(w_ternary, name)

            for name, param in model.named_parameters():
                is_ternary = any(t in name for t in TERNARY_PARAM_NAMES)
                if is_ternary:
                    continue
                if name == "lm_head.weight":
                    continue

                ndim = len(param.shape)
                shape = list(param.shape)
                while len(shape) < 4:
                    shape.append(1)
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<B", ndim))

                is_int8 = name in ("token_embedding.weight", "pos_embedding.weight")
                if is_int8:
                    f.write(struct.pack("<B", 1))
                    w = param.data.float().cpu().numpy().flatten()
                    scale = np.max(np.abs(w)) / 127.0
                    w_int8 = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
                    f.write(struct.pack("<4I", *shape))
                    f.write(struct.pack("<f", scale))
                    f.write(w_int8.tobytes())
                else:
                    f.write(struct.pack("<B", 0))
                    w_fp32 = param.data.float().cpu().numpy().flatten()
                    f.write(struct.pack("<4I", *shape))
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
    mode = config.get("mode", "ste")

    # Detect MLA from config or from state dict keys
    sd = ckpt["model_state_dict"]
    mla = config.get("mla", False) or any("kv_down_proj" in k for k in sd)

    if mode == "stochastic" and mla:
        # Infer MLA dims from packed_weights shapes in state dict
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
        print(f"MLA detected: kv_latent_dim={kv_latent_dim}, rope_per_head={rope_per_head}")

    if mode == "stochastic" and mla:
        from ternary_llm.transformer import StochasticMLAModel
        model = StochasticMLAModel(
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"],
            max_seq_len=config["max_seq_len"],
            scale=config.get("ternary_scale", 1.0),
            threshold=config.get("threshold", None),
            int8=config.get("int8", False),
            topk=config.get("topk", 1.0),
            group_size=config.get("group_size", 0),
            kv_latent_dim=config.get("kv_latent_dim", None),
            rope_per_head=config.get("rope_per_head", None),
        )
    elif mode == "stochastic":
        model = StochasticTransformerModel(
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"],
            max_seq_len=config["max_seq_len"],
            scale=config.get("ternary_scale", 1.0),
            threshold=config.get("threshold", None),
            int8=config.get("int8", False),
            topk=config.get("topk", 1.0),
            group_size=config.get("group_size", 0),
        )
    else:
        model = TernaryTransformerModel(
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"],
            max_seq_len=config["max_seq_len"],
        )
    model.load_state_dict(ckpt["model_state_dict"])

    scale = config.get("ternary_scale", 1.0)
    export_model(model, output_path, mode=mode, scale=scale)


if __name__ == "__main__":
    main()
