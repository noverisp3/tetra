"""Export model weights to binary format for C++ inference.

Binary format v5: header (64 bytes) + ternary sections + fp32/int8 sections + optional metadata.

Header v5 layout (64 bytes):
  0-3:   magic "TETR"
  4-7:   version (5)
  8-11:  vocab_size
  12-15: hidden_dim
  16-19: num_layers
  20-23: num_heads
  24-27: ffn_dim
  28-31: max_seq_len
  32-39: ternary_params (uint64)
  40-47: fp32_params (uint64)
  48-49: flags (bit0=is_mla, bit1=int8_embeddings)
  50-51: kv_latent_dim (uint16, MLA only)
  52-53: rope_per_head (uint16, MLA only)
  54-55: group_size (uint16, default quantization group size)
  56-63: reserved (zeroes)

Metadata section (optional, after all weights):
  4 bytes: "META" magic
  4 bytes: metadata_len (uint32 LE)
  N bytes: UTF-8 JSON string
"""
import sys
import struct
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ternary_llm.transformer import (
    TernaryTransformerModel, StochasticTransformerModel, StochasticMLAModel,
)

# v7: ±2 outliers map to code 11 (sign in the side-channel blob).
TERNARY_ENCODING = {-1: 0b00, 0: 0b01, 1: 0b10, 2: 0b11, -2: 0b11}

HEADER_V5_SIZE = 64
METADATA_MAGIC = b"META"


# ──────────────────────────────────────────────────────────────
# Packing helpers
# ──────────────────────────────────────────────────────────────

def pack_ternary(weights: torch.Tensor) -> bytes:
    """Pack ternary weights {-2, -1, 0, 1, 2} into 2-bit encoding.

    ±2 outliers (v7) map to code 11; the sign lives in the side-channel
    blob. v6 values never reach ±2, so v6 output is unchanged.
    """
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


# ──────────────────────────────────────────────────────────────
# Parameter counting
# ──────────────────────────────────────────────────────────────

TERNARY_PARAM_NAMES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_up_proj", "down_proj",
    "kv_down_proj", "k_up_proj", "v_up_proj",
    "q_rope_proj", "k_rope_proj",
]

FP32_PARAM_NAMES = [
    "token_embedding", "pos_embedding",
    "attn_norm", "ffn_norm", "norm",
    "lm_head",
]


def count_params(model) -> tuple[int, int]:
    """Count ternary vs fp32 parameters."""
    ternary_count = 0
    fp32_count = 0
    has_packed = any("packed_weights" in n for n, _ in model.named_buffers())

    if has_packed:
        for name, buf in model.named_buffers():
            if "packed_weights" in name:
                ternary_count += buf.numel() * 4
        for name, param in model.named_parameters():
            fp32_count += param.numel()
    else:
        for name, param in model.named_parameters():
            is_ternary = any(t in name for t in TERNARY_PARAM_NAMES)
            if is_ternary:
                ternary_count += param.numel()
            else:
                fp32_count += param.numel()

    return ternary_count, fp32_count


# ──────────────────────────────────────────────────────────────
# Stochastic module helpers
# ──────────────────────────────────────────────────────────────

def get_stochastic_module(model, buf_name: str):
    name = buf_name.replace(".packed_weights", "")
    parts = name.split(".")
    layer = model.layers[int(parts[1])]
    sub = layer
    for p in parts[2:]:
        sub = getattr(sub, p)
    return sub


def get_stochastic_shape(model, buf_name: str) -> tuple:
    sub = get_stochastic_module(model, buf_name)
    return (sub.out_features, sub.in_features)


# ──────────────────────────────────────────────────────────────
# INT8 quantization
# ──────────────────────────────────────────────────────────────

def quantize_fp32_to_int8(tensor: torch.Tensor):
    """Quantize float32 tensor to int8 with per-tensor scale.

    Returns (scale: float32, w_int8: int8 numpy array).
    """
    w = tensor.detach().float().cpu().numpy().flatten()
    scale = np.max(np.abs(w)) / 127.0
    if scale < 1e-10:
        scale = 1.0
    w_int8 = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return scale, w_int8


# ──────────────────────────────────────────────────────────────
# Binary write helpers
# ──────────────────────────────────────────────────────────────

def write_header_v5(f, *, vocab_size, hidden_dim, num_layers, num_heads,
                     ffn_dim, max_seq_len, ternary_count, fp32_count,
                     is_mla=False, kv_latent_dim=0, rope_per_head=0,
                     group_size=0, int8_embeddings=False, version=5):
    """Write 64-byte v5/v6 header."""
    flags = 0
    if is_mla:
        flags |= 1
    if int8_embeddings:
        flags |= 2

    header = struct.pack(
        "<4sIIIIIIIQQHHHH8s",
        b"TETR",
        version,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        ffn_dim,
        max_seq_len,
        ternary_count,
        fp32_count,
        flags,
        kv_latent_dim,
        rope_per_head,
        group_size,
        b"\x00" * 8,
    )
    assert len(header) == HEADER_V5_SIZE, f"Header size mismatch: {len(header)} != {HEADER_V5_SIZE}"
    f.write(header)


def write_ternary_entry(f, tensor, new_name, mod=None, alphas=None, accumulator=None,
                        outlier_blob=None):
    """Write a ternary weight entry.

    When ``accumulator`` (FP32, rows*cols) is given the entry is written in
    v6 form: the FP32 accumulator follows the packed ternary data. This is the
    learning state that lets the C++ runtime keep self-learning across runs.

    When ``outlier_blob`` (dense sign bits for code-11 ±2 weights, row-major
    MSB-first) is given the entry is written in v7 form: a uint32 outlier
    count followed by the sign blob is appended at the end of the entry.
    """
    nb = new_name.encode("utf-8")
    f.write(struct.pack("<I", len(nb)))
    f.write(nb)
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

    if accumulator is not None:
        acc = (accumulator.detach().float().cpu().numpy()
               if hasattr(accumulator, "cpu") else np.asarray(accumulator, dtype=np.float32))
        f.write(acc.reshape(-1).astype(np.float32).tobytes())

    if outlier_blob is not None:
        n_outliers = int((tensor.abs() > 1.5).sum())
        f.write(struct.pack("<I", n_outliers))
        # Fused gate+up blobs are concatenated per-sub-blob byte-padded; the
        # dense outlier bits (MSB-first) are repacked so the blob is exactly
        # ceil(n_outliers/8) bytes. No-op for single-module blobs.
        bits = np.unpackbits(np.frombuffer(outlier_blob, dtype=np.uint8))
        blob = np.packbits(bits[:n_outliers]).tobytes()
        f.write(blob)


def count_outliers_in_packed(packed: torch.Tensor) -> int:
    """Count code-11 (outlier) weights in a packed uint8 tensor (per 2-bit code).

    NOTE: bool ``+`` is OR for both torch and numpy tensors, so the four
    comparisons are cast to integer before summing to get per-weight counts.
    """
    arr = packed.detach().cpu().numpy().astype(np.uint16)
    c0 = (arr >> 6) & 3
    c1 = (arr >> 4) & 3
    c2 = (arr >> 2) & 3
    c3 = arr & 3
    return int(((c0 == 3).astype(np.int64) + (c1 == 3).astype(np.int64)
                + (c2 == 3).astype(np.int64) + (c3 == 3).astype(np.int64)).sum())


def _fuse_blobs(gate_blob: bytes, up_blob: bytes, g_n: int, u_n: int) -> bytes:
    """Concatenate two dense sign blobs, stripping per-blob byte padding.

    Each sub-blob is byte-padded, so a raw concat misaligns the second stream
    whenever the first ends mid-byte. Only the first ``g_n``/``u_n`` bits are
    valid; the fused blob is exactly ``ceil((g_n + u_n) / 8)`` bytes.
    """
    bits = np.unpackbits(np.frombuffer(gate_blob, dtype=np.uint8))[:g_n]
    bits = np.concatenate([
        bits,
        np.unpackbits(np.frombuffer(up_blob, dtype=np.uint8))[:u_n],
    ])
    return np.packbits(bits).tobytes()


def write_fp32_entry(f, name, param, quantize_int8=False):
    """Write a fp32 (or int8-quantized) weight entry."""
    ndim = len(param.shape)
    shape = list(param.shape)
    while len(shape) < 4:
        shape.append(1)

    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<I", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<B", ndim))

    is_int8_candidate = name in ("token_embedding.weight", "pos_embedding.weight")
    use_int8 = is_int8_candidate or quantize_int8

    if use_int8:
        scale, w_int8 = quantize_fp32_to_int8(param.data)
        f.write(struct.pack("<B", 1))  # dtype=1 (INT8)
        f.write(struct.pack("<4I", *shape))
        f.write(struct.pack("<f", scale))
        f.write(w_int8.tobytes())
    else:
        f.write(struct.pack("<B", 0))  # dtype=0 (FP32)
        w_fp32 = param.data.float().cpu().numpy().flatten()
        f.write(struct.pack("<4I", *shape))
        f.write(w_fp32.tobytes())


def write_metadata(f, metadata: dict):
    """Write optional metadata section at end of file."""
    json_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    f.write(METADATA_MAGIC)
    f.write(struct.pack("<I", len(json_bytes)))
    f.write(json_bytes)


# ──────────────────────────────────────────────────────────────
# Model architecture detection
# ──────────────────────────────────────────────────────────────

def detect_model_info(model) -> dict:
    """Detect model architecture info from a loaded model."""
    info = {}

    # MLA detection
    has_packed = any("packed_weights" in n for n, _ in model.named_buffers())
    if has_packed:
        info["mode"] = "stochastic"
        info["is_mla"] = any("kv_down_proj" in n for n, _ in model.named_buffers())
    else:
        info["mode"] = "ste"
        info["is_mla"] = False

    info["hidden_dim"] = model.hidden_dim
    info["num_layers"] = len(model.layers)
    info["vocab_size"] = model.token_embedding.num_embeddings
    info["max_seq_len"] = model.max_seq_len

    # Get attention params
    layer0 = model.layers[0]
    if hasattr(layer0, 'attn'):
        attn = layer0.attn
        info["num_heads"] = attn.num_heads
        if info["is_mla"]:
            info["kv_latent_dim"] = getattr(attn, 'kv_latent_dim', 0)
            info["rope_per_head"] = getattr(attn, 'rope_per_head', 0)
        else:
            info["kv_latent_dim"] = 0
            info["rope_per_head"] = 0

    # Get FFN dim
    ffn_layer = layer0.ffn
    if hasattr(ffn_layer, 'gate_up_proj'):
        info["ffn_dim"] = ffn_layer.gate_up_proj.out_features // 2
    elif hasattr(ffn_layer, 'gate_proj'):
        info["ffn_dim"] = ffn_layer.gate_proj.out_features
    else:
        info["ffn_dim"] = 0

    # Group size (from first ternary layer)
    info["group_size"] = 0
    if has_packed:
        for name, buf in model.named_buffers():
            if "packed_weights" in name:
                mod = get_stochastic_module(model, name)
                info["group_size"] = getattr(mod, 'group_size', 0)
                break

    # Check for int8 embeddings
    info["int8_embeddings"] = True  # we always quantize embeddings to int8

    # Total params
    ternary_count, fp32_count = count_params(model)
    info["ternary_count"] = ternary_count
    info["fp32_count"] = fp32_count
    info["total_count"] = ternary_count + fp32_count

    return info


# ──────────────────────────────────────────────────────────────
# Export: STE mode
# ──────────────────────────────────────────────────────────────

def export_ste(f, model, quantize_int8=False, v7=False):
    """Export STE-mode ternary weights (per-channel Δ + per-group alphas).

    v7: |W| > 1.5Δ weights are exported as ±2 outliers (code 11) with their
    signs in a per-entry side-channel blob.
    """
    from ternary_llm.quantization import TernaryQuantizer, pack_sign_blob

    for name, param in model.named_parameters():
        is_ternary = any(t in name for t in TERNARY_PARAM_NAMES)
        if not is_ternary:
            continue
        mod_path = name.rsplit(".latent_weights", 1)[0]
        mod = model
        for part in mod_path.split("."):
            mod = getattr(mod, part)
        if getattr(mod, "per_channel", False):
            delta = param.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-6) * mod.ternary_scale
            w_ternary = (param.detach() / delta).round().clamp(-2, 2)
        else:
            w_ternary = TernaryQuantizer.apply(param.data)
        blob = pack_sign_blob(w_ternary) if v7 else None
        w_ternary = w_ternary.to(torch.int8)
        write_ternary_entry(f, w_ternary, name, mod, outlier_blob=blob)

    for name, param in model.named_parameters():
        is_ternary = any(t in name for t in TERNARY_PARAM_NAMES)
        if is_ternary or name == "lm_head.weight":
            continue
        write_fp32_entry(f, name, param, quantize_int8=quantize_int8)


# ──────────────────────────────────────────────────────────────
# Export: Stochastic mode
# ──────────────────────────────────────────────────────────────

def export_stochastic(f, model, quantize_int8=False, v7=False):
    """Export stochastic-mode ternary weights with gate+up fusion.

    v7: code-11 outlier signs are written per entry from the module's
    side-channel buffer. v6/v7 entries always carry a zero FP32 accumulator so
    the binary layout is uniform (packed + accumulator + optional blob) and
    the C++ reader never has to guess whether the accumulator is present.
    """
    from ternary_llm.quantization import unpack_ternary_tensor as _unpack

    def _zero_acc(shape) -> torch.Tensor:
        n = int(shape[0]) * int(shape[1])
        return torch.zeros(n, dtype=torch.float32)

    def _blob_of(mod) -> bytes | None:
        """Trimmed sign bits for the module's actual outlier count.

        In v7 mode always returns bytes (possibly empty) so every v7 ternary
        entry carries a trailing uint32 outlier count + blob.
        """
        if mod is None or not hasattr(mod, "outlier_signs"):
            return None
        if not v7:
            return None
        n = count_outliers_in_packed(mod.packed_weights)
        if n == 0:
            return b""
        nb = (n + 7) // 8
        return mod.outlier_signs.detach().cpu().numpy()[:nb].tobytes()

    buffers = dict(model.named_buffers())
    written = set()

    for name, buf in buffers.items():
        if not name.endswith(".packed_weights"):
            continue
        if name in written:
            continue
        written.add(name)

        prefix = name.rsplit(".", 1)[0]
        layer_idx = name.split(".")[1]
        is_ffn_gate = "gate_proj" in name and f"layers.{layer_idx}.ffn" in name

        if is_ffn_gate:
            up_name = name.replace("gate_proj", "up_proj")
            if up_name not in buffers:
                continue
            written.add(up_name)
            s = get_stochastic_shape(model, name)
            gate_mod = get_stochastic_module(model, name)
            up_mod = get_stochastic_module(model, up_name)
            gate_w = _unpack(buf, s, gate_mod.outlier_signs if v7 else None)
            up_w = _unpack(buffers[up_name], s, up_mod.outlier_signs if v7 else None)
            fused = torch.cat([gate_w, up_w], dim=0).to(torch.int8)
            new_name = prefix.replace("gate_proj", "gate_up_proj") + ".latent_weights"
            fused_blob = None
            if v7:
                gb, ub = _blob_of(gate_mod), _blob_of(up_mod)
                fused_blob = _fuse_blobs(
                    gb or b"", ub or b"",
                    count_outliers_in_packed(gate_mod.packed_weights),
                    count_outliers_in_packed(up_mod.packed_weights),
                )
            if gate_mod.alphas is not None:
                combined = torch.cat([gate_mod.alphas, up_mod.alphas])
                write_ternary_entry(fused, new_name, alphas=combined,
                                    accumulator=_zero_acc(fused.shape), outlier_blob=fused_blob)
            else:
                write_ternary_entry(f, fused, new_name, accumulator=_zero_acc(fused.shape),
                                    outlier_blob=fused_blob)
        elif "up_proj" in name and f"layers.{layer_idx}.ffn" in name:
            continue
        else:
            s = get_stochastic_shape(model, name)
            mod = get_stochastic_module(model, name)
            w = _unpack(buf, s, mod.outlier_signs if v7 else None).to(torch.int8)
            new_name = prefix + ".latent_weights"
            write_ternary_entry(f, w, new_name, mod, accumulator=_zero_acc(w.shape),
                                outlier_blob=_blob_of(mod))

    for name, param in model.named_parameters():
        if name == "lm_head.weight":
            continue
        write_fp32_entry(f, name, param, quantize_int8=quantize_int8)


# ──────────────────────────────────────────────────────────────
# Export: Self-learning mode (v6)
# ──────────────────────────────────────────────────────────────

def export_self_learning(
    model, output_path,
    rule="c", threshold=20.0, acc_decay=0.99, flip_every_n=5,
    logit_scale=1.0 / 16.0, lr_embedding=1e-4, wd_embedding=0.1,
    block_size=128, toggle=False, reset_accs=False, metadata=None, verbose=True,
    v7=False,
):
    """Export a stochastic model to binary format v6/v7 for the C++ self-learning runtime.

    v6 = v5 + per-ternary FP32 accumulators (learning state) + ``sl_*`` config
    in the metadata JSON. The token embedding is written as FP32 (mutable) so
    the runtime can keep applying its local SGD.

    v7 = v6 + code-11 outlier signs (dense side-channel blob per entry).
    The C++ runtime resolves the outlier magnitude (±2α) at staging time.

    ``reset_accs`` zeroes every exported accumulator: the Python accumulator
    state is transient and unsafe to replay on device (finding #10) — toggled
    runs must start from zeroed accumulators.
    """
    model.eval()
    from ternary_llm.quantization import unpack_ternary_tensor as _unpack

    def _blob_of(mod) -> bytes | None:
        """Trimmed sign bits for the module's actual outlier count.

        In v7 mode always returns bytes (possibly empty) so every v7 ternary
        entry carries a trailing uint32 outlier count + blob.
        """
        if mod is None or not hasattr(mod, "outlier_signs"):
            return None
        if not v7:
            return None
        n = count_outliers_in_packed(mod.packed_weights)
        if n == 0:
            return b""
        nb = (n + 7) // 8
        return mod.outlier_signs.detach().cpu().numpy()[:nb].tobytes()

    info = detect_model_info(model)
    ternary_count = info["ternary_count"]
    fp32_count = info["fp32_count"]

    buffers = dict(model.named_buffers())
    acc_buffers = {
        n.replace(".packed_weights", ".accumulator"): buffers.get(n.replace(".packed_weights", ".accumulator"))
        for n, b in buffers.items() if n.endswith(".packed_weights")
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Tetra Export v{'7' if v7 else '6'} (self-learning)")
        print(f"{'='*50}")
        print(f"  Rule:          {rule}  (c=predictive coding)")
        print(f"  Threshold:     {threshold}")
        print(f"  Acc decay:     {acc_decay}")
        print(f"  Flip every:    {flip_every_n}")
        print(f"  Logit scale:   {logit_scale:.6f}")
        print(f"  Emb lr/wd:     {lr_embedding} / {wd_embedding}")
        print(f"  Block size:    {block_size}")
        print(f"  Toggle:        {toggle}")
        print(f"  Acc state:     {'zeroed (--sl-reset-acc)' if reset_accs else 'exported from checkpoint'}")
        print(f"  Hidden:        {info['hidden_dim']}")
        print(f"  Layers:        {info['num_layers']}")
        print(f"  Heads:         {info['num_heads']}")
        print(f"  FFN dim:       {info['ffn_dim']}")
        print(f"  Vocab:         {info['vocab_size']}")
        print(f"  Ternary:       {ternary_count:,} ({ternary_count * 2 / 8 / 1024:.1f} KB packed)")
        print(f"  Accumulators:  {ternary_count * 4 / 1024:.1f} KB FP32")
        print(f"{'='*50}")

    with open(output_path, "wb") as f:
        write_header_v5(
            f,
            vocab_size=info["vocab_size"],
            hidden_dim=info["hidden_dim"],
            num_layers=info["num_layers"],
            num_heads=info["num_heads"],
            ffn_dim=info["ffn_dim"],
            max_seq_len=info["max_seq_len"],
            ternary_count=ternary_count,
            fp32_count=fp32_count,
            is_mla=info["is_mla"],
            kv_latent_dim=info.get("kv_latent_dim", 0),
            rope_per_head=info.get("rope_per_head", 0),
            group_size=info.get("group_size", 0),
            int8_embeddings=False,
            version=7 if v7 else 6,
        )

        written = set()
        for name, buf in buffers.items():
            if not name.endswith(".packed_weights"):
                continue
            if name in written:
                continue
            written.add(name)

            prefix = name.rsplit(".", 1)[0]
            layer_idx = name.split(".")[1]
            is_ffn_gate = "gate_proj" in name and f"layers.{layer_idx}.ffn" in name
            acc = acc_buffers.get(prefix + ".accumulator")
            if reset_accs and acc is not None:
                acc = torch.zeros_like(acc)

            if is_ffn_gate:
                up_name = name.replace("gate_proj", "up_proj")
                if up_name not in buffers:
                    continue
                written.add(up_name)
                s = get_stochastic_shape(model, name)
                gate_mod = get_stochastic_module(model, name)
                up_mod = get_stochastic_module(model, up_name)
                gate_w = _unpack(buf, s, gate_mod.outlier_signs if v7 else None)
                up_w = _unpack(buffers[up_name], s, up_mod.outlier_signs if v7 else None)
                fused = torch.cat([gate_w, up_w], dim=0).to(torch.int8)
                new_name = prefix.replace("gate_proj", "gate_up_proj") + ".latent_weights"
                up_acc = acc_buffers.get(prefix.replace("gate_proj", "up_proj") + ".accumulator")
                fused_acc = (torch.cat([acc, up_acc], dim=0) if acc is not None and up_acc is not None else None)
                fused_blob = None
                if v7:
                    gb, ub = _blob_of(gate_mod), _blob_of(up_mod)
                    fused_blob = _fuse_blobs(
                        gb or b"", ub or b"",
                        count_outliers_in_packed(gate_mod.packed_weights),
                        count_outliers_in_packed(up_mod.packed_weights),
                    )
                if gate_mod.alphas is not None:
                    combined = torch.cat([gate_mod.alphas, up_mod.alphas])
                    write_ternary_entry(f, fused, new_name, alphas=combined,
                                        accumulator=fused_acc, outlier_blob=fused_blob)
                else:
                    write_ternary_entry(f, fused, new_name, accumulator=fused_acc,
                                        outlier_blob=fused_blob)
            elif "up_proj" in name and f"layers.{layer_idx}.ffn" in name:
                continue
            else:
                s = get_stochastic_shape(model, name)
                mod = get_stochastic_module(model, name)
                w = _unpack(buf, s, mod.outlier_signs if v7 and mod.outlier_signs.numel() else None).to(torch.int8)
                new_name = prefix + ".latent_weights"
                write_ternary_entry(f, w, new_name, mod, accumulator=acc,
                                    outlier_blob=_blob_of(mod))

        # FP32 weights (embedding must stay FP32 for local SGD)
        for name, param in model.named_parameters():
            if name == "lm_head.weight":
                continue
            is_embed = name in ("token_embedding.weight", "pos_embedding.weight")
            write_fp32_entry(f, name, param, quantize_int8=False if is_embed else False)

        meta = dict(metadata) if metadata else {}
        outlier_mult = 3.0
        for m in model.modules():
            if hasattr(m, "outlier_thr_mult"):
                outlier_mult = float(m.outlier_thr_mult)
                break
        meta.update({
            "_export_version": 7 if v7 else 6,
            "_export_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sl_rule": rule,
            "sl_threshold": float(threshold),
            "sl_acc_decay": float(acc_decay),
            "sl_flip_every_n": int(flip_every_n),
            "sl_toggle": int(toggle),
            "sl_logit_scale": float(logit_scale),
            "sl_lr_embedding": float(lr_embedding),
            "sl_wd_embedding": float(wd_embedding),
            "sl_block_size": int(block_size),
            "sl_outlier_mult": float(outlier_mult),
            "_info": {k: v for k, v in info.items() if k != "mode"},
        })
        write_metadata(f, meta)

    file_size = Path(output_path).stat().st_size
    if verbose:
        print(f"\n  Output:        {output_path}")
        print(f"  File size:     {file_size / 1024:.1f} KB")
        print(f"{'='*50}\n")

    return info


# ──────────────────────────────────────────────────────────────
# Main export function
# ──────────────────────────────────────────────────────────────

def export_model(model, output_path, mode="ste", quantize_int8=False,
                 metadata=None, verbose=True, v7=False):
    """Export model weights to binary format.

    Args:
        model: PyTorch model
        output_path: output .bin path
        mode: "ste" or "stochastic"
        quantize_int8: quantize all FP32 weights to INT8
        metadata: optional dict to store as JSON metadata section
        verbose: print detailed stats
        v7: write code-11 outlier side-channel (format version 7)
    """
    model.eval()

    info = detect_model_info(model)
    ternary_count = info["ternary_count"]
    fp32_count = info["fp32_count"]

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Tetra Export v5")
        print(f"{'='*50}")
        print(f"  Mode:          {info['mode']}")
        print(f"  MLA:           {info['is_mla']}")
        if info['is_mla']:
            print(f"  kv_latent_dim: {info['kv_latent_dim']}")
            print(f"  rope_per_head: {info['rope_per_head']}")
        print(f"  Hidden:        {info['hidden_dim']}")
        print(f"  Layers:        {info['num_layers']}")
        print(f"  Heads:         {info['num_heads']}")
        print(f"  FFN dim:       {info['ffn_dim']}")
        print(f"  Vocab:         {info['vocab_size']}")
        print(f"  Max seq:       {info['max_seq_len']}")
        print(f"  Group size:    {info['group_size']}")
        print(f"  Ternary:       {ternary_count:,} ({ternary_count * 2 / 8 / 1024:.1f} KB packed)")
        print(f"  FP32/INT8:     {fp32_count:,} ({fp32_count * 4 / 1024:.1f} KB)")
        if quantize_int8:
            print(f"  INT8 quant:    all FP32 weights → INT8 (4× compression)")
        print(f"{'='*50}")

    with open(output_path, "wb") as f:
        # Write v5 header
        write_header_v5(
            f,
            vocab_size=info["vocab_size"],
            hidden_dim=info["hidden_dim"],
            num_layers=info["num_layers"],
            num_heads=info["num_heads"],
            ffn_dim=info["ffn_dim"],
            max_seq_len=info["max_seq_len"],
            ternary_count=ternary_count,
            fp32_count=fp32_count,
            is_mla=info["is_mla"],
            kv_latent_dim=info.get("kv_latent_dim", 0),
            rope_per_head=info.get("rope_per_head", 0),
            group_size=info.get("group_size", 0),
            int8_embeddings=info.get("int8_embeddings", False),
            version=7 if v7 else 5,
        )

        # Write ternary + FP32 weights
        if mode == "stochastic":
            export_stochastic(f, model, quantize_int8=quantize_int8, v7=v7)
        else:
            export_ste(f, model, quantize_int8=quantize_int8, v7=v7)

        # Write metadata section
        if metadata:
            meta = dict(metadata)
            meta["_export_version"] = 7 if v7 else 5
            meta["_export_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            meta["_info"] = {
                k: v for k, v in info.items()
                if k not in ("mode",)  # exclude non-serializable
            }
            write_metadata(f, meta)

    file_size = Path(output_path).stat().st_size
    ternary_kb = ternary_count * 2 / 8 / 1024
    overhead_kb = (file_size - ternary_kb * 1024) / 1024

    if verbose:
        print(f"\n  Output:        {output_path}")
        print(f"  File size:     {file_size / 1024:.1f} KB")
        print(f"  Compression:   {info['total_count'] * 4 / 1024 / (file_size / 1024):.1f}× vs FP32")
        if metadata:
            print(f"  Metadata:      {len(json.dumps(metadata))} bytes (JSON)")
        print(f"{'='*50}\n")

    return info


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Tetra model to binary format for C++ inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_model.py checkpoint.pt
  python export_model.py checkpoint.pt -o model.bin
  python export_model.py checkpoint.pt --quantize-int8 --verify
  python export_model.py checkpoint.pt --metadata-step 15000 --metadata-loss 2.31
        """,
    )
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("-o", "--output", default="tetra_model.bin",
                        help="Output binary path (default: tetra_model.bin)")
    parser.add_argument("--quantize-int8", action="store_true",
                        help="Quantize all FP32 weights to INT8 (saves ~50%% non-ternary size)")
    parser.add_argument("--verify", action="store_true",
                        help="Auto-verify export against PyTorch model")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Skip embedding training metadata")
    parser.add_argument("--metadata-step", type=int, default=None,
                        help="Training step for metadata")
    parser.add_argument("--metadata-loss", type=float, default=None,
                        help="Training loss for metadata")
    parser.add_argument("--metadata-dataset", type=str, default=None,
                        help="Dataset name for metadata")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress verbose output")
    parser.add_argument("--self-learning", action="store_true",
                        help="Export to v6 with accumulators + sl_* config for the C++ self-learning runtime")
    parser.add_argument("--sl-rule", default="c",
                        help="Self-learning rule for C++ runtime (default: c=predictive coding)")
    parser.add_argument("--sl-threshold", type=float, default=20.0)
    parser.add_argument("--sl-acc-decay", type=float, default=0.99)
    parser.add_argument("--sl-flip-every-n", type=int, default=5)
    parser.add_argument("--sl-lr-embedding", type=float, default=1e-4)
    parser.add_argument("--sl-wd-embedding", type=float, default=0.1)
    parser.add_argument("--sl-block-size", type=int, default=128)
    parser.add_argument("--sl-toggle", action="store_true",
                        help="Anti-stiction toggle kicks in the C++ self-learning runtime")
    parser.add_argument("--sl-reset-acc", action="store_true",
                        help="Write zeroed accumulators (safe default for toggled runs; the Python acc state is transient — finding #10)")
    parser.add_argument("--v7", action="store_true",
                        help="Write v7 format: code-11 outlier (±2) side-channel blobs (header version 7)")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    sd = ckpt["model_state_dict"]
    mode = config.get("mode", "ste")

    # Detect discrete (gradient-free) checkpoints: DiscreteConfig has a "rule".
    is_discrete = "rule" in config and "acc_decay" in config

    if args.self_learning or is_discrete:
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
        model.load_state_dict(sd, strict=False)

        logit_scale = 1.0 / (config["hidden_dim"] ** 0.5)
        metadata = dict(config) if not args.no_metadata else {}
        for k in list(metadata.keys()):
            v = metadata[k]
            if isinstance(v, torch.Tensor):
                metadata[k] = v.tolist()
            elif not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                del metadata[k]

        export_self_learning(
            model, args.output,
            rule=args.sl_rule if not is_discrete or args.self_learning else config.get("rule", "c"),
            threshold=args.sl_threshold,
            acc_decay=args.sl_acc_decay,
            flip_every_n=args.sl_flip_every_n,
            logit_scale=logit_scale,
            lr_embedding=args.sl_lr_embedding,
            wd_embedding=args.sl_wd_embedding,
            block_size=args.sl_block_size,
            toggle=args.sl_toggle,
            reset_accs=args.sl_reset_acc,
            metadata=metadata,
            verbose=not args.quiet,
            v7=args.v7,
        )
        return

    # Detect MLA from config or state dict keys
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
        print(f"MLA detected: kv_latent_dim={kv_latent_dim}, rope_per_head={rope_per_head}")
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
            kv_latent_dim=kv_latent_dim,
            rope_per_head=rope_per_head,
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
    model.load_state_dict(sd)

    # Build metadata
    metadata = None
    if not args.no_metadata:
        metadata = dict(config)
        # Remove non-serializable items
        for k in list(metadata.keys()):
            v = metadata[k]
            if isinstance(v, torch.Tensor):
                metadata[k] = v.tolist()
            elif not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                del metadata[k]
        if args.metadata_step is not None:
            metadata["training_step"] = args.metadata_step
        if args.metadata_loss is not None:
            metadata["training_loss"] = args.metadata_loss
        if args.metadata_dataset is not None:
            metadata["dataset"] = args.metadata_dataset

    info = export_model(
        model, args.output, mode=mode,
        quantize_int8=args.quantize_int8,
        metadata=metadata,
        verbose=not args.quiet,
        v7=args.v7,
    )

    # Auto-verify
    if args.verify:
        print("--- Running verification ---")
        from inference.verify_export import verify_export
        verify_export(args.checkpoint, args.output, mode=mode)


if __name__ == "__main__":
    main()
