"""Ternary quantization and stochastic bit-flip training kernels.

Provides custom autograd Functions for ternary weight quantization:
- TernaryQuantizer: STE ternary quantization for standard training
- FusedTernaryLinear: fused quantize + matmul for STE training
- StochasticBitFlipLinear: packed 2-bit ternary with accumulator-based gradient
- Int8StochasticBitFlipLinear: INT8 activation variant for memory efficiency

Also provides pack/unpack utilities for 2-bit ternary encoding.
"""
import os
import sys
from typing import Optional

__all__ = [
    "FusedTernaryLinear", "StochasticBitFlipLinear", "Int8StochasticBitFlipLinear",
    "TernaryQuantizer", "init_ternary_weight", "unpack_ternary_tensor",
    "pack_ternary_tensor", "pack_ternary", "unpack_ternary",
    "pack_sign_blob", "pack_sign_blob_tensor", "apply_bit_flips",
    "ternary_matmul_forward", "ternary_forward_direct",
]

import torch
import torch.nn.functional as F

# Optional C++ SIMD extension for fast pack/unpack
_ternary_ops = None

def _try_load_from_cache(name: str):
    """Try to import a compiled extension from PyTorch's cache directory by module name."""
    cache_base = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                              "torch_extensions", "torch_extensions", "Cache")
    if not os.path.isdir(cache_base):
        return None
    for ver_dir in os.listdir(cache_base):
        pyd_path = os.path.join(cache_base, ver_dir, name, f"{name}.pyd")
        if os.path.exists(pyd_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(name, pyd_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
            except Exception:
                pass
    return None

def _load_cpp_extension() -> bool:
    """Load C++ ternary_ops extension.

    Tries AVX-512 then AVX2 with platform-appropriate compiler flags.
    Returns True if loaded successfully, False if no compiler available.
    """
    global _ternary_ops
    if _ternary_ops is not None:
        return True

    # torch 2.4's ninja build path calls distutils._msvccompiler._get_vc_env;
    # modern setuptools (>= 60) no longer installs that shim into stdlib
    # distutils. Alias the vendored copy so the extension can still build.
    if sys.platform == "win32":
        try:
            import distutils
            if not hasattr(distutils, "_msvccompiler"):
                from setuptools._distutils import _msvccompiler
                distutils._msvccompiler = _msvccompiler
        except Exception:
            pass

    csrc = os.path.join(os.path.dirname(__file__), "csrc")

    # Platform-appropriate SIMD flags
    if sys.platform == "win32":
        avx512_flag = "/arch:AVX512"
        avx2_flag = "/arch:AVX2"
    else:
        avx512_flag = "-mavx512f -mavx512bw"
        avx2_flag = "-mavx2"

    def _has_avx512():
        try:
            if sys.platform == "win32":
                import ctypes
                return ctypes.windll.kernel32.IsProcessorFeaturePresent(0x17) != 0
            else:
                import subprocess
                out = subprocess.run(
                    ["grep", "avx512", "/proc/cpuinfo"],
                    capture_output=True, text=True
                )
                return bool(out.stdout.strip())
        except Exception:
            return False

    # Attempt 1: AVX-512 variant
    if _has_avx512():
        mod = _try_load_from_cache("ternary_ops_avx512")
        if mod is not None:
            _ternary_ops = mod
            return True
        avx512_src = os.path.join(csrc, "ternary_ops_avx512.cpp")
        if os.path.exists(avx512_src):
            try:
                from torch.utils.cpp_extension import load
                _ternary_ops = load(
                    name="ternary_ops_avx512", sources=[avx512_src],
                    extra_cflags=[avx512_flag], verbose=False,
                )
                return True
            except Exception:
                pass

    # Attempt 2: AVX2 variant (portable fallback)
    mod = _try_load_from_cache("ternary_ops")
    if mod is not None:
        _ternary_ops = mod
        return True
    avx2_src = os.path.join(csrc, "ternary_ops_avx2.cpp")
    if not os.path.exists(avx2_src):
        return False
    try:
        from torch.utils.cpp_extension import load
        _ternary_ops = load(
            name="ternary_ops", sources=[avx2_src],
            extra_cflags=[avx2_flag], verbose=False,
        )
        return True
    except Exception:
        return False

_has_cpp = _load_cpp_extension()


class TernaryQuantizer(torch.autograd.Function):
    """Ternary quantization {-1, 0, +1} with Straight-Through Estimator.

    Dynamic threshold: Δ = scale x mean(|W|), default scale=1.0 (Exp 9 tuning:
    scale 1.0 lowers CE and halves ±2 outlier share vs the old 0.7 default).
    Lower scale increases entropy (fewer zeros in ternary matrix),
    preventing model collapse (all weights -> 0).

    v7 outlier encoding: |W| > 1.5Δ promotes to ±2 (2-bit code 11, sign in a
    dense side-channel blob). Valid outputs are in {-2, -1, 0, +1, +2}.

    Forward:  round(W/Δ) -> clamp(-2, 2) -> {-2, -1, 0, +1, +2}
    Backward: STE passes grad straight through
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        ctx.scale = scale
        delta = input.abs().mean().clamp(min=1e-6) * scale
        # |W| > 1.5Δ promotes to an outlier (±2, code 11 in the 2-bit format)
        w_ternary = (input / delta).round().clamp(-2, 2)
        ctx.save_for_backward(input)
        return w_ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output, None


class FusedTernaryLinear(torch.autograd.Function):
    """Fused ternary quantization + linear matmul.

    Dynamic threshold: Δ = scale x mean(|W|), default scale=1.0 (Exp 9 tuning).
    Per-channel: Δ_i = scale x mean(|W[i,:]|).

    Forward:  clamp(W/Δ, -1, 1) -> round -> matmul(x, W_ternary)
    Backward: grad_x = grad @ W_ternary.T, grad_W = x^T @ grad (STE)

    Soft-to-hard surrogate (``gamma`` not None, Exp 8): instead of hard round,
    the quantizer is a sum of shifted sigmoids with a dead zone around 0 and
    matched 5-level boundaries {-2,-1,0,+1,+2}:

        Q~(x; γ) = -2 + σ(γ(x+1.5)) + σ(γ(x+0.5)) + σ(γ(x-0.5)) + σ(γ(x-1.5))

    As γ: 2 -> 50+ the surrogate converges to round(W/Δ) without a
    discontinuity, and the backward passes the *exact* surrogate gradient
    (no STE clip). Pass gamma=None for the standard hard-round + STE path.

    Saves ternary weights to avoid recomputing abs()
    in backward - avoids OOM on memory-constrained devices.

    v8-forward (``v8_forward=True``, Exp 10): outliers (|x| >= 1.5) dequantize
    to their true latent value at 1/32 level resolution —
    round((W/Delta)*32)/32, exactly the format-v8 int8 blob — instead of the
    clamp at +-2. Makes training consistent with the v8 inference
    representation (the model can exploit real outlier magnitude).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, latent_weights: torch.Tensor,
                scale: float = 1.0, per_channel: bool = False,
                alphas: Optional[torch.Tensor] = None,
                group_size: int = 0,
                gamma: Optional[float] = None,
                v8_forward: bool = False) -> torch.Tensor:
        if per_channel:
            delta = latent_weights.abs().mean(dim=1, keepdim=True).clamp(min=1e-6) * scale
        else:
            delta = latent_weights.abs().mean().clamp(min=1e-6) * scale
        # |W| > 1.5Δ promotes to an outlier (±2, code 11 in the 2-bit format);
        # the learned per-group alpha scales it to ±2α at inference time.
        x_n = latent_weights / delta
        if gamma is not None and gamma > 0:
            # Sum of shifted sigmoids: dead zone around 0, 5-level boundaries
            # at ±0.5 and ±1.5, converges to round(x) as γ -> ∞.
            gx = gamma * x_n
            w_ternary = (-2.0
                         + torch.sigmoid(gx + 1.5 * gamma)
                         + torch.sigmoid(gx + 0.5 * gamma)
                         + torch.sigmoid(gx - 0.5 * gamma)
                         + torch.sigmoid(gx - 1.5 * gamma))
        elif v8_forward:
            # Exp 10: true-value outliers — code 11 positions keep their
            # latent magnitude quantized at 1/32 level (exactly the v8 blob
            # encoding: byte = round((W/Δ)·32), dequant = byte/32).
            mask = x_n.abs() > 1.5
            core = x_n.round().clamp(-1, 1)
            true_vals = (x_n * 32.0).round().clamp(-127.0, 127.0) / 32.0
            w_ternary = torch.where(mask, true_vals, core)
        else:
            w_ternary = x_n.round().clamp(-2, 2)
        w_ternary = w_ternary.to(x.dtype)
        ctx.save_for_backward(x, w_ternary.detach(),
                              alphas.detach() if alphas is not None else alphas,
                              x_n.detach() if gamma is not None and gamma > 0 else x_n)
        ctx.group_size = group_size
        ctx.gamma = gamma
        ctx.delta = delta.detach()
        ctx.per_channel = per_channel
        ctx.scale = scale
        if alphas is not None and group_size > 0:
            in_features = x.size(-1)
            num_groups = (in_features + group_size - 1) // group_size
            ctx.num_groups = num_groups
            alphas_expanded = torch.repeat_interleave(alphas, group_size, dim=1)
            if in_features % group_size != 0:
                alphas_expanded = alphas_expanded[:, :in_features]
            w_scaled = w_ternary * alphas_expanded
            return F.linear(x, w_scaled)
        elif alphas is not None:
            return F.linear(x, w_ternary) * alphas.unsqueeze(0)
        return F.linear(x, w_ternary)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, w_ternary, alphas, x_n = ctx.saved_tensors
        w_ternary = w_ternary.to(grad_output.dtype)
        x = x.to(grad_output.dtype)
        in_features = x.size(-1)
        out_features = grad_output.size(-1)
        grad_output_flat = grad_output.reshape(-1, out_features)
        x_flat = x.reshape(-1, in_features)

        if alphas is not None and ctx.group_size > 0:
            num_groups = ctx.num_groups
            alphas = ctx.alphas
            alphas_expanded = torch.repeat_interleave(alphas, ctx.group_size, dim=1)
            if in_features % ctx.group_size != 0:
                alphas_expanded = alphas_expanded[:, :in_features]
            w_scaled = w_ternary * alphas_expanded
            grad_x = torch.mm(grad_output_flat, w_scaled).view(*x.shape[:-1], in_features)
            grad_w_scaled = torch.mm(grad_output_flat.T, x_flat)
            grad_w = grad_w_scaled * alphas_expanded
            grad_alpha = torch.zeros(out_features, num_groups, device=x.device)
            for g in range(num_groups):
                start = g * ctx.group_size
                end = min(start + ctx.group_size, in_features)
                grad_alpha[:, g] = (grad_w_scaled[:, start:end] * w_ternary[:, start:end]).sum(dim=1)
            grad_alpha = grad_alpha.detach()
        elif alphas is not None:
            grad_y_scaled = grad_output_flat * alphas.unsqueeze(0)
            grad_alpha = (grad_output_flat * torch.mm(x_flat, w_ternary.T)).sum(dim=0).detach()
            grad_x = torch.mm(grad_y_scaled, w_ternary).view(*x.shape[:-1], in_features)
            grad_w = torch.mm(grad_y_scaled.T, x_flat)
        else:
            grad_x = F.linear(grad_output, w_ternary.T)
            grad_w = torch.mm(grad_output_flat.T, x_flat)
            grad_alpha = None

        if ctx.gamma is not None and ctx.gamma > 0:
            # Exact surrogate gradient: dQ~/dx = γ·Σ_k σ(γ(x-b_k))·(1-σ(γ(x-b_k))).
            # Chain to latent weights through x_n = W/Δ, where Δ = scale·mean|W|
            # (per-tensor) or scale·rowmean|W| (per-channel). Δ is a function of
            # W, so we propagate through it exactly instead of detaching:
            #   dL/dΔ = Σ_j dL/dx_n_j · dx_n_j/dΔ,  dx_n_j/dΔ = -x_n_j/Δ
            #   dΔ/dW = scale·sign(W)/N (per-tensor), scale·sign(W)/K (per-channel)
            # sign(x_n) = sign(W) because Δ > 0, so no extra save of W needed.
            g = ctx.gamma
            gx = g * x_n
            s1 = torch.sigmoid(gx + 1.5 * g)
            s2 = torch.sigmoid(gx + 0.5 * g)
            s3 = torch.sigmoid(gx - 0.5 * g)
            s4 = torch.sigmoid(gx - 1.5 * g)
            # Q~'(x_n) = γ·Σ_k σ(z_k)·(1-σ(z_k)); 1-σ(z) = σ(-z);
            # slope already contains the γ factor, so dL/dx_n = grad_w·slope.
            slope = g * (s1 * torch.sigmoid(-(gx + 1.5 * g))
                         + s2 * torch.sigmoid(-(gx + 0.5 * g))
                         + s3 * torch.sigmoid(-(gx - 0.5 * g))
                         + s4 * torch.sigmoid(-(gx - 1.5 * g)))
            d_xn = grad_w * slope   # dL/dx_n
            if ctx.per_channel:
                grad_delta = -((d_xn * x_n).sum(dim=1, keepdim=True) / ctx.delta)
                grad_w = d_xn / ctx.delta + grad_delta * (ctx.scale / x_n.size(-1)) * torch.sign(x_n)
            else:
                grad_delta = -((d_xn * x_n).sum() / ctx.delta)
                grad_w = d_xn / ctx.delta + grad_delta * (ctx.scale / x_n.numel()) * torch.sign(x_n)
        return grad_x, grad_w.float(), None, None, grad_alpha, None, None, None


def ternary_quantize(weights: torch.Tensor) -> torch.Tensor:
    """Apply ternary quantization to a weight tensor (inference)."""
    return TernaryQuantizer.apply(weights)


def pack_ternary(weights: torch.Tensor) -> bytes:
    """Pack ternary weights to 2-bit format (4 weights/byte).

    Encoding: 00=-1, 01=0, 10=+1, 11=outlier (±2, sign in side-channel),
    MSB first. Input: flat float tensor with values in {-2, -1, 0, +1, +2}.
    Output: packed bytes.
    """
    w = weights.detach().cpu().flatten().to(torch.int64)
    # {-2,-1,0,1,2} -> codes {3,0,1,2,3}
    w = torch.where(w.abs() > 1, torch.full_like(w, 3), (w + 1).clamp(0, 2))
    n = len(w)
    # Pad to multiple of 4
    padded = (n + 3) // 4 * 4
    if padded != n:
        w = torch.nn.functional.pad(w, (0, padded - n), value=1)  # pad with 0 (encoded as 1)
    w = w.view(-1, 4).to(torch.uint8)
    # Pack: MSB first, 2 bits per weight
    packed = (w[:, 0] << 6 |
              w[:, 1] << 4 |
              w[:, 2] << 2 |
              w[:, 3])
    return bytes(packed.tolist())


def pack_sign_blob(weights: torch.Tensor) -> bytes:
    """Pack outlier signs into a dense bit blob.

    One bit per outlier weight (code 11), row-major scan order,
    MSB first within each byte: 1 = positive outlier, 0 = negative.
    Returns b"" when there are no outliers.
    """
    w = weights.detach().cpu().flatten()
    mask = w.abs() > 1.5
    n = int(mask.sum().item())
    if n == 0:
        return b""
    bits = (w[mask] > 0).to(torch.uint8)
    padded = (n + 7) // 8 * 8
    if padded != n:
        bits = torch.nn.functional.pad(bits, (0, padded - n))
    bits = bits.view(-1, 8)
    packed = (bits[:, 0] << 7 | bits[:, 1] << 6 | bits[:, 2] << 5 |
              bits[:, 3] << 4 | bits[:, 4] << 3 | bits[:, 5] << 2 |
              bits[:, 6] << 1 | bits[:, 7])
    return bytes(packed.tolist())


def v8_quantize(w: torch.Tensor, k: int = 32, per_channel: bool = False,
                scale: float = 1.0) -> tuple:
    """v8 top-k outlier quantization (format v8, STE export/verify path).

    The core stays pure ternary {-1, 0, +1} (clamp ±1, no ±2 clamp). Per row,
    the ``k`` weights with the largest |W/Δ| above the 1.5Δ outlier boundary
    keep their *true* value: code 11 marks the position and the side-channel
    blob stores round((W/Δ)·32) as a signed int8 (level units, 1/32Δ
    resolution, range ±3.97Δ). Everything else rounds to ±1 or 0.

    Returns (core tensor with ±2 markers at outlier positions, blob bytes).
    """
    if per_channel:
        delta = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-6) * scale
    else:
        delta = w.abs().mean().clamp(min=1e-6) * scale
    x_n = w / delta
    core = x_n.round().clamp(-1, 1)
    mask = x_n.abs() > 1.5
    if k > 0 and mask.any():
        vals = x_n.abs() * mask
        kk = min(k, int(mask.sum(dim=1).max().item()))
        if kk > 0:
            cutoff = torch.topk(vals, kk, dim=1).values[:, -1:]
            mask = mask & (vals >= cutoff)
    out = core.clone()
    if not mask.any():
        return out, b""
    out[mask] = torch.where(x_n[mask] >= 0,
                            torch.full_like(x_n[mask], 2.0),
                            torch.full_like(x_n[mask], -2.0))
    blob = (x_n[mask] * 32).round().clamp(-127, 127).to(torch.int8)
    return out, blob.cpu().numpy().tobytes()


def _apply_sign_blob(flat: torch.Tensor, sign_blob: bytes) -> torch.Tensor:
    """Apply sign bits to outlier positions of a flat weight tensor."""
    import numpy as np
    mask = flat.abs() > 1.5
    n = int(mask.sum().item())
    if n == 0 or not sign_blob:
        return flat
    bits = np.unpackbits(np.frombuffer(sign_blob, dtype=np.uint8))
    bits = bits[:n].astype(np.float32) * 2.0 - 1.0  # 0 -> -1, 1 -> +1
    flat = flat.clone()
    flat[mask] *= torch.from_numpy(bits)
    return flat


def unpack_ternary(packed: bytes, shape: tuple, sign_blob: Optional[bytes] = None,
                   device: str = "cpu") -> torch.Tensor:
    """Unpack 2-bit packed ternary weights back to float tensor.

    Encoding: 00=-1, 01=0, 10=+1, 11=outlier ±2 (sign from ``sign_blob``),
    MSB first.
    """
    import numpy as np
    data = np.frombuffer(packed, dtype=np.uint8)
    # Unpack 4 weights per byte, MSB first
    w0 = ((data >> 6) & 3).astype(np.int8) - 1
    w1 = ((data >> 4) & 3).astype(np.int8) - 1
    w2 = ((data >> 2) & 3).astype(np.int8) - 1
    w3 = ((data >> 0) & 3).astype(np.int8) - 1
    flat = np.stack([w0, w1, w2, w3], axis=-1).flatten()
    flat = flat[:np.prod(shape)]
    flat = np.ascontiguousarray(flat)
    t = torch.from_numpy(flat.astype(np.float32))
    if sign_blob:
        t = _apply_sign_blob(t, sign_blob)
    return t.reshape(shape).to(device)


# Fast Tensor Pack/Unpack for Stochastic Bit-Flip

def pack_ternary_tensor(w: torch.Tensor) -> torch.Tensor:
    """Pack ternary float tensor {-2,-1,0,+1,+2} -> uint8 tensor (4 weights/byte).

    ±2 maps to code 11 (outlier); the sign lives in the side-channel blob.
    """
    if _has_cpp and w.is_cpu and w.dtype in (torch.float32, torch.float16):
        return _ternary_ops.pack_ternary(w.contiguous())
    # {-2,-1,0,1,2} -> codes {3,0,1,2,3}
    w_i = w.flatten().to(torch.int64)
    w_codes = torch.where(w_i.abs() > 1, torch.full_like(w_i, 3), (w_i + 1).clamp(0, 2))
    flat = w_codes.to(torch.uint8)
    n = flat.size(0)
    padded = (n + 3) // 4 * 4
    if padded != n:
        flat = torch.nn.functional.pad(flat, (0, padded - n), value=1)
    flat = flat.view(-1, 4)
    packed = flat[:, 0] * 64 + flat[:, 1] * 16 + flat[:, 2] * 4 + flat[:, 3]
    return packed.contiguous().to(torch.uint8)


def pack_sign_blob_tensor(w: torch.Tensor) -> torch.Tensor:
    """Pack outlier signs of ``w`` into a dense uint8 bit tensor.

    Row-major scan order, MSB first, 1 bit per outlier (1 = positive).
    Returns an empty tensor when there are no outliers.
    """
    w = w.detach().cpu().flatten()
    mask = w.abs() > 1.5
    n = int(mask.sum().item())
    if n == 0:
        return torch.zeros(0, dtype=torch.uint8)
    bits = (w[mask] > 0).to(torch.uint8)
    padded = (n + 7) // 8 * 8
    if padded != n:
        bits = torch.nn.functional.pad(bits, (0, padded - n))
    bits = bits.view(-1, 8)
    packed = (bits[:, 0] << 7 | bits[:, 1] << 6 | bits[:, 2] << 5 |
              bits[:, 3] << 4 | bits[:, 4] << 3 | bits[:, 5] << 2 |
              bits[:, 6] << 1 | bits[:, 7])
    return packed.contiguous().to(torch.uint8)


def _apply_signs_tensor(w: torch.Tensor, outlier_signs: torch.Tensor) -> torch.Tensor:
    """Apply side-channel signs to outlier positions of an unpacked tensor."""
    dev = w.device
    w = w.cpu()
    mask = w.abs() > 1.5
    n = int(mask.sum().item())
    if n == 0 or outlier_signs is None or outlier_signs.numel() == 0:
        return w.to(dev)
    import numpy as np
    bits = outlier_signs.detach().cpu().flatten()
    nbytes = (n + 7) // 8
    if bits.numel() < nbytes:
        raise ValueError(f"outlier_signs too small: {bits.numel()} bytes for {n} outliers")
    sign_bits = np.unpackbits(bits[:nbytes].numpy())[:n]
    signs = torch.from_numpy(sign_bits.astype(np.float32) * 2.0 - 1.0)
    w = w.clone()
    w[mask] *= signs
    return w.to(dev)


def unpack_ternary_tensor(packed: torch.Tensor, shape: tuple,
                          outlier_signs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Unpack uint8 tensor -> float tensor {-2, -1, 0, +1, +2}.

    Code 11 (outlier) resolves its sign from ``outlier_signs`` bits.
    Uses C++ SIMD unpack on CPU when available (fastest path),
    falls back to Python element-wise ops on the original device.
    """
    if _has_cpp:
        # C++ unpack always runs on CPU, then moves to target device
        target = packed.device
        w = _ternary_ops.unpack_ternary(packed.cpu().contiguous(), list(shape))
        w = _apply_signs_tensor(w, outlier_signs)
        if target.type != "cpu":
            w = w.to(target)
        return w
    w0 = (torch.div(packed, 64, rounding_mode='floor') % 4).to(torch.int8) - 1
    w1 = (torch.div(packed, 16, rounding_mode='floor') % 4).to(torch.int8) - 1
    w2 = (torch.div(packed, 4, rounding_mode='floor') % 4).to(torch.int8) - 1
    w3 = (packed % 4).to(torch.int8) - 1
    flat = torch.stack([w0, w1, w2, w3], dim=-1).flatten()
    total = 1
    for d in shape:
        total *= d
    flat = flat[:total]
    return _apply_signs_tensor(flat.float().reshape(shape), outlier_signs)


def init_ternary_weight(out_features: int, in_features: int, sparsity: float = 0.5,
                        balanced: bool = False) -> torch.Tensor:
    """Initialize packed ternary weights with given sparsity.

    sparsity=0.5 -> 50% zeros, 25% +1, 25% -1 (kaiming-like).
    balanced=True -> equal 1/3 each of {-1, 0, +1} (33/33/33).
    Returns flat uint8 packed tensor.
    """
    n = out_features * in_features
    w = torch.zeros(n, dtype=torch.uint8)
    if balanced:
        each = n // 3
        w = torch.full((n,), 1, dtype=torch.uint8)  # 1 = value 0
        idx = torch.randperm(n)
        w[idx[:each]] = 2  # +1
        w[idx[each:2 * each]] = 0  # -1
        # remaining third stays 1 (= 0)
    else:
        nz = int(n * (1 - sparsity))  # non-zero count
        if nz > 0:
            pos = nz // 2
            neg = nz - pos
            idx = torch.randperm(n)
            w[idx[:pos]] = 2   # +1
            w[idx[pos:pos + neg]] = 0  # -1 (encoded as 0, i.e. value -1+1=0)
    # Pad and pack
    padded = (n + 3) // 4 * 4
    if padded != n:
        w = torch.nn.functional.pad(w, (0, padded - n), value=1)
    w = w.view(-1, 4)
    packed = (w[:, 0] << 6) | (w[:, 1] << 4) | (w[:, 2] << 2) | w[:, 3]
    return packed.contiguous()


class StochasticBitFlipLinear(torch.autograd.Function):
    """Stochastic Bit-Flip for ternary weights.

    Forward: ternary matmul with per-group alphas -> output.
    Backward: accumulate gradient into accumulator (bit flip deferred
              to model.apply_bit_flips() called every N steps)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        packed_flat: torch.Tensor,
        w_raw: torch.Tensor,
        scale: float,
        accumulator: torch.Tensor,
        threshold: float,
        alphas: Optional[torch.Tensor] = None,
        group_size: int = 0,
        acc_decay: float = 1.0,
        energy: bool = False,
    ) -> torch.Tensor:
        """Forward with pre-unpacked w_raw (cached by module).

        Args:
            x: input tensor (..., in_features)
            packed_flat: packed ternary weights (unused, kept for signature)
            w_raw: unpacked ternary weights (out_features, in_features) in {-1, 0, +1}
            scale: ternary weight scale factor
            accumulator: gradient accumulator tensor (same shape as w_raw)
            threshold: bit-flip threshold
            alphas: optional per-group (out_features, num_groups) or per-channel (out_features,)
            group_size: per-group block size (0 = scalar or per-channel)

        Returns:
            Output tensor (..., out_features)
        """
        ctx.save_for_backward(x)
        ctx.w_raw = w_raw
        ctx.scale = scale
        ctx.accumulator = accumulator
        ctx.group_size = group_size
        ctx.acc_decay = acc_decay
        ctx.energy = energy

        if alphas is not None and group_size > 0:
            ctx.alphas = alphas
            in_features = x.size(-1)
            num_groups = (in_features + group_size - 1) // group_size
            ctx.num_groups = num_groups
            # Fused group matmul: expand alphas to full in_features, scale w_raw once
            alphas_expanded = torch.repeat_interleave(alphas, group_size, dim=1)
            if in_features % group_size != 0:
                alphas_expanded = alphas_expanded[:, :in_features]
            w_scaled = w_raw * alphas_expanded
            return F.linear(x, w_scaled)
        elif alphas is not None:
            ctx.alphas = alphas
            return F.linear(x, w_raw) * alphas.unsqueeze(0)
        return F.linear(x, w_raw) * scale

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0].to(grad_output.dtype)
        in_features = x.size(-1)
        out_features = grad_output.size(-1)

        grad_output_flat = grad_output.reshape(-1, out_features)
        w_raw = ctx.w_raw.to(grad_output.dtype)

        if ctx.group_size > 0:
            num_groups = ctx.num_groups
            alphas = ctx.alphas
            bsz = x.shape[:-1]
            # Reconstruct expanded alphas and scaled weights
            alphas_expanded = torch.repeat_interleave(alphas, ctx.group_size, dim=1)
            if in_features % ctx.group_size != 0:
                alphas_expanded = alphas_expanded[:, :in_features]
            w_scaled = w_raw * alphas_expanded
            # Single fused matmul for grad_x and grad_w_scaled
            x_flat = x.reshape(-1, in_features)
            grad_x = torch.mm(grad_output_flat, w_scaled).view(*bsz, in_features)
            grad_w_scaled = torch.mm(grad_output_flat.T, x_flat)
            grad_w = grad_w_scaled * alphas_expanded
            # Scatter grad_alpha: sum over columns per group (still a loop, but trivial O(num_groups))
            grad_alpha = torch.zeros(out_features, num_groups, device=x.device)
            for g in range(num_groups):
                start = g * ctx.group_size
                end = min(start + ctx.group_size, in_features)
                grad_alpha[:, g] = (grad_w_scaled[:, start:end] * w_raw[:, start:end]).sum(dim=1)
            grad_alpha = grad_alpha.detach()
        elif hasattr(ctx, 'alphas') and ctx.alphas is not None:
            grad_y_scaled = grad_output_flat * ctx.alphas.unsqueeze(0)
            grad_alpha = (grad_output_flat * F.linear(x, w_raw)).sum(dim=0).detach()
            grad_x = torch.mm(grad_y_scaled, w_raw).view(*x.shape[:-1], in_features)
            grad_w = torch.mm(grad_y_scaled.T, x.reshape(-1, in_features))
        else:
            grad_y_raw = grad_output_flat * ctx.scale
            grad_x = torch.mm(grad_y_raw, w_raw).view(*x.shape[:-1], in_features)
            grad_w = torch.mm(grad_y_raw.T, x.reshape(-1, in_features))
            grad_alpha = None

        with torch.no_grad():
            if ctx.energy:
                # Energy accumulator: leaky EMA of the negative gradient.
                # acc = acc_decay*acc - grad_w (magnitude-weighted, sign-correct).
                # Requires adaptive threshold to stay scale-invariant (Exp 3).
                ctx.accumulator.mul_(ctx.acc_decay).add_(-grad_w)
            else:
                grad_w.sign_().neg_()
                ctx.accumulator.add_(grad_w)

        del grad_w, grad_output_flat

        return grad_x, None, None, None, None, None, grad_alpha, None, None, None


class Int8StochasticBitFlipLinear(torch.autograd.Function):
    """Stochastic Bit-Flip with INT8 activations.

    Forward: quantize x -> int8, matmul, dequantize.
    - CPU: uses C++ int8 ternary matmul kernel (no float multiplications).
    - DML/GPU: pure-PyTorch fallback (dequant+float matmul, same quantization noise).
    Backward: STE - grad flows through float matmul.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        packed_w: torch.Tensor,
        w_raw: torch.Tensor,
        scale: float,
        accumulator: torch.Tensor,
        threshold: float,
        outlier_signs: Optional[torch.Tensor] = None,
        acc_decay: float = 1.0,
        energy: bool = False,
    ) -> torch.Tensor:
        """Forward with INT8 quantized activations.

        Args:
            x: input tensor (..., in_features)
            packed_w: packed ternary weights
            w_raw: unpacked ternary weights (out_features, in_features)
            scale: ternary weight scale factor
            accumulator: gradient accumulator tensor
            threshold: bit-flip threshold
            outlier_signs: dense uint8 side-channel signs for code-11 outliers

        Returns:
            Output tensor (..., out_features)
        """
        max_abs = x.abs().max()
        scale_x = max_abs / 127.0 if max_abs > 1e-10 else 1.0
        x_q = (x / scale_x).round().clamp(-128, 127).to(torch.int8)

        ctx.save_for_backward(x.float())
        ctx.w_raw = w_raw
        ctx.scale = scale
        ctx.accumulator = accumulator
        ctx.acc_decay = acc_decay
        ctx.energy = energy

        # Grad carrier: float matmul so grad flows through x
        out = F.linear(x.float(), w_raw.float()) * scale * scale_x

        # Replace values with int8 matmul result (no grad contribution)
        with torch.no_grad():
            if x.device.type == "cpu" and _ternary_ops is not None:
                # Real int8 matmul via C++ kernel (fast on CPU)
                sign_args = (outlier_signs.cpu().contiguous()
                             if outlier_signs is not None else None)
                int_out = _ternary_ops.ternary_matmul_int8(
                    x_q.contiguous(), packed_w.contiguous(),
                    w_raw.size(0), w_raw.size(1),
                    sign_args,
                ).float()
            else:
                # Pure-PyTorch fallback: dequant -> float matmul
                # Same quantization noise, no CPU copies on DML
                int_out = F.linear(x_q.float() * scale_x, w_raw.float()) * scale
        out.data = int_out.to(x.device)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0].to(grad_output.dtype)
        scale = ctx.scale
        in_features = x.size(-1)

        grad_output_flat = grad_output.reshape(-1, grad_output.size(-1))
        grad_y_raw = grad_output_flat * scale
        w_raw = ctx.w_raw.to(grad_y_raw.dtype)

        grad_x_flat = torch.mm(grad_y_raw, w_raw)
        grad_x = grad_x_flat.view(*x.shape[:-1], in_features)

        grad_w = torch.mm(grad_y_raw.T, x.reshape(-1, x.size(-1)))
        with torch.no_grad():
            if ctx.energy:
                ctx.accumulator.mul_(ctx.acc_decay).add_(-grad_w)
            else:
                grad_w.sign_().neg_()
                ctx.accumulator.add_(grad_w)

        del grad_w

        return grad_x, None, None, None, None, None, None, None, None


@torch.no_grad()
def apply_bit_flips(
    packed_weights: torch.Tensor,
    accumulator: torch.Tensor,
    threshold: float,
    scale: float,
    shape_w: tuple,
    toggle: bool = False,
    outlier_signs: Optional[torch.Tensor] = None,
    outlier_thr_mult: float = 3.0,
        ungated: bool = False,
        stats: Optional[dict] = None,
        adaptive_thr: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Check accumulators and flip bits where threshold exceeded.

    Called externally every N steps instead of per-step in backward.
    Resets flipped accumulator entries to zero.

    v7 outlier dynamics (code 11 = ±2, sign in a dense side-channel):
      - promote:  |acc| > outlier_thr_mult × threshold and |w| ≤ 1 → ±2
                  (sign of acc); acc is parked at ±threshold so it stays
                  promoted until the leaky decay pulls it below threshold.
      - demote:   outlier pushed against its sign → ±1 (normal flip path);
                  outlier whose |acc| drops below threshold → ±1 toward acc.
      - outliers pushed in their own direction stay put.

    Exp 3 adaptive threshold: when ``adaptive_thr`` is not None, the flip
    threshold is computed per output channel as
    ``tau = adaptive_thr * RMS(accumulator, dim=in_features)`` instead of the
    fixed scalar ``threshold``. This makes the flip criterion scale-invariant
    (independent of gradient/logit scale), which is required when the
    accumulator holds magnitude-weighted gradient energy (``energy=True`` in
    the autograd Functions) rather than ±1 votes. tau is clamped to a small
    floor so idle channels (RMS ~ 0) don't flip on noise.

    Args:
        packed_weights: packed ternary weights (modified in-place)
        accumulator: gradient accumulator tensor (same shape as unpacked weights)
        threshold: flip threshold (accumulator values beyond ±threshold trigger flip)
        scale: weight scale factor (unused, kept for compatibility)
        shape_w: shape of the unpacked weight matrix (out_features, in_features)
        toggle: anti-stiction mode — a weight already saturated in the push
            direction (+1 pushed up / -1 pushed down) is kicked to the
            opposite extreme instead of staying a no-op.
        outlier_signs: dense uint8 sign blob for code-11 weights (row-major
            scan order, MSB first, 1=positive). None = v6 behavior (no outliers).
        outlier_thr_mult: promotion threshold multiplier (default 3.0).
        ungated: flip on every accumulator sign, ignoring the threshold.
        stats: optional dict accumulating flip counters ("flips", "n_calls").
        adaptive_thr: Exp 3 — if set, use k·RMS(acc) per output channel as the
            flip threshold (None = fixed scalar ``threshold``).

    Returns:
        The rebuilt sign blob tensor (uint8) when ``outlier_signs`` is not
        None, else None. Callers must store it back (outliers moved).
    """
    if adaptive_thr is not None:
        # Per-output-channel RMS of the accumulator = typical gradient energy.
        # tau = k * RMS, clamped so near-idle channels require a small but
        # non-zero energy to flip (avoids flip-on-noise for RMS ~ 0).
        rms = accumulator.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-4)
        thr = adaptive_thr * rms
        promote_thr = thr * outlier_thr_mult
        demote_thr = thr
        flip_up = accumulator > thr
        flip_down = accumulator < -thr
    else:
        thr = threshold
        promote_thr = threshold * outlier_thr_mult
        demote_thr = threshold
        flip_up = accumulator > threshold
        flip_down = accumulator < -threshold
    if ungated:
        flip_up = accumulator > 0
        flip_down = accumulator < 0
    promote = accumulator.abs() > promote_thr
    if not (flip_up.any() or flip_down.any() or promote.any()):
        return None
    w_raw = unpack_ternary_tensor(packed_weights, shape_w, outlier_signs)
    is_std = w_raw.abs() <= 1.5
    promote &= is_std
    out_mask = ~is_std
    flip_dir = torch.where(flip_up, 1.0, 0.0) + torch.where(flip_down, -1.0, 0.0)
    w_new = (w_raw + flip_dir).clamp(-1, 1)
    if toggle:
        sat_up = (w_raw > 0.5) & flip_up
        sat_down = (w_raw < -0.5) & flip_down
        if sat_up.any() or sat_down.any():
            w_new = torch.where(
                sat_up, torch.full_like(w_new, -1.0),
                torch.where(sat_down, torch.full_like(w_new, 1.0), w_new))
    if promote.any():
        w_new = torch.where(promote, torch.where(accumulator > 0, 2.0, -2.0), w_new)
    # Outliers pushed in their own direction stay as outliers
    same_dir = (w_raw > 0) & flip_up | (w_raw < 0) & flip_down
    if same_dir.any():
        w_new = torch.where(out_mask & same_dir, w_raw, w_new)
    # Outliers whose accumulator relaxed below threshold demote to ±1
    demote = out_mask & (accumulator.abs() < demote_thr)
    if demote.any():
        direction = torch.where(
            accumulator > 0, 1.0,
            torch.where(accumulator < 0, -1.0,
                        torch.where(w_raw > 0, 1.0, -1.0)))
        w_new = torch.where(demote, direction, w_new)
    packed_weights.copy_(pack_ternary_tensor(w_new).to(packed_weights.device))
    if stats is not None:
        stats["flips"] = stats.get("flips", 0) + int((w_new != w_raw).sum().item())
        stats["n_calls"] = stats.get("n_calls", 0) + 1
    promote_sign = accumulator[promote].sign() if promote.any() else None
    accumulator[flip_up | flip_down] = 0.0
    if promote.any():
        if isinstance(demote_thr, torch.Tensor):
            # Per-channel parking: park each promoted weight at its channel's
            # base threshold (stays promoted until leaky decay pulls it below
            # the same threshold, i.e. demote).
            promote_idx = promote.nonzero()
            park = demote_thr[promote_idx[:, 0], 0]
            accumulator[promote_idx[:, 0], promote_idx[:, 1]] = promote_sign * park
        else:
            accumulator[promote] = promote_sign * demote_thr
    if demote.any():
        accumulator[demote] = 0.0
    if outlier_signs is not None:
        return pack_sign_blob_tensor(w_new).to(packed_weights.device)
    return None
