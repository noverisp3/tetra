import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os, sys

# Optional C++ SIMD extension for fast pack/unpack
_ternary_ops = None

def _try_load_from_cache(name):
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

def _load_cpp_extension():
    """Load C++ ternary_ops extension with CPU-capability dispatch.

    Tries (in order):
      1. AVX-512: direct cache load or torch.load for ternary_ops_avx512
      2. AVX2:    direct cache load or torch.load for ternary_ops (baseline)
      3. Returns False if neither works
    """
    global _ternary_ops
    if _ternary_ops is not None:
        return True

    csrc = os.path.join(os.path.dirname(__file__), "csrc")

    # Attempt 1: AVX-512 variant (best perf)
    # Check CPU for AVX-512 support via Windows API
    def _has_avx512():
        try:
            if sys.platform == 'win32':
                import ctypes
                return ctypes.windll.kernel32.IsProcessorFeaturePresent(0x17) != 0
        except Exception:
            pass
        return False

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
                    extra_cflags=["/arch:AVX512"], verbose=False,
                )
                return True
            except Exception:
                pass

    # Attempt 2: AVX2 variant (fallback)
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
            extra_cflags=["/arch:AVX2"], verbose=False,
        )
        return True
    except Exception:
        return False

_has_cpp = _load_cpp_extension()


class TernaryQuantizer(torch.autograd.Function):
    """Ternary quantization {-1, 0, +1} with Straight-Through Estimator.

    Dynamic threshold: Δ = scale × mean(|W|), default scale=0.7.
    Lower scale increases entropy (fewer zeros in ternary matrix),
    preventing model collapse (all weights → 0).

    Forward:  clamp(W/Δ, -1, 1) → round → {-1, 0, +1}
    Backward: STE passes grad straight through
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, scale: float = 0.7) -> torch.Tensor:
        ctx.scale = scale
        delta = input.abs().mean().clamp(min=1e-6) * scale
        w_ternary = (input / delta).clamp(-1, 1).round()
        ctx.save_for_backward(input)
        return w_ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output, None


class _quantize_ternary:
    """Ternary quantize with computed threshold Δ = scale × mean(|W|)."""

    @staticmethod
    def forward(input: torch.Tensor, scale: float = 0.7) -> torch.Tensor:
        delta = input.abs().mean().clamp(min=1e-6) * scale
        return (input / delta).clamp(-1, 1).round()

    @staticmethod
    def per_channel(input: torch.Tensor, scale: float = 0.7) -> torch.Tensor:
        delta = input.abs().mean(dim=1, keepdim=True).clamp(min=1e-6) * scale
        return (input / delta).clamp(-1, 1).round()


class FusedTernaryLinear(torch.autograd.Function):
    """Fused ternary quantization + linear matmul.

    Dynamic threshold: Δ = scale × mean(|W|), default scale=0.7.
    Per-channel: Δ_i = scale × mean(|W[i,:]|).

    Forward:  clamp(W/Δ, -1, 1) → round → matmul(x, W_ternary)
    Backward: grad_x = grad @ W_ternary.T, grad_W = x^T @ grad (STE)

    Saves ternary weights to avoid recomputing abs()
    in backward — avoids OOM on memory-constrained devices.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, latent_weights: torch.Tensor,
                scale: float = 0.7, per_channel: bool = False) -> torch.Tensor:
        if per_channel:
            delta = latent_weights.abs().mean(dim=1, keepdim=True).clamp(min=1e-6) * scale
        else:
            delta = latent_weights.abs().mean().clamp(min=1e-6) * scale
        w_ternary = (latent_weights / delta).clamp(-1, 1).round().to(x.dtype)
        ctx.save_for_backward(x, w_ternary.detach())
        return F.linear(x, w_ternary)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, w_ternary = ctx.saved_tensors
        grad_x = F.linear(grad_output, w_ternary.T)
        grad_w = torch.mm(
            grad_output.reshape(-1, grad_output.size(-1)).T,
            x.reshape(-1, x.size(-1))
        )
        return grad_x, grad_w.float(), None, None


def ternary_quantize(weights: torch.Tensor) -> torch.Tensor:
    """Apply ternary quantization to a weight tensor (inference)."""
    return TernaryQuantizer.apply(weights)


def pack_ternary(weights: torch.Tensor) -> bytes:
    """Pack ternary weights {-1, 0, +1} to 2-bit format (4 weights/byte).

    Encoding: 00=-1, 01=0, 10=+1, MSB first.
    Input: flat float tensor with values in {-1, 0, +1}.
    Output: packed bytes.
    """
    w = weights.detach().cpu().flatten().to(torch.int8)
    # Map {-1,0,+1} -> {0,1,2}
    w = (w + 1).to(torch.uint8)  # -1->0, 0->1, +1->2
    n = len(w)
    # Pad to multiple of 4
    padded = (n + 3) // 4 * 4
    if padded != n:
        w = torch.nn.functional.pad(w, (0, padded - n), value=1)  # pad with 0 (encoded as 1)
    w = w.view(-1, 4)
    # Pack: MSB first, 2 bits per weight
    packed = (w[:, 0].to(torch.uint8) << 6 |
              w[:, 1].to(torch.uint8) << 4 |
              w[:, 2].to(torch.uint8) << 2 |
              w[:, 3].to(torch.uint8))
    return bytes(packed.tolist())


def unpack_ternary(packed: bytes, shape: tuple, device: str = "cpu") -> torch.Tensor:
    """Unpack 2-bit packed ternary weights back to {-1, 0, +1} float tensor.

    Encoding: 00=-1, 01=0, 10=+1, MSB first.
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
    return torch.from_numpy(flat.astype(np.float32)).reshape(shape).to(device)


# Fast Tensor Pack/Unpack for Stochastic Bit-Flip

def pack_ternary_tensor(w: torch.Tensor) -> torch.Tensor:
    """Pack ternary float tensor {-1, 0, +1} → uint8 tensor (4 weights/byte)."""
    if _has_cpp and w.is_cpu and w.dtype in (torch.float32, torch.float16):
        return _ternary_ops.pack_ternary(w.contiguous())
    w_u8 = (w + 1).to(torch.uint8)  # -1→0, 0→1, +1→2
    flat = w_u8.flatten()
    n = flat.size(0)
    padded = (n + 3) // 4 * 4
    if padded != n:
        flat = torch.nn.functional.pad(flat, (0, padded - n), value=1)
    flat = flat.view(-1, 4)
    packed = flat[:, 0] * 64 + flat[:, 1] * 16 + flat[:, 2] * 4 + flat[:, 3]
    return packed.contiguous().to(torch.uint8)


def unpack_ternary_tensor(packed: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Unpack uint8 tensor → float tensor {-1, 0, +1}.

    Uses C++ SIMD unpack on CPU when available (fastest path),
    falls back to Python element-wise ops on the original device.
    """
    if _has_cpp:
        # C++ unpack always runs on CPU, then moves to target device
        target = packed.device
        w = _ternary_ops.unpack_ternary(packed.cpu().contiguous(), list(shape))
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
    return flat.float().reshape(shape)


def init_ternary_weight(out_features: int, in_features: int, sparsity: float = 0.5) -> torch.Tensor:
    """Initialize packed ternary weights with given sparsity.

    sparsity=0.5 → 50% zeros, 25% +1, 25% -1 (kaiming-like).
    Returns flat uint8 packed tensor.
    """
    n = out_features * in_features
    nz = int(n * (1 - sparsity))  # non-zero count
    w = torch.zeros(n, dtype=torch.uint8)
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


# INT8 Autograd Helper

class Int8QuantizeSTE(torch.autograd.Function):
    """Quantize float → int8 with Straight-Through Estimator for backward."""
    @staticmethod
    def forward(ctx, x, scale_x):
        ctx.scale_x = scale_x
        return (x / scale_x).round().clamp(-128, 127).to(torch.int8)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.float(), None

class StochasticBitFlipLinear(torch.autograd.Function):
    """Stochastic Bit-Flip for ternary weights.

    Forward: unpack 2-bit → ternary float → scale → matmul
    Backward: accumulate gradient into accumulator (bit flip deferred
              to model.apply_bit_flips() called every N steps)
    """

    @staticmethod
    def forward(ctx, x, packed_flat, w_raw, scale, accumulator, threshold):
        """Forward with pre-unpacked w_raw (cached by module, avoids 1.2s unpack/step).

        w_raw: float tensor of shape (out_features, in_features), values in {-1, 0, +1}.
        """
        ctx.save_for_backward(x)
        ctx.w_raw = w_raw
        ctx.scale = scale
        ctx.accumulator = accumulator
        return F.linear(x, w_raw) * scale

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        scale = ctx.scale
        in_features = x.size(-1)

        # Forward computes: output = F.linear(x, w_raw) * scale
        # Backward: grad_y_raw = grad_output * scale (2.6M elements vs 500M for w_ternary)
        grad_output_flat = grad_output.reshape(-1, grad_output.size(-1))
        grad_y_raw = grad_output_flat * scale

        grad_x_flat = torch.mm(grad_y_raw, ctx.w_raw)
        grad_x = grad_x_flat.view(*x.shape[:-1], in_features)

        grad_w = torch.mm(grad_y_raw.T, x.reshape(-1, x.size(-1)))

        with torch.no_grad():
            grad_w.sign_().neg_()
            ctx.accumulator.add_(grad_w)

        del grad_w, grad_y_raw, grad_output_flat

        return grad_x, None, None, None, None, None


class Int8StochasticBitFlipLinear(torch.autograd.Function):
    """Stochastic Bit-Flip with INT8 activations.

    Forward: quantize x → int8, matmul, dequantize.
    - CPU: uses C++ int8 ternary matmul kernel (no float multiplications).
    - DML/GPU: pure-PyTorch fallback (dequant+float matmul, same quantization noise).
    Backward: STE — grad flows through float matmul.
    """

    @staticmethod
    def forward(ctx, x, packed_w, w_raw, scale, accumulator, threshold):
        max_abs = x.abs().max()
        scale_x = max_abs / 127.0 if max_abs > 1e-10 else 1.0
        x_q = (x / scale_x).round().clamp(-128, 127).to(torch.int8)

        ctx.save_for_backward(x.float())
        ctx.w_raw = w_raw
        ctx.scale = scale
        ctx.accumulator = accumulator

        # Grad carrier: float matmul so grad flows through x
        out = F.linear(x.float(), w_raw.float()) * scale * scale_x

        # Replace values with int8 matmul result (no grad contribution)
        with torch.no_grad():
            if x.device.type == "cpu" and _ternary_ops is not None:
                # Real int8 matmul via C++ kernel (fast on CPU)
                int_out = _ternary_ops.ternary_matmul_int8(
                    x_q.contiguous(), packed_w.contiguous(),
                    w_raw.size(0), w_raw.size(1),
                ).float()
            else:
                # Pure-PyTorch fallback: dequant → float matmul
                # Same quantization noise, no CPU copies on DML
                int_out = F.linear(x_q.float() * scale_x, w_raw.float()) * scale
        out.data = int_out.to(x.device)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        scale = ctx.scale
        in_features = x.size(-1)

        grad_output_flat = grad_output.reshape(-1, grad_output.size(-1))
        grad_y_raw = grad_output_flat * scale

        grad_x_flat = torch.mm(grad_y_raw, ctx.w_raw)
        grad_x = grad_x_flat.view(*x.shape[:-1], in_features)

        grad_w = torch.mm(grad_y_raw.T, x.reshape(-1, x.size(-1)))
        with torch.no_grad():
            grad_w.sign_().neg_()
            ctx.accumulator.add_(grad_w)

        return grad_x, None, None, None, None, None


@torch.no_grad()
def apply_bit_flips(packed_weights: torch.Tensor, accumulator: torch.Tensor,
                     threshold: float, scale: float, shape_w: tuple) -> None:
    """Check accumulators and flip bits where threshold exceeded.

    Called externally every N steps instead of per-step in backward.
    Resets flipped accumulator entries to zero.
    """
    flip_up = accumulator > threshold
    flip_down = accumulator < -threshold
    if flip_up.any() or flip_down.any():
        w_raw = unpack_ternary_tensor(packed_weights, shape_w)
        flip_dir = torch.where(flip_up, 1.0, 0.0) + torch.where(flip_down, -1.0, 0.0)
        w_new = (w_raw + flip_dir).clamp(-1, 1)
        packed_weights.copy_(pack_ternary_tensor(w_new).to(packed_weights.device))
        accumulator[flip_up | flip_down] = 0.0
