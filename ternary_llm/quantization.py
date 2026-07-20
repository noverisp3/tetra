import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TernaryQuantizer(torch.autograd.Function):
    """Ternary quantization {-1, 0, +1} with Straight-Through Estimator.

    Dynamic threshold: Δ = scale × mean(|W|), default scale=0.7.
    Δ được scale xuống để tăng entropy, giảm số lượng zero trong ma trận
    ternary, tránh suy thoái mô hình (toàn bộ weight → 0).

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
        w_ternary = (latent_weights / delta).clamp(-1, 1).round()
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
        return grad_x, grad_w, None, None


class Int8Quantizer(torch.autograd.Function):
    """Symmetric INT8 quantization for activations (fake-quantize).

    Forward:  scale = max(|x|) / 127, quantize to [-128, 127], dequantize back to float
    Backward: STE passes grad through as-is

    Output stays float to maintain gradient flow during training.
    Real INT8 conversion happens at inference time.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        scale = input.abs().max() / 127.0
        scale = scale.clamp(min=1e-6)

        # Fake quantize: quantize then dequantize (keeps float, simulates INT8)
        q_dequant = (input / scale).round().clamp(-128, 127) * scale

        ctx.save_for_backward(input, scale)
        return q_dequant

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # STE: pass gradient through as-is
        return grad_output


def ternary_quantize(weights: torch.Tensor) -> torch.Tensor:
    """Apply ternary quantization to a weight tensor (inference)."""
    return TernaryQuantizer.apply(weights)


def int8_quantize(activations: torch.Tensor) -> torch.Tensor:
    """Apply INT8 quantization to activation tensor."""
    return Int8Quantizer.apply(activations)


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


# ─── Fast Tensor Pack/Unpack cho Stochastic Bit-Flip ───────────────────────

def pack_ternary_tensor(w: torch.Tensor) -> torch.Tensor:
    """Pack ternary float tensor {-1, 0, +1} → uint8 tensor (4 weights/byte)."""
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
    """Unpack uint8 tensor → float tensor {-1, 0, +1}."""
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
    # Pad và pack
    padded = (n + 3) // 4 * 4
    if padded != n:
        w = torch.nn.functional.pad(w, (0, padded - n), value=1)
    w = w.view(-1, 4)
    packed = (w[:, 0] << 6) | (w[:, 1] << 4) | (w[:, 2] << 2) | w[:, 3]
    return packed.contiguous()


# ─── Stochastic Bit-Flip Autograd ─────────────────────────────────────────

class StochasticBitFlipLinear(torch.autograd.Function):
    """Stochastic Bit-Flip training cho ternary weights.

    Forward: unpack 2-bit → ternary float → scale → matmul
    Backward: accumulate gradient → lật bit khi vượt threshold
    """

    @staticmethod
    def forward(ctx, x, packed_flat, shape_w, scale, accumulator, threshold):
        w_ternary = unpack_ternary_tensor(packed_flat, shape_w).to(x.dtype) * scale
        ctx.save_for_backward(x)
        ctx.packed_flat = packed_flat
        ctx.shape_w = shape_w
        ctx.scale = scale
        ctx.accumulator = accumulator
        ctx.threshold = threshold
        return F.linear(x, w_ternary)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        scale = ctx.scale
        in_features = x.size(-1)

        w_ternary = unpack_ternary_tensor(ctx.packed_flat, ctx.shape_w).to(x.dtype) * scale
        grad_output_flat = grad_output.reshape(-1, grad_output.size(-1))
        grad_x_flat = torch.mm(grad_output_flat, w_ternary)
        grad_x = grad_x_flat.view(*x.shape[:-1], in_features)

        grad_w = torch.mm(
            grad_output_flat.T,
            x.reshape(-1, x.size(-1))
        )

        with torch.no_grad():
            # Gradient sign: chi direction, bo magnitude → on dinh, khong phu thuoc batch size
            ctx.accumulator.add_(torch.sign(-grad_w) / scale)
            flip_up = ctx.accumulator > ctx.threshold
            flip_down = ctx.accumulator < -ctx.threshold

            if flip_up.any() or flip_down.any():
                w_curr = unpack_ternary_tensor(ctx.packed_flat, ctx.shape_w).to(x.device)
                flip_dir = torch.where(flip_up, 1.0, 0.0) + torch.where(flip_down, -1.0, 0.0)
                w_new = (w_curr + flip_dir).clamp(-1, 1)
                ctx.packed_flat.copy_(pack_ternary_tensor(w_new).to(ctx.packed_flat.device))
                ctx.accumulator[flip_up | flip_down] = 0.0

        return grad_x, None, None, None, None, None
