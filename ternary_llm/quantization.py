import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TernaryQuantizer(torch.autograd.Function):
    """Ternary quantization {-1, 0, +1} with Straight-Through Estimator.

    Absmean quantization: threshold γ = mean(|W|)
    Forward:  clamp(W/γ, -1, 1) → round → {-1, 0, +1}
    Backward: STE passes grad straight through
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        # Absmean threshold
        gamma = input.abs().mean()
        if gamma < 1e-6:
            gamma = torch.tensor(1e-6, device=input.device, dtype=input.dtype)

        # Normalize, clamp to [-1, 1], then round to ternary
        w_normalized = input / gamma
        w_ternary = w_normalized.clamp(-1, 1).round()

        ctx.save_for_backward(input, gamma)
        return w_ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, gamma = ctx.saved_tensors
        # STE: pass gradient straight through (standard BitNet approach)
        # The clamping in forward already handles saturation;
        # masking here would prevent weights from recovering once they drift.
        return grad_output.clone()


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
        if scale < 1e-6:
            scale = torch.tensor(1e-6, device=input.device, dtype=input.dtype)

        # Fake quantize: quantize then dequantize (keeps float, simulates INT8)
        q = (input / scale).round().clamp(-128, 127)
        q_dequant = q * scale

        ctx.save_for_backward(input, scale)
        return q_dequant

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # STE: pass gradient through as-is
        return grad_output.clone()


def ternary_quantize(weights: torch.Tensor) -> torch.Tensor:
    """Apply ternary quantization to a weight tensor (inference)."""
    return TernaryQuantizer.apply(weights)


def int8_quantize(activations: torch.Tensor) -> torch.Tensor:
    """Apply INT8 quantization to activation tensor."""
    return Int8Quantizer.apply(activations)
