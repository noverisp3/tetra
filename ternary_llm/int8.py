"""INT8 fake-quantization for activations (STE-based, not used in model training).

Int8Quantizer provides symmetric INT8 fake-quantization with STE backward.
Moved from quantization.py since it is not called by the model.
Kept as reference for possible future INT8 activation quantization experiments.
"""

import torch


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


def int8_quantize(activations: torch.Tensor) -> torch.Tensor:
    """Apply INT8 quantization to activation tensor."""
    return Int8Quantizer.apply(activations)
