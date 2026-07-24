import math
__all__ = [
    "RMSNorm", "TernaryLinear", "StochasticTernaryLinear", "TopKActivation",
]

import torch
import torch.nn as nn
from .quantization import (
    FusedTernaryLinear,
    Int8StochasticBitFlipLinear,
    StochasticBitFlipLinear,
    TernaryQuantizer,
    _ternary_ops,
    init_ternary_weight,
    unpack_ternary_tensor,
)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm, no mean centering.
    Formula: y = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2))
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = (x_f * rms) * self.weight
        return out.to(x.dtype)


class TernaryLinear(nn.Module):
    """Linear layer with ternary weight quantization.

    During training:
        - Stores latent weights (FP32) as nn.Parameter
        - Applies ternary quantization in forward pass via STE

    During inference:
        - Only ternary weights {-1, 0, +1} needed
        - No floating point multiplication

    Args:
        in_features: input dimension
        out_features: output dimension
        bias: whether to use bias (default: False, following BitNet)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        ternary_scale: float = 0.7,
        per_channel: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ternary_scale = ternary_scale
        self.per_channel = per_channel

        # Latent weights (FP32) - shadow weights for gradient updates
        self.latent_weights = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.latent_weights, a=math.sqrt(5))

        # No bias following BitNet design
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = FusedTernaryLinear.apply(x, self.latent_weights, self.ternary_scale, self.per_channel)
        if self.bias is not None:
            output = output + self.bias
        return output

    def get_ternary_weights(self) -> torch.Tensor:
        """Get quantized ternary weights for deployment."""
        return TernaryQuantizer.apply(self.latent_weights.data)

    def get_num_bits(self) -> int:
        """Calculate total bits for ternary weights."""
        return math.ceil(math.log2(3)) * self.latent_weights.numel()


class StochasticTernaryLinear(nn.Module):
    """Ternary linear layer with Stochastic Bit-Flip training.

    No latent weights FP32. Weights are always ternary {-1,0,+1}
    in packed 2-bit format. Gradient accumulates in accumulator, flips when
    threshold is exceeded.

    Args:
        in_features: input dimension
        out_features: output dimension
        bias: whether to use bias
        scale: ternary weight scale factor (default: 1.0)
        threshold: flip threshold (default: 20.0 / scale, auto-computed)
        per_channel: per-output-channel alphas (default: False)
        group_size: block size for per-group alpha (0=disabled, >0 enables per-group).
                    When group_size > 0, num_groups = ceil(in_features / group_size).
                    Overrides per_channel. (default: 0)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        scale: float = 1.0,
        threshold: float | None = None,
        int8: bool = False,
        per_channel: bool = False,
        group_size: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.int8 = int8
        self.per_channel = per_channel or (group_size > 0)
        # Integer threshold (20). Accumulator adds ±1 per step, threshold=20 means
        # flip after ~20 gradient steps. This replaces threshold = 20/scale from
        # the scaled-accumulator approach, avoiding /scale on every backward pass.
        self.threshold = 20.0 if threshold is None else threshold

        # Packed 2-bit ternary weights (4 weights/byte)
        packed = init_ternary_weight(out_features, in_features, sparsity=0.5)
        self.register_buffer('packed_weights', packed)

        # Gradient accumulator (FP32)
        self.register_buffer('accumulator', torch.zeros(out_features, in_features))

        # Per-group / per-channel scaling alphas (FP32, trainable)
        if group_size > 0:
            self.group_size = group_size
            self.num_groups = (in_features + group_size - 1) // group_size
            self.alphas = nn.Parameter(torch.full((out_features, self.num_groups), scale))
        elif per_channel:
            self.group_size = 0
            self.num_groups = 1
            self.alphas = nn.Parameter(torch.full((out_features,), scale))
        else:
            self.group_size = 0
            self.num_groups = 0
            self.register_parameter("alphas", None)

        # Cached unpacked weights (recomputed after apply_bit_flips)
        self._w_raw_cache = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._w_raw_cache is None:
            self._w_raw_cache = unpack_ternary_tensor(
                self.packed_weights, (self.out_features, self.in_features)
            ).to(x.dtype)

        if self.int8 and _ternary_ops is not None:
            output = Int8StochasticBitFlipLinear.apply(
                x, self.packed_weights, self._w_raw_cache,
                self.scale, self.accumulator, self.threshold
            )
        else:
            output = StochasticBitFlipLinear.apply(
                x, self.packed_weights, self._w_raw_cache,
                self.scale, self.accumulator, self.threshold,
                self.alphas, self.group_size
            )

        if self.bias is not None:
            output = output + self.bias
        return output

    @torch.no_grad()
    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        """Check accumulator and flip bits where threshold exceeded."""
        from .quantization import apply_bit_flips as _apply_bit_flips
        _apply_bit_flips(
            self.packed_weights, self.accumulator,
            self.threshold, self.scale,
            (self.out_features, self.in_features)
        )
        # Invalidate cached unpacked weights (packed weights changed)
        self._w_raw_cache = None

    @torch.no_grad()
    def get_ternary_weights(self) -> torch.Tensor:
        if self._w_raw_cache is not None:
            return self._w_raw_cache
        return unpack_ternary_tensor(self.packed_weights, (self.out_features, self.in_features))

    def get_num_bits(self) -> int:
        return self.out_features * self.in_features * 2  # 2 bits per weight


class TopKActivation(nn.Module):
    """Keep top-k% activations, zero the rest. STE backward.

    Forward: mask = score > threshold_of_top_k, output = input * mask
    Backward: grad flows to all positions (allows recovery).
    """

    def __init__(self, keep_ratio: float = 0.2):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.keep_ratio >= 1.0 or not self.training:
            return x
        x_f = x.float()
        k = max(1, int(x_f.size(-1) * self.keep_ratio))
        # threshold: (..., 1) via take top-kth value along last dim
        vals = x_f.abs().topk(k, dim=-1).values
        threshold = vals[..., -1:]  # (..., 1), the k-th largest absolute value
        mask = (x_f.abs() >= threshold).to(x.dtype)
        return x * mask

    def extra_repr(self) -> str:
        return f"keep_ratio={self.keep_ratio}"
