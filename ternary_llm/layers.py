import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .quantization import FusedTernaryLinear, TernaryQuantizer, StochasticBitFlipLinear, init_ternary_weight


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
        return (x_f * rms).type_as(x) * self.weight


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
    """Ternary linear layer với Stochastic Bit-Flip training.

    Không có latent weights FP32. Trọng số luôn là ternary {-1,0,+1}
    dưới dạng packed 2-bit. Gradient tích lũy vào accumulator, flip khi
    vượt threshold.

    Args:
        in_features: input dimension
        out_features: output dimension
        bias: whether to use bias
        scale: ternary weight scale factor (default: 1.0)
        threshold: flip threshold (default: 20.0 / scale, auto-computed)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        scale: float = 1.0,
        threshold: float | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        # Auto-threshold: threshold * scale = 20 (invariant)
        self.threshold = 20.0 / scale if threshold is None else threshold

        # Packed 2-bit ternary weights (4 weights/byte)
        packed = init_ternary_weight(out_features, in_features, sparsity=0.5)
        self.register_buffer('packed_weights', packed)

        # Gradient accumulator (FP32)
        self.register_buffer('accumulator', torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = StochasticBitFlipLinear.apply(
            x, self.packed_weights, (self.out_features, self.in_features),
            self.scale, self.accumulator, self.threshold
        )
        if self.bias is not None:
            output = output + self.bias
        return output

    @torch.no_grad()
    def get_ternary_weights(self) -> torch.Tensor:
        from .quantization import unpack_ternary_tensor
        return unpack_ternary_tensor(self.packed_weights, (self.out_features, self.in_features))

    def get_num_bits(self) -> int:
        return self.out_features * self.in_features * 2  # 2 bits per weight
