import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TernaryLinear
from .quantization import FusedTernaryLinear


class TernaryFFN(nn.Module):
    """Feed-Forward Network with ternary weights.

    Architecture (SwiGLU variant, following modern LLMs):
        - Gate projection: ternary weights
        - Up projection: ternary weights
        - Down projection: ternary weights
        - Activation: SiLU (float, not quantized)

    SwiGLU: output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    Fused gate+up: single ternary matmul with 2*ffn_dim output, then chunk.

    Args:
        hidden_dim: model dimension
        ffn_dim: feed-forward hidden dimension (typically 4 * hidden_dim)
        dropout: dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        ternary_scale: float = 0.7,
        per_channel: bool = False,
    ):
        super().__init__()

        # Ternary projections
        self.gate_proj = TernaryLinear(hidden_dim, ffn_dim, ternary_scale=ternary_scale, per_channel=per_channel)
        self.up_proj = TernaryLinear(hidden_dim, ffn_dim, ternary_scale=ternary_scale, per_channel=per_channel)
        self.down_proj = TernaryLinear(ffn_dim, hidden_dim, ternary_scale=ternary_scale, per_channel=per_channel)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gate+up: concat latent weights → one quantize + one matmul
        w_fused = torch.cat(
            [self.gate_proj.latent_weights, self.up_proj.latent_weights], dim=0
        )
        fused_out = FusedTernaryLinear.apply(x, w_fused, self.gate_proj.ternary_scale, self.gate_proj.per_channel)
        gate, up = fused_out.chunk(2, dim=-1)

        # SwiGLU activation
        hidden = F.silu(gate) * up

        # Down projection
        output = self.down_proj(hidden)
        output = self.dropout(output)

        return output


class StochasticFFN(nn.Module):
    """FFN với Stochastic Bit-Flip (không latent weights, packed 2-bit)."""

    def __init__(self, hidden_dim, ffn_dim, dropout=0.0, scale=1.0, threshold=None):
        super().__init__()
        from .layers import StochasticTernaryLinear
        self.gate_proj = StochasticTernaryLinear(hidden_dim, ffn_dim, scale=scale, threshold=threshold)
        self.up_proj = StochasticTernaryLinear(hidden_dim, ffn_dim, scale=scale, threshold=threshold)
        self.down_proj = StochasticTernaryLinear(ffn_dim, hidden_dim, scale=scale, threshold=threshold)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        output = self.down_proj(hidden)
        return self.dropout(output)
