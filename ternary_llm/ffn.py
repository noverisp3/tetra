import torch
__all__ = [
    "TernaryFFN", "StochasticFFN",
]

import torch.nn as nn
import torch.nn.functional as F
from .layers import TernaryLinear

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

        # Fused gate+up: single ternary linear with 2*ffn_dim output
        self.gate_up_proj = TernaryLinear(hidden_dim, 2 * ffn_dim, ternary_scale=ternary_scale, per_channel=per_channel)
        self.down_proj = TernaryLinear(ffn_dim, hidden_dim, ternary_scale=ternary_scale, per_channel=per_channel)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gate+up: one matmul → chunk
        fused_out = self.gate_up_proj(x)
        gate, up = fused_out.chunk(2, dim=-1)

        # SwiGLU activation (float32 to prevent overflow in down_proj)
        hidden = F.silu(gate).float() * up.float()

        # Down projection
        output = self.down_proj(hidden)
        output = self.dropout(output)

        return output


class StochasticFFN(nn.Module):
    """FFN with Stochastic Bit-Flip (no latent weights, packed 2-bit)."""

    def __init__(self, hidden_dim, ffn_dim, dropout=0.0, scale=1.0, threshold=None, int8=False):
        super().__init__()
        from .layers import StochasticTernaryLinear
        self.gate_proj = StochasticTernaryLinear(hidden_dim, ffn_dim, scale=scale, threshold=threshold, int8=int8)
        self.up_proj = StochasticTernaryLinear(hidden_dim, ffn_dim, scale=scale, threshold=threshold, int8=int8)
        self.down_proj = StochasticTernaryLinear(ffn_dim, hidden_dim, scale=scale, threshold=threshold, int8=int8)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate).float() * up.float()
        output = self.down_proj(hidden)
        return self.dropout(output)

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        self.gate_proj.apply_bit_flips()
        self.up_proj.apply_bit_flips()
        self.down_proj.apply_bit_flips()
