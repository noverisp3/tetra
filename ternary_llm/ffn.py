"""Feed-Forward Network modules with ternary weights.

Implements SwiGLU FFN variant (following modern LLMs) with two training modes:
- TernaryFFN: STE-trained ternary weights (absmean quantization)
- StochasticFFN: Stochastic bit-flip training (packed 2-bit, no latent FP32)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = ["TernaryFFN", "StochasticFFN"]


class TernaryFFN(nn.Module):
    """SwiGLU Feed-Forward Network with ternary weights (STE training).

    Architecture:
        output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    Gate and up projections are fused into a single ternary matmul with
    2*ffn_dim output, then chunked.

    Args:
        hidden_dim: model dimension (input/output)
        ffn_dim: feed-forward hidden dimension (typically 4 * hidden_dim)
        dropout: dropout rate on output
        ternary_scale: scale factor for ternary quantization
        per_channel: per-output-channel alpha scaling
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

        self.gate_up_proj = TernaryLinear(
            hidden_dim, 2 * ffn_dim,
            ternary_scale=ternary_scale, per_channel=per_channel,
        )
        self.down_proj = TernaryLinear(
            ffn_dim, hidden_dim,
            ternary_scale=ternary_scale, per_channel=per_channel,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor (..., hidden_dim)

        Returns:
            Output tensor (..., hidden_dim)
        """
        fused_out = self.gate_up_proj(x)
        gate, up = fused_out.chunk(2, dim=-1)

        # SwiGLU activation (float32 to prevent overflow in down_proj)
        hidden = F.silu(gate).float() * up.float()

        output = self.down_proj(hidden)
        output = self.dropout(output)
        return output


class StochasticFFN(nn.Module):
    """SwiGLU Feed-Forward Network with Stochastic Bit-Flip training.

    No latent FP32 weights — all projections are packed 2-bit ternary.
    Uses separate gate_proj, up_proj, down_proj (not fused like TernaryFFN).

    Args:
        hidden_dim: model dimension (input/output)
        ffn_dim: feed-forward hidden dimension
        dropout: dropout rate on output
        scale: ternary weight scale factor
        threshold: bit-flip threshold (None = auto-compute)
        int8: use INT8 matmul kernel
        per_channel: per-output-channel alpha scaling
        group_size: per-group alpha block size (0=disabled)
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        scale: float = 1.0,
        threshold: Optional[float] = None,
        int8: bool = False,
        per_channel: bool = False,
        group_size: int = 0,
    ):
        super().__init__()
        from .layers import StochasticTernaryLinear

        self.gate_proj = StochasticTernaryLinear(
            hidden_dim, ffn_dim, scale=scale, threshold=threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.up_proj = StochasticTernaryLinear(
            hidden_dim, ffn_dim, scale=scale, threshold=threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.down_proj = StochasticTernaryLinear(
            ffn_dim, hidden_dim, scale=scale, threshold=threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor (..., hidden_dim)

        Returns:
            Output tensor (..., hidden_dim)
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate).float() * up.float()
        output = self.down_proj(hidden)
        return self.dropout(output)

    @torch.no_grad()
    def set_thresholds(self, threshold: float) -> None:
        """Set bit-flip threshold for all projections.

        Args:
            threshold: new threshold value
        """
        self.gate_proj.set_threshold(threshold)
        self.up_proj.set_threshold(threshold)
        self.down_proj.set_threshold(threshold)

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        """Apply bit flips based on accumulator thresholds."""
        self.gate_proj.apply_bit_flips()
        self.up_proj.apply_bit_flips()
        self.down_proj.apply_bit_flips()
