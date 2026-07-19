import torch
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
    ):
        super().__init__()

        # Ternary projections
        self.gate_proj = TernaryLinear(hidden_dim, ffn_dim)
        self.up_proj = TernaryLinear(hidden_dim, ffn_dim)
        self.down_proj = TernaryLinear(ffn_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        # Element-wise multiply
        hidden = gate * up

        # Down projection
        output = self.down_proj(hidden)
        output = self.dropout(output)

        return output
