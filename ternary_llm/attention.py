import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import TernaryLinear, RMSNorm


class TernaryMultiHeadAttention(nn.Module):
    """Multi-Head Attention with ternary projections.

    Architecture (following BitNet b1.58):
        - Q, K, V projections: ternary weights
        - Attention scores: INT8 dot product
        - Softmax: INT8 quantized
        - Output projection: ternary weights

    Args:
        hidden_dim: model dimension
        num_heads: number of attention heads
        dropout: dropout rate (applied to attention weights)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Ternary projections
        self.q_proj = TernaryLinear(hidden_dim, hidden_dim)
        self.k_proj = TernaryLinear(hidden_dim, hidden_dim)
        self.v_proj = TernaryLinear(hidden_dim, hidden_dim)
        self.o_proj = TernaryLinear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V (ternary weights applied internally)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: scaled dot product
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided (causal mask for decoder)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        # Softmax (in practice, could use INT8 approximation)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        output = self.o_proj(attn_output)

        return output
