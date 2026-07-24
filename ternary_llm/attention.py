import torch
__all__ = [
    "TernaryMultiHeadAttention", "StochasticMultiHeadAttention",
]

import torch.nn as nn
import torch.nn.functional as F
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
        ternary_scale: float = 0.7,
        per_channel: bool = False,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Ternary projections
        self.q_proj = TernaryLinear(hidden_dim, hidden_dim, ternary_scale=ternary_scale, per_channel=per_channel)
        self.k_proj = TernaryLinear(hidden_dim, hidden_dim, ternary_scale=ternary_scale, per_channel=per_channel)
        self.v_proj = TernaryLinear(hidden_dim, hidden_dim, ternary_scale=ternary_scale, per_channel=per_channel)
        self.o_proj = TernaryLinear(hidden_dim, hidden_dim, ternary_scale=ternary_scale, per_channel=per_channel)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V (ternary weights applied internally)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # KV cache
        if past_k is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # Fused attention (float32 to prevent softmax overflow in float16)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        is_causal = (mask is None) and (past_k is None)
        attn_output = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(),
            dropout_p=dropout_p,
            is_causal=is_causal,
        ).to(x.dtype)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.hidden_dim)
        )
        output = self.o_proj(attn_output)

        return output, k, v


class StochasticMultiHeadAttention(nn.Module):
    """Multi-Head Attention with Stochastic Bit-Flip."""

    def __init__(self, hidden_dim, num_heads, dropout=0.0, scale=1.0, threshold=None, int8=False, per_channel=False):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        from .layers import StochasticTernaryLinear
        self.q_proj = StochasticTernaryLinear(hidden_dim, hidden_dim, scale=scale, threshold=threshold, int8=int8, per_channel=per_channel)
        self.k_proj = StochasticTernaryLinear(hidden_dim, hidden_dim, scale=scale, threshold=threshold, int8=int8, per_channel=per_channel)
        self.v_proj = StochasticTernaryLinear(hidden_dim, hidden_dim, scale=scale, threshold=threshold, int8=int8, per_channel=per_channel)
        self.o_proj = StochasticTernaryLinear(hidden_dim, hidden_dim, scale=scale, threshold=threshold, int8=int8, per_channel=per_channel)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                past_k: torch.Tensor | None = None, past_v: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if past_k is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        dp = self.attn_dropout.p if self.training else 0.0
        is_causal = (mask is None) and (past_k is None)
        out = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(),
            dropout_p=dp, is_causal=is_causal
        ).to(x.dtype)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), k, v

    @torch.no_grad()
    def set_thresholds(self, threshold: float) -> None:
        self.q_proj.set_threshold(threshold)
        self.k_proj.set_threshold(threshold)
        self.v_proj.set_threshold(threshold)
        self.o_proj.set_threshold(threshold)

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        self.q_proj.apply_bit_flips()
        self.k_proj.apply_bit_flips()
        self.v_proj.apply_bit_flips()
        self.o_proj.apply_bit_flips()
