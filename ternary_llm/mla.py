"""Multi-Head Latent Attention (MLA) with Stochastic Bit-Flip training.

DeepSeek-V2 style MLA: compress K,V into a small latent vector for efficient
KV cache. Uses decoupled RoPE (separate Q/K rope projections) and stochastic
ternary projections.

Reference: https://arxiv.org/abs/2405.04434
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = ["StochasticMLAAttention", "precompute_freqs_cis", "apply_rotary_emb"]


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Precompute complex-valued RoPE frequency tensor.

    Args:
        dim: number of frequency pairs (rope_per_head, NOT the full rope_dim).
        max_seq_len: maximum sequence length to precompute.
        base: RoPE base frequency (default: 10000.0).
        device: target device for the tensor.

    Returns:
        Complex tensor of shape (max_seq_len, dim) for rotary embedding.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embedding to input tensor.

    Args:
        x: input tensor of shape (..., seq_len, rope_per_head).
        freqs_cis: precomputed complex frequencies of shape (seq_len, rope_per_head/2).

    Returns:
        Rotated tensor of same shape as input.
    """
    x_float = x.float()
    x_complex = torch.view_as_complex(x_float.reshape(*x_float.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[None, None, : x.shape[-2], :]
    x_rotated = torch.view_as_real(x_complex * freqs_cis).reshape(*x.shape)
    return x_rotated.to(x.dtype)


class StochasticMLAAttention(nn.Module):
    """Multi-Head Latent Attention with Stochastic Bit-Flip training.

    MLA compresses K,V into a small latent vector (kv_latent_dim) for
    efficient KV cache. During training, all projections are ternary via
    stochastic bit-flip.

    Architecture:
        Q  = x @ W_q       (full head_dim output)
        Q_rope = x @ W_qr  (rope_per_head output, RoPE applied)
        KV_latent = x @ W_kvd  (kv_latent_dim output, compressed)
        K = KV_latent @ W_ku    (head_dim output, decompressed)
        V = KV_latent @ W_vu    (head_dim output, decompressed)
        K_rope = x @ W_kr  (rope_per_head output, RoPE applied)
        O = Attn(Q, [K;K_rope], [V;V_rope]) @ W_o

    Effective head_dim = head_dim + rope_per_head (Q/K have rope appended).

    Args:
        hidden_dim: model dimension
        num_heads: number of attention heads
        dropout: attention dropout rate
        scale: ternary weight scale factor
        threshold: bit-flip threshold (None = auto-compute)
        int8: use INT8 matmul kernel (requires C++ extension)
        per_channel: per-output-channel alphas
        group_size: per-group alpha block size (0=disabled)
        kv_latent_dim: KV compression dimension (None = 2 * head_dim)
        rope_per_head: RoPE dimension per head (None = max(4, head_dim // 4))
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        scale: float = 1.0,
        threshold: Optional[float] = None,
        int8: bool = False,
        per_channel: bool = False,
        group_size: int = 0,
        kv_latent_dim: Optional[int] = None,
        rope_per_head: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.rope_per_head = rope_per_head or max(4, self.head_dim // 4)
        self.rope_dim = self.rope_per_head * num_heads
        self.kv_latent_dim = kv_latent_dim or (self.head_dim * 2)
        self.eff_head_dim = self.head_dim + self.rope_per_head
        self.scale_factor = self.eff_head_dim ** -0.5

        from .layers import StochasticTernaryLinear

        # 7 ternary projections per layer (vs 4 in standard attention)
        self.q_proj = StochasticTernaryLinear(
            hidden_dim, hidden_dim, scale=scale, threshold=threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.kv_down_proj = StochasticTernaryLinear(
            hidden_dim, self.kv_latent_dim, scale=scale, threshold=threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.k_up_proj = StochasticTernaryLinear(
            self.kv_latent_dim, hidden_dim, scale=scale, threshold=threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.v_up_proj = StochasticTernaryLinear(
            self.kv_latent_dim, hidden_dim, scale=scale, threshold=threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.q_rope_proj = StochasticTernaryLinear(
            hidden_dim, self.rope_dim, scale=scale, threshold=threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.k_rope_proj = StochasticTernaryLinear(
            hidden_dim, self.rope_dim, scale=scale, threshold=threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.o_proj = StochasticTernaryLinear(
            hidden_dim, hidden_dim, scale=scale, threshold=threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer("freqs_cis", None, persistent=False)

    def _get_freqs(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or lazily compute RoPE frequencies for the given sequence length.

        Args:
            seq_len: required sequence length.
            device: target device.

        Returns:
            Complex frequency tensor of shape (seq_len, rope_per_head).
        """
        if self.freqs_cis is None or self.freqs_cis.size(-2) < seq_len:
            self.freqs_cis = precompute_freqs_cis(
                self.rope_per_head, max(seq_len * 2, 512), device=device
            )
        return self.freqs_cis[:seq_len, :].to(device)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with optional KV cache.

        Args:
            x: input tensor of shape (batch, seq_len, hidden_dim).
            mask: optional attention mask (not typically used with MLA).
            past_kv: tuple of (past_kv_latent, past_k_rope) for autoregressive
                     generation. If None, starts fresh.

        Returns:
            Tuple of:
                - output: (batch, seq_len, hidden_dim)
                - kv_latent: (batch, full_seq_len, kv_latent_dim) for cache
                - k_rope: (batch, num_heads, full_seq_len, rope_per_head) for cache
        """
        B, T, C = x.shape
        device = x.device

        # Projections
        q = self.q_proj(x)
        kv_latent = self.kv_down_proj(x)
        q_rope = self.q_rope_proj(x)
        k_rope = self.k_rope_proj(x)

        # Apply RoPE to Q_rope and K_rope
        freqs_cis = self._get_freqs(T, device)
        q_rope = apply_rotary_emb(
            q_rope.view(B, T, self.num_heads, self.rope_per_head).transpose(1, 2),
            freqs_cis,
        )
        k_rope = apply_rotary_emb(
            k_rope.view(B, T, self.num_heads, self.rope_per_head).transpose(1, 2),
            freqs_cis,
        )

        # KV cache
        if past_kv is not None:
            past_latent, past_k_rope = past_kv
            kv_latent = torch.cat([past_latent, kv_latent], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)
        T_full = kv_latent.size(1)

        # Decompress K, V from latent
        k = self.k_up_proj(kv_latent)
        v = self.v_up_proj(kv_latent)

        # Reshape to (batch, heads, seq, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)

        # Concatenate rope components to Q/K
        q = torch.cat([q, q_rope], dim=-1)
        k = torch.cat([k, k_rope], dim=-1)

        # Scaled dot-product attention
        dp = self.attn_dropout.p if self.training else 0.0
        is_causal = mask is None and past_kv is None
        out = F.scaled_dot_product_attention(
            q.float() * self.scale_factor, k.float(), v.float(),
            dropout_p=dp, is_causal=is_causal,
        ).to(x.dtype)

        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), kv_latent, k_rope

    @torch.no_grad()
    def set_thresholds(self, threshold: float) -> None:
        """Set bit-flip threshold for all projections.

        Args:
            threshold: new threshold value.
        """
        for m in [
            self.q_proj, self.kv_down_proj, self.k_up_proj, self.v_up_proj,
            self.q_rope_proj, self.k_rope_proj, self.o_proj,
        ]:
            m.set_threshold(threshold)

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        """Apply bit flips to all projections based on accumulator thresholds."""
        for m in [
            self.q_proj, self.kv_down_proj, self.k_up_proj, self.v_up_proj,
            self.q_rope_proj, self.k_rope_proj, self.o_proj,
        ]:
            m.apply_bit_flips()
