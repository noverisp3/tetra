import torch
__all__ = ["StochasticMLAAttention", "precompute_freqs_cis", "apply_rotary_emb"]

import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0, device: str = "cpu"):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    x_complex = torch.view_as_complex(x_float.reshape(*x_float.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[None, None, : x.shape[-2], :]
    x_rotated = torch.view_as_real(x_complex * freqs_cis).reshape(*x.shape)
    return x_rotated.to(x.dtype)


class StochasticMLAAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0, scale=1.0, threshold=None,
                 int8=False, per_channel=False, group_size=0, kv_latent_dim=None, rope_per_head=None):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.rope_per_head = rope_per_head or max(4, self.head_dim // 4)
        self.rope_dim = self.rope_per_head * num_heads
        self.kv_latent_dim = kv_latent_dim or (self.head_dim * 2)
        self.eff_head_dim = self.head_dim + self.rope_per_head
        self.scale_factor = self.eff_head_dim ** -0.5

        from .layers import StochasticTernaryLinear
        self.q_proj = StochasticTernaryLinear(hidden_dim, hidden_dim, scale=scale, threshold=threshold,
                                              int8=int8, per_channel=per_channel, group_size=group_size)
        self.kv_down_proj = StochasticTernaryLinear(hidden_dim, self.kv_latent_dim, scale=scale, threshold=threshold,
                                                    int8=int8, per_channel=per_channel, group_size=group_size)
        self.k_up_proj = StochasticTernaryLinear(self.kv_latent_dim, hidden_dim, scale=scale, threshold=threshold,
                                                 int8=int8, per_channel=per_channel, group_size=group_size)
        self.v_up_proj = StochasticTernaryLinear(self.kv_latent_dim, hidden_dim, scale=scale, threshold=threshold,
                                                 int8=int8, per_channel=per_channel, group_size=group_size)
        self.q_rope_proj = StochasticTernaryLinear(hidden_dim, self.rope_dim, scale=scale, threshold=threshold,
                                                   int8=int8, per_channel=per_channel, group_size=group_size)
        self.k_rope_proj = StochasticTernaryLinear(hidden_dim, self.rope_dim, scale=scale, threshold=threshold,
                                                   int8=int8, per_channel=per_channel, group_size=group_size)
        self.o_proj = StochasticTernaryLinear(hidden_dim, hidden_dim, scale=scale, threshold=threshold,
                                              int8=int8, per_channel=per_channel, group_size=group_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer("freqs_cis", None, persistent=False)

    def _get_freqs(self, seq_len, device):
        if self.freqs_cis is None or self.freqs_cis.size(-2) < seq_len:
            self.freqs_cis = precompute_freqs_cis(self.rope_per_head, max(seq_len * 2, 512), device=device)
        return self.freqs_cis[:seq_len, :].to(device)

    def forward(self, x, mask=None, past_kv=None):
        B, T, C = x.shape
        device = x.device
        q = self.q_proj(x)
        kv_latent = self.kv_down_proj(x)
        q_rope = self.q_rope_proj(x)
        k_rope = self.k_rope_proj(x)
        freqs_cis = self._get_freqs(T, device)
        q_rope = apply_rotary_emb(
            q_rope.view(B, T, self.num_heads, self.rope_per_head).transpose(1, 2), freqs_cis
        )
        k_rope = apply_rotary_emb(
            k_rope.view(B, T, self.num_heads, self.rope_per_head).transpose(1, 2), freqs_cis
        )
        if past_kv is not None:
            past_latent, past_k_rope = past_kv
            kv_latent = torch.cat([past_latent, kv_latent], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)
        T_full = kv_latent.size(1)
        k = self.k_up_proj(kv_latent)
        v = self.v_up_proj(kv_latent)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)
        q = torch.cat([q, q_rope], dim=-1)
        k = torch.cat([k, k_rope], dim=-1)
        dp = self.attn_dropout.p if self.training else 0.0
        is_causal = mask is None and past_kv is None
        out = F.scaled_dot_product_attention(
            q.float() * self.scale_factor, k.float(), v.float(),
            dropout_p=dp, is_causal=is_causal,
        ).to(x.dtype)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), kv_latent, k_rope

    @torch.no_grad()
    def set_thresholds(self, threshold):
        for m in [self.q_proj, self.kv_down_proj, self.k_up_proj, self.v_up_proj,
                  self.q_rope_proj, self.k_rope_proj, self.o_proj]:
            m.set_threshold(threshold)

    @torch.no_grad()
    def apply_bit_flips(self):
        for m in [self.q_proj, self.kv_down_proj, self.k_up_proj, self.v_up_proj,
                  self.q_rope_proj, self.k_rope_proj, self.o_proj]:
            m.apply_bit_flips()
