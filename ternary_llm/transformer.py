"""Ternary transformer decoder models.

Implements three variants:
- TernaryTransformerModel: STE-trained ternary weights (absmean quantization)
- StochasticTransformerModel: Stochastic bit-flip training (packed 2-bit)
- StochasticMLAModel: Stochastic bit-flip with Multi-head Latent Attention

All models follow pre-norm architecture (RMSNorm before attention/FFN).
"""
import time
from typing import Optional

__all__ = [
    "TernaryTransformerBlock", "TernaryTransformerModel",
    "StochasticTransformerBlock", "StochasticTransformerModel",
    "StochasticMLABlock", "StochasticMLAModel",
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RMSNorm, TopKActivation
from .attention import TernaryMultiHeadAttention
from .ffn import TernaryFFN
from .mla import StochasticMLAAttention


# ──────────────────────────────────────────────────────────────
# STE-trained ternary models
# ──────────────────────────────────────────────────────────────


class TernaryTransformerBlock(nn.Module):
    """Single transformer decoder block with ternary weights (STE).

    Architecture (Pre-Norm, following BitNet b1.58):
        x -> RMSNorm -> TopK -> MultiHeadAttention -> Residual Add
          -> RMSNorm -> TopK -> FFN -> Residual Add

    Args:
        hidden_dim: model dimension
        num_heads: number of attention heads
        ffn_dim: feed-forward hidden dimension
        dropout: dropout rate
        ternary_scale: scale factor for ternary quantization
        per_channel: use per-channel alpha scaling
        topk: top-k activation sparsity ratio (1.0 = no sparsity)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        ternary_scale: float = 0.7,
        per_channel: bool = False,
        topk: float = 1.0,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(hidden_dim)
        self.attn_topk = TopKActivation(topk)
        self.attn = TernaryMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            ternary_scale=ternary_scale,
            per_channel=per_channel,
        )

        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn_topk = TopKActivation(topk)
        self.ffn = TernaryFFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            ternary_scale=ternary_scale,
            per_channel=per_channel,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with residual connections.

        Args:
            x: input tensor (batch, seq_len, hidden_dim)
            mask: optional attention mask
            past_kv: optional (K, V) cache tuple for autoregressive generation

        Returns:
            Tuple of (output tensor, (K, V) cache)
        """
        # Attention block
        residual = x
        x = self.attn_norm(x)
        x = self.attn_topk(x)
        past_k, past_v = past_kv if past_kv is not None else (None, None)
        attn_out, k, v = self.attn(x, mask=mask, past_k=past_k, past_v=past_v)
        x = attn_out + residual

        # FFN block
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn_topk(x)
        x = self.ffn(x)
        x = x + residual

        return x, (k, v)


class TernaryTransformerModel(nn.Module):
    """Full ternary transformer decoder with STE training.

    Stack of TernaryTransformerBlocks with token embedding and LM head.
    Weights are ternary {-1, 0, +1} via absmean quantization with STE.

    Args:
        vocab_size: vocabulary size
        hidden_dim: model dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        ffn_dim: feed-forward hidden dimension
        max_seq_len: maximum sequence length
        dropout: dropout rate
        ternary_scale: scale factor for ternary quantization
        per_channel: per-channel alpha scaling
        topk: top-k activation sparsity ratio
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        ternary_scale: float = 0.7,
        per_channel: bool = False,
        topk: float = 1.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        self.layers = nn.ModuleList([
            TernaryTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                ternary_scale=ternary_scale,
                per_channel=per_channel,
                topk=topk,
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)

    def _apply(self, fn):
        """Ensure LM head stays tied to embedding after device moves."""
        super()._apply(fn)
        self.lm_head.weight = self.token_embedding.weight
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        activation_dtype: Optional[torch.dtype] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token ids
            targets: (batch_size, seq_len) target token ids for loss computation
            activation_dtype: cast activations to this dtype (e.g. float16 for AMP)
            past_key_values: list of (K, V) tuples per layer for KV cache

        Returns:
            Tuple of:
                - logits: (batch_size, seq_len, vocab_size)
                - loss: scalar loss if targets provided, else None
                - new_key_values: list of (K, V) tuples per layer (None during training)
        """
        batch_size, seq_len = input_ids.shape

        if past_key_values is None:
            pos_offset = 0
        else:
            pos_offset = past_key_values[0][0].size(-2)
        positions = torch.arange(
            pos_offset, pos_offset + seq_len, device=input_ids.device
        ).unsqueeze(0)
        input_ids = input_ids.clamp(0, self.token_embedding.num_embeddings - 1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        if activation_dtype is not None:
            x = x.to(activation_dtype)
        x = self.dropout(x)

        new_key_values = []
        self._layer_times = []
        for i, layer in enumerate(self.layers):
            t0 = time.perf_counter()
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv = layer(x, past_kv=past_kv)
            self._layer_times.append(time.perf_counter() - t0)
            new_key_values.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x).float()

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss, new_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively with KV cache.

        Args:
            input_ids: (batch_size, seq_len) initial token ids
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling (None = disabled)

        Returns:
            Generated token ids including input, shape (batch_size, total_len)
        """
        past_key_values = None
        for step in range(max_new_tokens):
            if step == 0:
                idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                logits, _, past_key_values = self(idx_cond, past_key_values=None)
            else:
                last_token = input_ids[:, -1:]
                logits, _, past_key_values = self(last_token, past_key_values=past_key_values)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = idx_next.clamp(0, self.token_embedding.num_embeddings - 1)

            input_ids = torch.cat([input_ids, idx_next], dim=1)

        return input_ids


# ──────────────────────────────────────────────────────────────
# Stochastic bit-flip models (packed 2-bit, no latent FP32)
# ──────────────────────────────────────────────────────────────


class StochasticTransformerBlock(nn.Module):
    """Transformer block with Stochastic Bit-Flip layers.

    Same pre-norm architecture as TernaryTransformerBlock but uses
    StochasticTernaryLinear (packed 2-bit weights, accumulator-based
    gradient, no latent FP32 weights).

    Args:
        hidden_dim: model dimension
        num_heads: number of attention heads
        ffn_dim: feed-forward hidden dimension
        dropout: dropout rate
        scale: ternary weight scale factor
        threshold: bit-flip threshold (None = auto-compute)
        int8: use INT8 matmul kernel
        topk: top-k activation sparsity ratio
        per_channel: per-channel alpha scaling
        group_size: per-group alpha block size (0=disabled)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        scale: float = 1.0,
        threshold: Optional[float] = None,
        int8: bool = False,
        topk: float = 1.0,
        per_channel: bool = False,
        group_size: int = 0,
    ):
        super().__init__()
        from .attention import StochasticMultiHeadAttention
        from .ffn import StochasticFFN

        self.attn_norm = RMSNorm(hidden_dim)
        self.attn_topk = TopKActivation(topk)
        self.attn = StochasticMultiHeadAttention(
            hidden_dim, num_heads, dropout, scale, threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn_topk = TopKActivation(topk)
        self.ffn = StochasticFFN(
            hidden_dim, ffn_dim, dropout, scale, threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with residual connections.

        Args:
            x: input tensor (batch, seq_len, hidden_dim)
            mask: optional attention mask
            past_kv: optional (K, V) cache for autoregressive generation

        Returns:
            Tuple of (output tensor, (K, V) cache)
        """
        r = x
        x = self.attn_norm(x)
        x = self.attn_topk(x)
        past_k, past_v = past_kv if past_kv is not None else (None, None)
        x, k, v = self.attn(x, mask=mask, past_k=past_k, past_v=past_v)
        x = x + r
        r = x
        x = self.ffn_norm(x)
        x = self.ffn_topk(x)
        x = self.ffn(x)
        x = x + r
        return x, (k, v)

    @torch.no_grad()
    def set_thresholds(self, threshold: float) -> None:
        """Set bit-flip threshold for all layers."""
        self.attn.set_thresholds(threshold)
        self.ffn.set_thresholds(threshold)

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        """Apply bit flips based on accumulator thresholds."""
        self.attn.apply_bit_flips()
        self.ffn.apply_bit_flips()


class StochasticTransformerModel(nn.Module):
    """Full transformer with Stochastic Bit-Flip training.

    Packed 2-bit ternary weights with accumulator-based gradient.
    No optimizer for ternary weights — gradient auto-accumulates and
    flips when threshold exceeded.

    Args:
        vocab_size: vocabulary size
        hidden_dim: model dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        ffn_dim: feed-forward hidden dimension
        max_seq_len: maximum sequence length
        dropout: dropout rate
        scale: ternary weight scale factor
        threshold: bit-flip threshold (None = auto-compute)
        int8: use INT8 matmul kernel
        topk: top-k activation sparsity ratio
        per_channel: per-channel alpha scaling
        group_size: per-group alpha block size (0=disabled)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        scale: float = 1.0,
        threshold: Optional[float] = None,
        int8: bool = False,
        topk: float = 1.0,
        per_channel: bool = False,
        group_size: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.layers = nn.ModuleList([
            StochasticTransformerBlock(
                hidden_dim, num_heads, ffn_dim, dropout, scale, threshold,
                int8=int8, topk=topk, per_channel=per_channel, group_size=group_size,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)

    def _apply(self, fn):
        """Ensure LM head stays tied to embedding after device moves."""
        super()._apply(fn)
        self.lm_head.weight = self.token_embedding.weight
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        activation_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token ids
            targets: (batch_size, seq_len) target token ids for loss
            past_key_values: list of (K, V) tuples per layer for KV cache
            activation_dtype: cast activations to this dtype

        Returns:
            Tuple of (logits, loss, new_key_values)
        """
        B, T = input_ids.shape
        if past_key_values is None:
            pos_offset = 0
        else:
            pos_offset = past_key_values[0][0].size(-2)
        positions = torch.arange(pos_offset, pos_offset + T, device=input_ids.device).unsqueeze(0)
        input_ids = input_ids.clamp(0, self.token_embedding.num_embeddings - 1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        if activation_dtype is not None:
            x = x.to(activation_dtype)
        x = self.dropout(x)

        new_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv = layer(x, past_kv=past_kv)
            new_key_values.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x).float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1,
            )
        return logits, loss, new_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively with KV cache."""
        past_key_values = None
        for step in range(max_new_tokens):
            if step == 0:
                idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                logits, _, past_key_values = self(idx_cond, past_key_values=None)
            else:
                last_token = input_ids[:, -1:]
                logits, _, past_key_values = self(last_token, past_key_values=past_key_values)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).clamp(0, self.token_embedding.num_embeddings - 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    @torch.no_grad()
    def set_thresholds(self, threshold: float) -> None:
        """Set bit-flip threshold for all layers."""
        for layer in self.layers:
            layer.set_thresholds(threshold)

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        """Apply bit flips for all layers."""
        for layer in self.layers:
            layer.apply_bit_flips()


# ──────────────────────────────────────────────────────────────
# MLA models (stochastic bit-flip + Multi-head Latent Attention)
# ──────────────────────────────────────────────────────────────


class StochasticMLABlock(nn.Module):
    """Transformer block with MLA and Stochastic Bit-Flip layers.

    Uses StochasticMLAAttention for KV-compressed attention and
    StochasticFFN for the feed-forward network.

    Args:
        hidden_dim: model dimension
        num_heads: number of attention heads
        ffn_dim: feed-forward hidden dimension
        dropout: dropout rate
        scale: ternary weight scale factor
        threshold: bit-flip threshold (None = auto-compute)
        int8: use INT8 matmul kernel
        topk: top-k activation sparsity ratio
        per_channel: per-channel alpha scaling
        group_size: per-group alpha block size (0=disabled)
        kv_latent_dim: KV compression dimension (None = 2 * head_dim)
        rope_per_head: RoPE dimension per head (None = auto)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        scale: float = 1.0,
        threshold: Optional[float] = None,
        int8: bool = False,
        topk: float = 1.0,
        per_channel: bool = False,
        group_size: int = 0,
        kv_latent_dim: Optional[int] = None,
        rope_per_head: Optional[int] = None,
    ):
        super().__init__()
        from .ffn import StochasticFFN

        self.attn_norm = RMSNorm(hidden_dim)
        self.attn_topk = TopKActivation(topk)
        self.attn = StochasticMLAAttention(
            hidden_dim, num_heads, dropout, scale, threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
            kv_latent_dim=kv_latent_dim, rope_per_head=rope_per_head,
        )
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn_topk = TopKActivation(topk)
        self.ffn = StochasticFFN(
            hidden_dim, ffn_dim, dropout, scale, threshold,
            int8=int8, per_channel=per_channel, group_size=group_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with MLA attention.

        Args:
            x: input tensor (batch, seq_len, hidden_dim)
            mask: optional attention mask
            past_kv: optional (kv_latent, k_rope) cache for MLA

        Returns:
            Tuple of (output tensor, (kv_latent, k_rope) cache)
        """
        r = x
        x = self.attn_norm(x)
        x = self.attn_topk(x)
        x, kv_latent, k_rope = self.attn(x, mask=mask, past_kv=past_kv)
        x = x + r
        r = x
        x = self.ffn_norm(x)
        x = self.ffn_topk(x)
        x = self.ffn(x)
        x = x + r
        return x, (kv_latent, k_rope)

    @torch.no_grad()
    def set_thresholds(self, threshold: float) -> None:
        """Set bit-flip threshold for all layers."""
        self.attn.set_thresholds(threshold)
        self.ffn.set_thresholds(threshold)

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        """Apply bit flips for all layers."""
        self.attn.apply_bit_flips()
        self.ffn.apply_bit_flips()


class StochasticMLAModel(nn.Module):
    """Full transformer with MLA and Stochastic Bit-Flip training.

    Multi-head Latent Attention (DeepSeek-V2 style) compresses K,V into
    a small latent vector for efficient KV cache. All projections are
    ternary via stochastic bit-flip training.

    Args:
        vocab_size: vocabulary size
        hidden_dim: model dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        ffn_dim: feed-forward hidden dimension
        max_seq_len: maximum sequence length
        dropout: dropout rate
        scale: ternary weight scale factor
        threshold: bit-flip threshold (None = auto-compute)
        int8: use INT8 matmul kernel
        topk: top-k activation sparsity ratio
        per_channel: per-channel alpha scaling
        group_size: per-group alpha block size (0=disabled)
        kv_latent_dim: KV compression dimension (None = 2 * head_dim)
        rope_per_head: RoPE dimension per head (None = auto)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        scale: float = 1.0,
        threshold: Optional[float] = None,
        int8: bool = False,
        topk: float = 1.0,
        per_channel: bool = False,
        group_size: int = 0,
        kv_latent_dim: Optional[int] = None,
        rope_per_head: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            StochasticMLABlock(
                hidden_dim, num_heads, ffn_dim, dropout, scale, threshold,
                int8=int8, topk=topk, per_channel=per_channel, group_size=group_size,
                kv_latent_dim=kv_latent_dim, rope_per_head=rope_per_head,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)

    def _apply(self, fn):
        """Ensure LM head stays tied to embedding after device moves."""
        super()._apply(fn)
        self.lm_head.weight = self.token_embedding.weight
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        activation_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token ids
            targets: (batch_size, seq_len) target token ids for loss
            past_key_values: list of (kv_latent, k_rope) tuples per layer
            activation_dtype: cast activations to this dtype

        Returns:
            Tuple of (logits, loss, new_key_values)
        """
        B, T = input_ids.shape
        if past_key_values is None:
            pos_offset = 0
        else:
            pos_offset = past_key_values[0][0].size(-2)
        positions = torch.arange(pos_offset, pos_offset + T, device=input_ids.device).unsqueeze(0)
        input_ids = input_ids.clamp(0, self.token_embedding.num_embeddings - 1)
        x = self.token_embedding(input_ids)
        if activation_dtype is not None:
            x = x.to(activation_dtype)
        x = self.dropout(x)

        new_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv = layer(x, past_kv=past_kv)
            new_key_values.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x).float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1,
            )
        return logits, loss, new_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively with MLA KV cache."""
        past_key_values = None
        for step in range(max_new_tokens):
            if step == 0:
                idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                logits, _, past_key_values = self(idx_cond, past_key_values=None)
            else:
                last_token = input_ids[:, -1:]
                logits, _, past_key_values = self(last_token, past_key_values=past_key_values)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).clamp(0, self.token_embedding.num_embeddings - 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    @torch.no_grad()
    def set_thresholds(self, threshold: float) -> None:
        """Set bit-flip threshold for all layers."""
        for layer in self.layers:
            layer.set_thresholds(threshold)

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        """Apply bit flips for all layers."""
        for layer in self.layers:
            layer.apply_bit_flips()
