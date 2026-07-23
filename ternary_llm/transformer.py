import time
__all__ = [
    "TernaryTransformerBlock", "TernaryTransformerModel",
    "StochasticTransformerBlock", "StochasticTransformerModel",
]

import torch
import torch.nn as nn
from .layers import RMSNorm, TopKActivation
from .attention import TernaryMultiHeadAttention
from .ffn import TernaryFFN


class TernaryTransformerBlock(nn.Module):
    """Single transformer decoder block with ternary weights.

    Architecture (Pre-Norm, following BitNet b1.58):
        x -> RMSNorm -> MultiHeadAttention -> Residual Add
          -> RMSNorm -> FFN -> Residual Add

    Args:
        hidden_dim: model dimension
        num_heads: number of attention heads
        ffn_dim: feed-forward hidden dimension
        dropout: dropout rate
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

        # Attention block with pre-norm
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn_topk = TopKActivation(topk)
        self.attn = TernaryMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            ternary_scale=ternary_scale,
            per_channel=per_channel,
        )

        # FFN block with pre-norm
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
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Attention block with residual connection
        residual = x
        x = self.attn_norm(x)
        x = self.attn_topk(x)
        past_k, past_v = past_kv if past_kv is not None else (None, None)
        attn_out, k, v = self.attn(x, mask=mask, past_k=past_k, past_v=past_v)
        x = attn_out + residual

        # FFN block with residual connection
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn_topk(x)
        x = self.ffn(x)
        x = x + residual

        return x, (k, v)


class TernaryTransformerModel(nn.Module):
    """Tetra: Full Ternary Transformer Decoder Model.

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

        # Token embedding (not ternary, following BitNet)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding (learned)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer blocks
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

        # Final normalization
        self.norm = RMSNorm(hidden_dim)

        # LM head (ternary)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying (embedding and LM head share weights)
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)

    def _apply(self, fn):
        super()._apply(fn)
        self.lm_head.weight = self.token_embedding.weight
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        activation_dtype: torch.dtype | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Args:
            input_ids: (batch_size, seq_len) token ids
            targets: (batch_size, seq_len) target token ids for loss
            activation_dtype: cast activations to this dtype (e.g. float16 for AMP)
            past_key_values: list of (K, V) tuples per layer (for generation KV cache)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar loss if targets provided, else None
            new_key_values: list of (K, V) tuples per layer (None during training)
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        if past_key_values is None:
            pos_offset = 0
        else:
            pos_offset = past_key_values[0][0].size(-2)
        positions = torch.arange(pos_offset, pos_offset + seq_len, device=input_ids.device).unsqueeze(0)
        input_ids = input_ids.clamp(0, self.token_embedding.num_embeddings - 1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        if activation_dtype is not None:
            x = x.to(activation_dtype)
        x = self.dropout(x)

        # Transformer layers
        new_key_values = []
        self._layer_times = []
        for i, layer in enumerate(self.layers):
            t0 = time.perf_counter()
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, kv = layer(x, past_kv=past_kv)
            self._layer_times.append(time.perf_counter() - t0)
            new_key_values.append(kv)

        # Final norm
        x = self.norm(x)

        # LM head
        logits = self.lm_head(x).float()

        # Compute loss if targets provided
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
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate text autoregressively with KV cache."""
        past_key_values = None
        for step in range(max_new_tokens):
            if step == 0:
                # Full prompt forward to build initial cache
                idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                logits, _, past_key_values = self(idx_cond, past_key_values=None)
            else:
                # Single token forward with KV cache
                last_token = input_ids[:, -1:]
                logits, _, past_key_values = self(last_token, past_key_values=past_key_values)

            # Get last token logits
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Clamp to vocab size
            idx_next = idx_next.clamp(0, self.token_embedding.num_embeddings - 1)

            # Append
            input_ids = torch.cat([input_ids, idx_next], dim=1)

        return input_ids


class StochasticTransformerBlock(nn.Module):
    """Transformer block with Stochastic Bit-Flip layers."""

    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.0, scale=1.0, threshold=None, int8=False, topk=1.0):
        super().__init__()
        from .attention import StochasticMultiHeadAttention
        from .ffn import StochasticFFN
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn_topk = TopKActivation(topk)
        self.attn = StochasticMultiHeadAttention(hidden_dim, num_heads, dropout, scale, threshold, int8=int8)
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn_topk = TopKActivation(topk)
        self.ffn = StochasticFFN(hidden_dim, ffn_dim, dropout, scale, threshold, int8=int8)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        r = x
        x = self.attn_norm(x)
        x = self.attn_topk(x)
        x = self.attn(x, mask=mask)
        x = x + r
        r = x
        x = self.ffn_norm(x)
        x = self.ffn_topk(x)
        x = self.ffn(x)
        x = x + r
        return x

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        self.attn.apply_bit_flips()
        self.ffn.apply_bit_flips()


class StochasticTransformerModel(nn.Module):
    """Full transformer with Stochastic Bit-Flip (packed 2-bit weights, accumulator flip).

    No optimizer for ternary weights — gradient auto-accumulates into
    accumulator and flips when threshold exceeded.
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ffn_dim,
                 max_seq_len=2048, dropout=0.0, scale=1.0, threshold=None, int8=False, topk=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.layers = nn.ModuleList([
            StochasticTransformerBlock(hidden_dim, num_heads, ffn_dim, dropout, scale, threshold, int8=int8, topk=topk)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)

    def _apply(self, fn):
        super()._apply(fn)
        self.lm_head.weight = self.token_embedding.weight
        return self

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None,
                past_key_values: list | None = None, activation_dtype: torch.dtype | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x).float()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss, None

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            logits, _, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).clamp(0, self.token_embedding.num_embeddings - 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    @torch.no_grad()
    def apply_bit_flips(self) -> None:
        for layer in self.layers:
            layer.apply_bit_flips()
