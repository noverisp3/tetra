import torch
__all__ = [
    "HybridBlock", "HybridTransformerModel",
]

import torch.nn as nn
import torch.nn.functional as F
from .layers import RMSNorm, TopKActivation
from .attention import StochasticMultiHeadAttention
from .ssm import TernarySSMBlock


class HybridBlock(nn.Module):
    """Single hybrid block: either SSM or Attention, based on position.

    Every `ssm_every`-th block is Attention, the rest are SSM.
    Each block has RMSNorm + TopK + SSM|Attn + FFN.
    """

    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.0,
                 scale=1.0, threshold=None, int8=False, topk=1.0, expand_factor=2,
                 is_attention=False):
        super().__init__()
        self.is_attention = is_attention
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn_topk = TopKActivation(topk)

        if is_attention:
            self.mix = StochasticMultiHeadAttention(
                hidden_dim, num_heads, dropout, scale, threshold, int8=int8
            )
        else:
            self.mix = TernarySSMBlock(
                hidden_dim, expand_factor=expand_factor,
                scale=scale, threshold=threshold, int8=int8
            )

        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn_topk = TopKActivation(topk)
        from .ffn import StochasticFFN
        self.ffn = StochasticFFN(hidden_dim, ffn_dim, dropout, scale, threshold, int8=int8)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        r = x
        x = self.attn_norm(x)
        x = self.attn_topk(x)
        x = self.mix(x, mask=mask) if self.is_attention else self.mix(x)
        x = x + r
        r = x
        x = self.ffn_norm(x)
        x = self.ffn_topk(x)
        x = self.ffn(x)
        x = x + r
        return x

    @torch.no_grad()
    def apply_bit_flips(self):
        if hasattr(self.mix, 'apply_bit_flips'):
            self.mix.apply_bit_flips()
        self.ffn.apply_bit_flips()


class HybridTransformerModel(nn.Module):
    """Hybrid SSM-Attention transformer.

    80% layers are TSSM (Ternary SSM), 20% are Attention.
    Default: every 5 blocks, 4 SSM + 1 Attention (ssm_every=5).

    Args:
        vocab_size: vocabulary size
        hidden_dim: model dimension
        num_layers: total layers
        num_heads: attention heads (for attention layers)
        ffn_dim: FFN hidden dimension
        max_seq_len: maximum context length
        dropout: dropout rate
        scale: ternary weight scale
        threshold: bit-flip threshold
        int8: use INT8 forward
        topk: keep top-k fraction of activations
        expand_factor: SSM expansion factor
        ssm_every: place attention every N blocks (default 5)
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, ffn_dim,
                 max_seq_len=2048, dropout=0.0, scale=1.0, threshold=None,
                 int8=False, topk=1.0, expand_factor=2, ssm_every=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Build layers: attention every `ssm_every` blocks, rest SSM
        # Ensure at least 1 attention layer even for small models
        attn_interval = max(2, min(ssm_every, num_layers - 1))
        layers = []
        for i in range(num_layers):
            is_attn = (i % attn_interval == attn_interval - 1) or (i == num_layers - 1 and num_layers > 1)
            layers.append(HybridBlock(
                hidden_dim, num_heads, ffn_dim, dropout, scale, threshold,
                int8, topk, expand_factor, is_attention=is_attn,
            ))
        self.layers = nn.ModuleList(layers)

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
        input_ids = input_ids.clamp(0, self.token_embedding.num_embeddings - 1)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)
        if activation_dtype is not None:
            x = x.to(activation_dtype)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x).float()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
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
    def apply_bit_flips(self):
        for layer in self.layers:
            layer.apply_bit_flips()
