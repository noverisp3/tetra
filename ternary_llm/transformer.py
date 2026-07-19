import torch
import torch.nn as nn
import math
from .layers import RMSNorm
from .attention import TernaryMultiHeadAttention
from .ffn import TernaryFFN


class TernaryTransformerBlock(nn.Module):
    """Single transformer decoder block with ternary weights.

    Architecture (Pre-Norm, following BitNet b1.58):
        x → RMSNorm → MultiHeadAttention → Residual Add
          → RMSNorm → FFN → Residual Add

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
    ):
        super().__init__()

        # Attention block with pre-norm
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = TernaryMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # FFN block with pre-norm
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = TernaryFFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention block with residual connection
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, mask=mask)
        x = x + residual

        # FFN block with residual connection
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual

        return x


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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: (batch_size, seq_len) token ids
            targets: (batch_size, seq_len) target token ids for loss

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape

        # Create causal mask
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=input_ids.device)
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        # Final norm
        x = self.norm(x)

        # LM head
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get last token logits
            logits = logits[:, -1, :] / temperature
            logits = torch.nan_to_num(logits)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = nn.functional.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs)
            probs = probs.clamp(min=1e-10)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Clamp to vocab size
            idx_next = idx_next.clamp(0, self.token_embedding.num_embeddings - 1)

            # Append
            input_ids = torch.cat([input_ids, idx_next], dim=1)

        return input_ids
