import torch
import pytest
from ternary_llm.attention import TernaryMultiHeadAttention
from ternary_llm.ffn import TernaryFFN
from ternary_llm.transformer import TernaryTransformerBlock, TernaryTransformerModel


class TestTernaryMultiHeadAttention:
    """Tests for Multi-Head Attention."""

    def test_output_shape(self):
        attn = TernaryMultiHeadAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == (2, 10, 64)

    def test_causal_mask(self):
        attn = TernaryMultiHeadAttention(hidden_dim=64, num_heads=4)
        x = torch.randn(1, 5, 64)
        mask = torch.tril(torch.ones(5, 5)).unsqueeze(0).unsqueeze(0)
        out = attn(x, mask=mask)
        assert out.shape == (1, 5, 64)

    def test_gradient_flows(self):
        attn = TernaryMultiHeadAttention(hidden_dim=32, num_heads=2)
        x = torch.randn(1, 5, 32, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestTernaryFFN:
    """Tests for Feed-Forward Network."""

    def test_output_shape(self):
        ffn = TernaryFFN(hidden_dim=64, ffn_dim=256)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_gradient_flows(self):
        ffn = TernaryFFN(hidden_dim=32, ffn_dim=128)
        x = torch.randn(1, 5, 32, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestTernaryTransformerBlock:
    """Tests for Transformer Block."""

    def test_output_shape(self):
        block = TernaryTransformerBlock(
            hidden_dim=64, num_heads=4, ffn_dim=256
        )
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_residual_connection(self):
        block = TernaryTransformerBlock(
            hidden_dim=32, num_heads=2, ffn_dim=128
        )
        x = torch.randn(1, 5, 32)
        out = block(x)
        # Output should be different from input (not just identity)
        assert not torch.allclose(x, out, atol=1e-6)

    def test_gradient_flows(self):
        block = TernaryTransformerBlock(
            hidden_dim=32, num_heads=2, ffn_dim=128
        )
        x = torch.randn(1, 5, 32, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestTernaryTransformerModel:
    """Tests for full Transformer Model."""

    def test_forward_shape(self):
        model = TernaryTransformerModel(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=256,
        )
        input_ids = torch.randint(0, 1000, (2, 10))
        logits, loss = model(input_ids)
        assert logits.shape == (2, 10, 1000)
        assert loss is None

    def test_forward_with_targets(self):
        model = TernaryTransformerModel(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=256,
        )
        input_ids = torch.randint(0, 1000, (2, 10))
        targets = torch.randint(0, 1000, (2, 10))
        logits, loss = model(input_ids, targets)
        assert logits.shape == (2, 10, 1000)
        assert loss is not None
        assert loss.item() > 0

    def test_gradient_flows(self):
        model = TernaryTransformerModel(
            vocab_size=500,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            ffn_dim=64,
        )
        input_ids = torch.randint(0, 500, (1, 5))
        targets = torch.randint(0, 500, (1, 5))
        logits, loss = model(input_ids, targets)
        loss.backward()
        # Check gradients exist for key parameters
        assert model.token_embedding.weight.grad is not None
        assert model.layers[0].attn.q_proj.latent_weights.grad is not None

    def test_generate(self):
        model = TernaryTransformerModel(
            vocab_size=100,
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            ffn_dim=64,
        )
        input_ids = torch.randint(0, 100, (1, 5))
        output = model.generate(input_ids, max_new_tokens=10)
        assert output.shape == (1, 15)  # 5 original + 10 new

    def test_num_parameters(self):
        model = TernaryTransformerModel(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=256,
        )
        total = sum(p.numel() for p in model.parameters())
        assert total > 0
        print(f"Total parameters: {total:,}")
