import torch
import pytest
import math
from ternary_llm.layers import TernaryLinear, RMSNorm


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(32)
        x = torch.randn(4, 8, 32) * 100
        out = norm(x)
        # After RMSNorm, RMS of each vector should be ~1
        rms = out.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_learnable_weight(self):
        dim = 16
        norm = RMSNorm(dim)
        assert norm.weight.shape == (dim,)
        assert norm.weight.requires_grad

    def test_gradient_flows(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert norm.weight.grad is not None


class TestTernaryLinear:
    """Tests for TernaryLinear layer."""

    def test_output_shape(self):
        layer = TernaryLinear(64, 128)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == (2, 10, 128)

    def test_no_bias(self):
        layer = TernaryLinear(32, 64, bias=False)
        assert layer.bias is None

    def test_with_bias(self):
        layer = TernaryLinear(32, 64, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (64,)

    def test_latent_weights_exist(self):
        layer = TernaryLinear(16, 32)
        assert hasattr(layer, "latent_weights")
        assert layer.latent_weights.shape == (32, 16)
        assert layer.latent_weights.requires_grad

    def test_gradient_flows_through(self):
        layer = TernaryLinear(32, 64)
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.latent_weights.grad is not None

    def test_get_ternary_weights(self):
        layer = TernaryLinear(16, 32)
        w_ternary = layer.get_ternary_weights()
        unique = w_ternary.unique()
        assert all(v in [-1.0, 0.0, 1.0] for v in unique.tolist())

    def test_get_num_bits(self):
        layer = TernaryLinear(16, 32)
        bits = layer.get_num_bits()
        # log2(3) * 16 * 32
        expected = math.ceil(math.log2(3)) * 16 * 32
        assert bits == expected

    def test_batch_dimension_agnostic(self):
        layer = TernaryLinear(32, 64)
        # Single sample
        x1 = torch.randn(1, 32)
        out1 = layer(x1)
        assert out1.shape == (1, 64)

        # Batch
        x2 = torch.randn(8, 32)
        out2 = layer(x2)
        assert out2.shape == (8, 64)

    def test_sequence_dimension(self):
        layer = TernaryLinear(32, 64)
        x = torch.randn(2, 20, 32)
        out = layer(x)
        assert out.shape == (2, 20, 64)
