import torch
import pytest
import math
from ternary_llm.layers import TernaryLinear, RMSNorm
from ternary_llm.quantization import (
    Int8StochasticBitFlipLinear,
    pack_ternary_tensor,
)


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
        # v7 outlier encoding allows ±2 (code 11); ternary set is {-2,-1,0,1,2}.
        assert all(v in [-2.0, -1.0, 0.0, 1.0, 2.0] for v in unique.tolist())

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

    def test_group_alpha_backward(self):
        layer = TernaryLinear(7, 5, group_size=3)
        x = torch.randn(2, 4, 7, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert layer.latent_weights.grad is not None
        assert layer.alphas.grad is not None
        assert layer.alphas.grad.shape == (5, 3)


class TestInt8StochasticBitFlipLinear:
    """Regression tests for quantized activations with alpha scaling."""

    def test_group_alpha_is_applied_and_has_gradient(self):
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0]], requires_grad=True)
        w_raw = torch.tensor([
            [1.0, 0.0, -1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, -1.0, 1.0],
        ])
        packed = pack_ternary_tensor(w_raw)
        accumulator = torch.zeros_like(w_raw)
        outlier_signs = torch.zeros(0, dtype=torch.uint8)
        alphas = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], requires_grad=True)

        out = Int8StochasticBitFlipLinear.apply(
            x, packed, w_raw, 1.0, accumulator, 20.0, outlier_signs,
            1.0, False, alphas, 2,
        )
        max_abs = x.detach().abs().max()
        scale_x = max_abs / 127.0
        x_q = (x.detach() / scale_x).round().clamp(-128, 127).to(torch.int8)
        expanded = torch.repeat_interleave(alphas.detach(), 2, dim=1)[:, :5]
        expected = torch.nn.functional.linear(x_q.float() * scale_x, w_raw * expanded)
        assert torch.allclose(out.detach(), expected, atol=1e-5, rtol=1e-5)

        out.sum().backward()
        assert x.grad is not None
        assert alphas.grad is not None
        assert alphas.grad.shape == alphas.shape

    def test_no_alpha_output_is_dequantized(self):
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], requires_grad=True)
        w_raw = torch.tensor([[1.0, 0.0, -1.0, 1.0]])
        packed = pack_ternary_tensor(w_raw)
        accumulator = torch.zeros_like(w_raw)
        outlier_signs = torch.zeros(0, dtype=torch.uint8)

        out = Int8StochasticBitFlipLinear.apply(
            x, packed, w_raw, 0.5, accumulator, 20.0, outlier_signs,
            1.0, False, None, 0,
        )
        scale_x = x.detach().abs().max() / 127.0
        x_q = (x.detach() / scale_x).round().clamp(-128, 127).to(torch.int8)
        expected = torch.nn.functional.linear(x_q.float() * scale_x, w_raw) * 0.5
        assert torch.allclose(out.detach(), expected, atol=1e-5, rtol=1e-5)
