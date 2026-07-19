import torch
import pytest
import math
from ternary_llm.quantization import TernaryQuantizer, Int8Quantizer, ternary_quantize, int8_quantize


class TestTernaryQuantizer:
    """Tests for ternary weight quantization."""

    def test_output_values_are_ternary(self):
        w = torch.randn(64, 64) * 2
        w_ternary = TernaryQuantizer.apply(w)
        unique = w_ternary.unique()
        assert all(v in [-1.0, 0.0, 1.0] for v in unique.tolist())

    def test_output_shape_preserved(self):
        w = torch.randn(32, 16)
        w_ternary = TernaryQuantizer.apply(w)
        assert w_ternary.shape == w.shape

    def test_zero_weights_stay_zero(self):
        w = torch.zeros(10, 10)
        w_ternary = TernaryQuantizer.apply(w)
        assert (w_ternary == 0).all()

    def test_large_positive_become_one(self):
        w = torch.ones(5, 5) * 100
        w_ternary = TernaryQuantizer.apply(w)
        assert (w_ternary == 1).all()

    def test_large_negative_become_minus_one(self):
        w = torch.ones(5, 5) * -100
        w_ternary = TernaryQuantizer.apply(w)
        assert (w_ternary == -1).all()

    def test_gradient_flows_through(self):
        w = torch.randn(8, 8, requires_grad=True)
        w_ternary = TernaryQuantizer.apply(w)
        loss = w_ternary.sum()
        loss.backward()
        assert w.grad is not None
        assert w.grad.shape == w.shape

    def test_gradient_is_not_none(self):
        w = torch.randn(4, 4, requires_grad=True)
        w_ternary = TernaryQuantizer.apply(w)
        loss = (w_ternary ** 2).sum()
        loss.backward()
        assert w.grad is not None

    def test_ternary_ratio(self):
        w = torch.randn(1000, 1000)
        w_ternary = TernaryQuantizer.apply(w)
        zeros = (w_ternary == 0).float().mean()
        assert 0.0 < zeros < 1.0, "Should have some zeros but not all"


class TestInt8Quantizer:
    """Tests for INT8 activation quantization (fake-quantize)."""

    def test_output_is_float(self):
        x = torch.randn(32, 64)
        x_q = Int8Quantizer.apply(x)
        assert x_q.dtype == torch.float32

    def test_output_shape_preserved(self):
        x = torch.randn(16, 32)
        x_q = Int8Quantizer.apply(x)
        assert x_q.shape == x.shape

    def test_simulated_quantization_effect(self):
        x = torch.randn(32, 64) * 100
        x_q = Int8Quantizer.apply(x)
        # After fake-quantize, values should be at INT8-level granularity
        # but dequantized back to float scale
        scale = x.abs().max() / 127.0
        expected_levels = (x_q / scale).round()
        assert torch.allclose(x_q, expected_levels * scale, atol=1e-5)

    def test_gradient_flows_through(self):
        x = torch.randn(8, 8, requires_grad=True)
        x_q = Int8Quantizer.apply(x)
        loss = x_q.sum()
        loss.backward()
        assert x.grad is not None

    def test_near_zero_input(self):
        x = torch.randn(4, 4) * 0.001
        x_q = Int8Quantizer.apply(x)
        assert x_q.dtype == torch.float32


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_ternary_quantize_returns_ternary(self):
        w = torch.randn(32, 32)
        w_ternary = ternary_quantize(w)
        unique = w_ternary.unique()
        assert all(v in [-1.0, 0.0, 1.0] for v in unique.tolist())

    def test_int8_quantize_returns_float(self):
        x = torch.randn(32, 32)
        x_q = int8_quantize(x)
        assert x_q.dtype == torch.float32
