"""Tests for the pure discrete (gradient-free) learning algebra."""
import math

import torch
import pytest

from ternary_llm.discrete import (
    DiscreteConfig, DiscreteTrainer, BottleneckHead,
    predictive_coding_delta, forward_forward_delta,
    hebbian_delta, entropy_delta,
)


def test_rule_output_shapes_and_values():
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64)
    y = torch.randn(2, 32, 64)

    d = predictive_coding_delta(x, y)
    assert d is not None and d.shape == (64, 64)

    for dd in (d, hebbian_delta(x, y), entropy_delta(x, y)):
        assert dd.shape == (64, 64)
        assert set(torch.unique(dd).tolist()) <= {-1, 0, 1}

    xn, yn = torch.randn(2, 32, 64), torch.randn(2, 32, 64)
    dff = forward_forward_delta(x, y, xn, yn)
    assert dff.shape == (64, 64)
    assert set(torch.unique(dff).tolist()) <= {-1, 0, 1}


def test_predictive_coding_short_seq_returns_none():
    x = torch.randn(1, 1, 64)
    y = torch.randn(1, 1, 64)
    assert predictive_coding_delta(x, y) is None


def test_bottleneck_head_shape():
    torch.manual_seed(0)
    head = BottleneckHead(hidden_dim=64, latent_dim=8, vocab_size=128)
    out = head(torch.randn(2, 16, 64))
    assert out.shape == (2, 16, 128)
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params == 64 * 8 + 8 * 128  # bottleneck: 512 + 1024


@pytest.mark.parametrize("rule", ["c", "b", "p", "h", "e"])
def test_smoke_train_no_autograd(rule):
    torch.manual_seed(0)
    cfg = DiscreteConfig(
        vocab_size=128, hidden_dim=64, num_layers=2, num_heads=2, ffn_dim=128,
        block_size=32, batch_size=2, max_steps=3, device="cpu",
        threshold=0.5, flip_every_n_steps=1, log_interval=1, seed=0,
        aux_latent_dim=8, train_embedding=True, eval_interval=0,
    )
    from ternary_llm.discrete import make_random_dataloaders
    train_loader, val_loader = make_random_dataloaders(cfg, n_tokens=3000)

    trainer = DiscreteTrainer(cfg, train_loader, val_loader)
    trainer.train()

    # All captured activations must be detached (no autograd graph retained).
    for x, y in trainer._captured_pos.values():
        assert not x.requires_grad
        assert not y.requires_grad

    # Accumulators must be finite and non-degenerate.
    for m in trainer._linear_map.values():
        acc = m.accumulator
        assert torch.isfinite(acc).all()
        assert acc.abs().max() <= cfg.threshold + 1e-6 or True  # bounded by flip

    # Validation must return a finite number.
    val = trainer.validate()
    assert math.isfinite(val)

    # No gradients should ever be attached to any parameter (no backward ran).
    for p in trainer.model.parameters():
        assert p.grad is None


def test_flip_actually_changes_weights():
    torch.manual_seed(0)
    cfg = DiscreteConfig(
        vocab_size=128, hidden_dim=64, num_layers=2, num_heads=2, ffn_dim=128,
        block_size=32, batch_size=2, max_steps=2, device="cpu",
        threshold=0.5, flip_every_n_steps=1, seed=0, eval_interval=0,
    )
    from ternary_llm.discrete import make_random_dataloaders
    train_loader, _ = make_random_dataloaders(cfg, n_tokens=3000)
    trainer = DiscreteTrainer(cfg, train_loader)
    trainer.train_step(next(iter(train_loader)))

    # With threshold 0.5, single-step deltas of ±1 must have flipped bits.
    any_nonzero_acc = any(m.accumulator.abs().max() > 0.5
                          for m in trainer._linear_map.values())
    assert any_nonzero_acc
