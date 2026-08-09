"""Unit tests for ERC (Echo Residual Committer): residual decomposition,
commit math, forward parity, gradient flow, state-dict compatibility and the
export/commit path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ternary_llm.layers import TernaryLinear
from ternary_llm.transformer import TernaryTransformerBlock
from ternary_llm.erc import ERCLinear, enable_erc, commit_erc_state_dict


def _tiny_linear(seed=0, per_channel=False):
    torch.manual_seed(seed)
    return TernaryLinear(16, 8, per_channel=per_channel)


def _tiny_block(seed=0):
    torch.manual_seed(seed)
    return TernaryTransformerBlock(hidden_dim=32, num_heads=4, ffn_dim=64, dropout=0.0)


def test_enable_erc_replaces_all():
    model = _tiny_block()
    n_plain = sum(1 for m in model.modules() if isinstance(m, TernaryLinear))
    n = enable_erc(model)
    assert n == n_plain > 0
    assert all(isinstance(m, ERCLinear) for m in model.modules() if isinstance(m, TernaryLinear))
    n2 = enable_erc(model)
    assert n2 == 0


def test_zero_residual_parity():
    """R=0 (init) -> ERC forward == plain TernaryLinear forward exactly."""
    torch.manual_seed(0)
    m_plain = TernaryLinear(16, 8)
    torch.manual_seed(0)
    m_erc = TernaryLinear(16, 8)
    erc = ERCLinear.from_linear(m_erc)
    assert torch.equal(erc.residual, torch.zeros_like(erc.residual))
    x = torch.randn(4, 16)
    with torch.no_grad():
        y_plain = m_plain(x)
        y_erc = erc(x)
    assert torch.allclose(y_plain, y_erc, atol=1e-6)


def test_gradients_flow_to_both():
    m = _tiny_linear()
    erc = ERCLinear.from_linear(m)
    x = torch.randn(4, 16)
    y = erc(x).sum()
    y.backward()
    assert erc.latent_weights.grad is not None and erc.latent_weights.grad.abs().sum() > 0
    assert erc.residual.grad is not None and erc.residual.grad.abs().sum() > 0


def test_commit_threshold_math():
    """R is in LEVEL units (1.0 = one ternary level). Positions with
    |R| >= 0.5 commit: latent += sign*Delta (exactly one level), R -= sign
    (keeping the sub-level remainder)."""
    erc = ERCLinear.from_linear(_tiny_linear())
    delta = erc.get_quant_step()
    assert delta.item() > 0
    with torch.no_grad():
        erc.residual.zero_()
        erc.residual[0, 0] = 0.6          # >= 0.5 level -> commit +1
        erc.residual[0, 1] = -0.55        # <= -0.5 level -> commit -1
        erc.residual[0, 2] = 0.4          # < 0.5 level -> no commit
        latent_before = erc.latent_weights.clone()
        n_committed = erc.commit()
        assert n_committed == 2
        assert torch.allclose(erc.latent_weights[0, 0], latent_before[0, 0] + delta)
        assert torch.allclose(erc.latent_weights[0, 1], latent_before[0, 1] - delta)
        assert torch.allclose(erc.latent_weights[0, 2], latent_before[0, 2])
        # residual remainder: |R| < 0.5 everywhere
        assert erc.residual.abs().max().item() < 0.5 + 1e-6


def test_fullcarry_matches_state_dict():
    """commit(full=True) and commit_erc_state_dict produce the same latent."""
    torch.manual_seed(3)
    erc1 = ERCLinear.from_linear(TernaryLinear(16, 8, per_channel=True))
    torch.manual_seed(3)
    erc2 = ERCLinear.from_linear(TernaryLinear(16, 8, per_channel=True))
    erc2.load_state_dict(erc1.state_dict())  # identical latent + residual
    with torch.no_grad():
        erc1.residual.copy_(torch.randn(8, 16) * 2.0)
        erc2.residual.copy_(erc1.residual)
        n1 = erc1.commit(full=True)
        wrapper = torch.nn.Sequential(erc2)  # gives prefixed "0.residual" keys
        sd = {k: v.clone() for k, v in wrapper.state_dict().items()}
        n2 = commit_erc_state_dict(sd, {"per_channel": True, "ternary_scale": 1.0})
        assert n1 == n2
        assert torch.allclose(sd["0.latent_weights"], erc1.latent_weights)
        assert "residual" not in sd


def test_commit_all_drains_residual():
    """Repeated threshold commits drain R -> 0; residual stays bounded."""
    erc = ERCLinear.from_linear(_tiny_linear())
    torch.manual_seed(1)
    with torch.no_grad():
        erc.residual.normal_(0.0, 2.0)
        n = 0
        guard = 0
        while True:
            k = erc.commit()
            n += k
            guard += 1
            if k == 0:
                break
            assert guard < 1000
        assert erc.residual.abs().max().item() < 0.5 + 1e-6


def test_commit_is_output_neutral():
    """A threshold commit changes forward output only at the level
    boundaries it crosses: with R < 0.5 everywhere it changes nothing."""
    torch.manual_seed(7)
    erc = ERCLinear.from_linear(TernaryLinear(16, 8))
    x = torch.randn(4, 16)
    with torch.no_grad():
        erc.residual.normal_(0.0, 0.2)  # |R| < 0.5 everywhere (small tail)
        erc.residual.clamp_(-0.4, 0.4)
        y_before = erc(x)
        n = erc.commit()
        y_after = erc(x)
        assert n == 0
        assert torch.allclose(y_before, y_after, atol=1e-6)


def test_state_dict_compat():
    """ERC stores TernaryLinear keys plus an extra `residual`; a plain
    checkpoint loads into an ERC model with strict=False and R=0 gives the
    plain model's output."""
    torch.manual_seed(0)
    m = TernaryLinear(16, 8)
    torch.manual_seed(0)
    erc = ERCLinear.from_linear(TernaryLinear(16, 8))
    plain_keys = set(m.state_dict().keys())
    erc_keys = set(erc.state_dict().keys())
    assert plain_keys <= erc_keys
    assert "residual" in erc_keys
    missing, unexpected = erc.load_state_dict(m.state_dict(), strict=False)
    assert missing == ["residual"]
    assert unexpected == []
    assert torch.equal(erc.residual, torch.zeros_like(erc.residual))
    x = torch.randn(4, 16)
    with torch.no_grad():
        assert torch.allclose(m(x), erc(x), atol=1e-6)


def test_fp16_residual():
    erc = ERCLinear.from_linear(_tiny_linear())
    erc.to(torch.float16)
    assert erc.residual.dtype == torch.float16
    x = torch.randn(4, 16, dtype=torch.float16)
    y = erc(x)
    y.sum().backward()
    assert erc.residual.grad is not None
