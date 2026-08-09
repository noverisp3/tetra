"""ERC (Echo Residual Committer) — two-timescale memory for the ternary core.

Decomposes every ternary weight matrix into a slow discrete core and a fast
continuous residual:

    W_effective = Ternary(W_core) + R

Phase 1 (fast adaptation — "absorbing surprise"):
    W_core stays frozen or gets a very small LR; the residual tensor R
    (FP16/FP32, initialised to 0) learns from the new data stream at a high
    LR with an optional leaky EMA decay. R acts as a shock absorber for the
    new-domain gradient noise, so the long-term ternary memory is never
    overwritten directly.

Phase 2 (consolidation — threshold committing):
    Periodically, every position (i, j) with |R_ij| >= Delta/2 commits its
    energy into the core:

        W_core,ij <- W_core,ij + sign(R_ij)
        R_ij      <- R_ij - sign(R_ij) * Delta

    Delta = quantizer step = scale * mean(|W_core|) (per-tensor) or
    scale * rowmean (per-channel), matching FusedTernaryLinear. After the
    carry, the residual keeps only the sub-step remainder, so a commit is
    approximately output-neutral — it moves energy, not signal.

At export time a full carry ("freeze & commit") bakes R into the latent
weights, the residual is zeroed, and the standard ternary export path
produces a plain 2-bit binary that the C++ engine loads unchanged.

This is the PyTorch-side implementation of the hippocampal (fast) /
neocortical (slow) consolidation analogy applied to ternary weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ERCLinear", "enable_erc", "commit_erc", "decay_erc",
    "erc_module_stats", "commit_erc_state_dict",
]

from .layers import TernaryLinear
from .quantization import FusedTernaryLinear


class ERCLinear(TernaryLinear):
    """TernaryLinear with an attached fast residual path (Echo Residual).

    forward:  FusedTernaryLinear(x, W_latent) + F.linear(x, R)

    R is stored in LEVEL units (same scale as the quantizer's integer output
    round(W/Delta)): a residual of 1.0 is exactly one ternary level, so the
    commit threshold |R| >= 0.5 means "crossing a quantization boundary".
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        ternary_scale: float = 1.0,
        per_channel: bool = False,
        group_size: int = 0,
        init_mode: str = "kaiming",
        residual_dtype: str = "fp32",
    ):
        super().__init__(
            in_features, out_features, bias=bias,
            ternary_scale=ternary_scale, per_channel=per_channel,
            group_size=group_size, init_mode=init_mode,
        )
        if residual_dtype not in ("fp32", "fp16"):
            raise ValueError(f"residual_dtype must be 'fp32' or 'fp16', got {residual_dtype}")
        self.residual_dtype = residual_dtype
        # R: fast residual (init 0 — the echo starts silent).
        self.residual = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.float32)
        )
        self._residual_cast_cache = {}

    @classmethod
    def from_linear(cls, linear: TernaryLinear, residual_dtype: str = "fp32") -> "ERCLinear":
        """Build an ERCLinear from an existing TernaryLinear (data copied)."""
        erc = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            ternary_scale=linear.ternary_scale,
            per_channel=linear.per_channel,
            group_size=linear.group_size,
            init_mode="kaiming",
            residual_dtype=residual_dtype,
        )
        erc.latent_weights.data.copy_(linear.latent_weights.data)
        if linear.alphas is not None and erc.alphas is not None:
            erc.alphas.data.copy_(linear.alphas.data)
        if linear.bias is not None:
            erc.bias.data.copy_(linear.bias.data)
        erc.soft_gamma = getattr(linear, "soft_gamma", None)
        return erc

    def _residual_for(self, dtype: torch.dtype) -> torch.Tensor:
        """Cached cast of the residual to the activation dtype (no-op if fp32)."""
        if self.residual.dtype == dtype:
            return self.residual
        cached = self._residual_cast_cache.get(dtype)
        if cached is None:
            cached = self.residual.detach().clone().to(dtype)
            self._residual_cast_cache[dtype] = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = FusedTernaryLinear.apply(
            x, self.latent_weights, self.ternary_scale, self.per_channel,
            self.alphas, self.group_size, self.soft_gamma,
        )
        output = output + F.linear(x, self._residual_for(x.dtype))
        if self.bias is not None:
            output = output + self.bias
        return output

    def get_quant_step(self) -> torch.Tensor:
        """Delta = scale * mean|W| (per-tensor) or scale * rowmean (per-channel).

        Matches FusedTernaryLinear's threshold: a Delta-sized latent bump
        crosses exactly one quantization level (the quantizer is
        round(W/Delta) with no post-round Delta multiply).
        """
        w = self.latent_weights.float()
        if self.per_channel:
            return w.abs().mean(dim=1, keepdim=True).clamp(min=1e-6) * self.ternary_scale
        return w.abs().mean().clamp(min=1e-6) * self.ternary_scale

    @torch.no_grad()
    def commit(self, full: bool = False) -> int:
        """Carry residual energy into the latent core.

        The residual R is stored in LEVEL units (1.0 = one ternary level).

        Threshold commit (full=False): positions with |R| >= 0.5 commit:
            W_core += sign(R) * Delta        (crosses exactly one level)
            R      -= sign(R)                (exactly one level, keeps the
                                              sub-level remainder)
        Full commit (full=True, export time): W_core += round(R) * Delta,
        R -= round(R) - the remainder is discarded at export.

        This is output-neutral: a Delta-sized latent bump shifts the
        quantizer's integer output by exactly +-1, which the residual
        decrease cancels 1:1.
        """
        latent = self.latent_weights
        R = self.residual.float()
        delta = self.get_quant_step()
        if full:
            carry = R.round()
        else:
            carry = torch.where(R.abs() >= 0.5, torch.sign(R), torch.zeros_like(R))
        n_changed = int((carry.abs() >= 1).sum().item())
        if n_changed == 0:
            return 0
        latent.data.add_(carry * delta)
        # R <- R - carry (level units; keeps only the sub-level remainder)
        self.residual.data.copy_((R - carry).to(self.residual.dtype))
        if self._residual_cast_cache:
            self._residual_cast_cache.clear()
        return n_changed

    def commit_ratio(self) -> float:
        """Fraction of positions currently past the 0.5-level commit threshold."""
        R = self.residual.float()
        over = (R.abs() >= 0.5)
        return float(over.float().mean().item())

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"{base}, residual={self.residual_dtype}"


def enable_erc(model: nn.Module, residual_dtype: str = "fp32") -> int:
    """Replace every TernaryLinear in ``model`` with an ERCLinear (in place).

    State-dict keys are unchanged (only the module class changes), so ERC
    checkpoints load into plain models and vice versa with strict=False
    (residual defaults to zero).

    Returns the number of layers converted.
    """
    n = 0
    for parent in model.modules():
        for key, child in list(parent._modules.items()):
            if isinstance(child, TernaryLinear) and not isinstance(child, ERCLinear):
                erc = ERCLinear.from_linear(child, residual_dtype=residual_dtype)
                erc.to(child.latent_weights.device)  # child may live on GPU/DML already
                parent._modules[key] = erc
                n += 1
    return n


@torch.no_grad()
def commit_erc(model: nn.Module, full: bool = False) -> tuple[int, int]:
    """Run the commit pass on every ERCLinear in the model.

    Returns (n_committed, n_positions) across all layers.
    """
    n_committed = 0
    n_positions = 0
    for m in model.modules():
        if isinstance(m, ERCLinear):
            n_committed += m.commit(full=full)
            n_positions += m.residual.numel()
    return n_committed, n_positions


@torch.no_grad()
def decay_erc(model: nn.Module, factor: float) -> None:
    """Leaky EMA decay of every residual: R *= factor (factor < 1 fades old echoes)."""
    if factor >= 1.0:
        return
    for m in model.modules():
        if isinstance(m, ERCLinear):
            m.residual.mul_(factor)
            if m._residual_cast_cache:
                m._residual_cast_cache.clear()


@torch.no_grad()
def erc_module_stats(model: nn.Module) -> dict:
    """Per-layer residual energy + commit ratios (diagnostics)."""
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, ERCLinear):
            R = m.residual.float()
            stats[name] = {
                "rms": float(R.pow(2).mean().sqrt().item()),
                "max_abs": float(R.abs().max().item()),
                "over_thr": m.commit_ratio(),
                "delta": float(m.get_quant_step().mean().item()),
            }
    return stats


def commit_erc_state_dict(sd: dict, config: dict) -> int:
    """Export-time full carry on a raw state dict (no model needed).

    The residual R is stored in LEVEL units (1.0 = one ternary level), so
    for every ``*.residual`` tensor: latent += round(R)*Delta, then the
    residual key is dropped so the plain ternary model / binary exporter
    never sees it. Delta uses the checkpoint's per_channel / ternary_scale
    so it matches training.

    Returns the number of committed positions.
    """
    per_channel = bool(config.get("per_channel", False))
    scale = float(config.get("ternary_scale", 1.0))
    n_committed = 0
    for key in list(sd.keys()):
        if not key.endswith(".residual"):
            continue
        latent_key = key[: -len(".residual")] + ".latent_weights"
        if latent_key not in sd:
            continue
        latent = sd[latent_key].float()
        R = sd[key].float()
        if per_channel:
            delta = latent.abs().mean(dim=1, keepdim=True).clamp(min=1e-6) * scale
        else:
            delta = latent.abs().mean().clamp(min=1e-6) * scale
        carry = R.round()
        latent = latent + carry * delta
        n_committed += int((carry.abs() >= 1).sum().item())
        sd[latent_key] = latent.to(sd[latent_key].dtype)
        del sd[key]
    return n_committed
