import torch
__all__ = ["TernarySSMBlock"]

import torch.nn as nn
import torch.nn.functional as F
from .layers import RMSNorm, StochasticTernaryLinear


class TernarySSMBlock(nn.Module):
    """Ternary State Space Model block (simplified Mamba).

    Architecture:
        x -> RMSNorm -> TernaryLinear(expand 2x) -> split x, gate
        x -> Conv1d -> SiLU -> SSM scan
        gate -> SiLU -> multiply -> TernaryLinear(project back)

    SSM recurrence (diagonal, per-channel):
        h_t = (1 - Δ·A) ⊙ h_{t-1} + Δ·B·x_t
        y_t = h_t  (identity observation, C=I)

    All projections use ternary weights.
    Recurrence is element-wise (d_state = inner_dim for simplicity).

    Args:
        hidden_dim: model dimension
        expand_factor: expansion factor for inner dim (default: 2)
        scale: ternary weight scale
        threshold: bit-flip threshold
        int8: use INT8 forward
    """

    def __init__(self, hidden_dim, expand_factor=2,
                 scale=1.0, threshold=None, int8=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        inner_dim = hidden_dim * expand_factor

        self.norm = RMSNorm(hidden_dim)

        # Fused input projection: hidden_dim -> 2 x inner_dim (x, gate)
        self.x_proj = StochasticTernaryLinear(
            hidden_dim, 2 * inner_dim, scale=scale, threshold=threshold, int8=int8
        )

        # Conv1d for local context (depthwise, same-length)
        self.conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size=3, padding=1, groups=inner_dim
        )

        # SSM parameters (per-channel)
        self.log_delta = nn.Parameter(torch.zeros(inner_dim))
        self.A_log = nn.Parameter(torch.zeros(inner_dim))

        # Output projection
        self.out_proj = StochasticTernaryLinear(
            inner_dim, hidden_dim, scale=scale, threshold=threshold, int8=int8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.norm(x)

        # Fused projection -> split
        fused = self.x_proj(x)  # (B, T, 2xinner)
        x_in, gate = fused.chunk(2, dim=-1)

        # Conv1d: (B, T, C) -> (B, C, T) -> Conv -> (B, C, T) -> (B, T, C)
        x_conv = self.conv(x_in.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM scan via parallel prefix: h_t = decay^t · cumsum(Δ·x_j / decay^j)
        # No Python loop — O(T) with 4 vectorized ops.
        inner_dim = x_conv.size(-1)
        delta = F.softplus(self.log_delta).view(1, inner_dim)
        # decay close to 1 to avoid underflow in decay^T for long sequences
        decay = (1 - delta * torch.sigmoid(self.A_log).view(1, inner_dim)).clamp(min=0.98, max=0.9999)
        T = x_conv.size(1)
        t_idx = torch.arange(T, device=x.device, dtype=torch.float32).view(T, 1)
        decay_pow = decay.pow(t_idx)
        s = delta * x_conv
        z = s / decay_pow.unsqueeze(0)
        cum = z.cumsum(dim=1)
        y = decay_pow.unsqueeze(0) * cum

        # Gate: y x SiLU(gate)
        out = y * F.silu(gate)
        return self.out_proj(out) + residual

    @torch.no_grad()
    def apply_bit_flips(self):
        self.x_proj.apply_bit_flips()
        self.out_proj.apply_bit_flips()
