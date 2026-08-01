"""Pure discrete (gradient-free) learning algebra for Tetra.

Replaces global backpropagation with LOCAL update rules that feed the
existing stochastic bit-flip machinery (``StochasticTernaryLinear.accumulator``
+ ``apply_bit_flips``). No autograd graph is built for the ternary weights, so
training memory is O(1) w.r.t. the number of layers.

Rules (see ``DiscreteConfig.rule``):

- ``'c'``  Predictive Coding (default, recommended):
           ``e = y(t) - y(t-1)`` temporal error, ``delta = -sign(x^T e)``.
           The error signal is self-correcting (negative feedback), so the
           accumulator does not saturate.
- ``'b'``  Forward-Forward:
           ``goodness = sum(y^2)`` over two passes (positive / corrupted
           negative), ``delta = sign(x_pos^T y_pos - x_neg^T y_neg)``.
- ``'p'``  Target Predictive Coding:
           the top block is driven by the real lm_head cross-entropy error;
           intermediate blocks are driven by per-block ``BottleneckHead``
           cross-entropy error (bottleneck ``hidden -> latent -> vocab`` to
           keep the aux heads tiny).
- ``'h'``  Hebbian (EXPERIMENTAL): ``delta = sign(x^T y)``.
           Saturation / weight explosion risk — use with a high threshold.
- ``'e'``  Entropy (EXPERIMENTAL): push outputs toward concentrated
           (low normalized entropy) distributions via a centered residual.
"""
import math
from dataclasses import dataclass, field
from functools import partial

__all__ = [
    "DiscreteConfig", "DiscreteTrainer", "BottleneckHead",
    "predictive_coding_delta", "forward_forward_delta",
    "hebbian_delta", "entropy_delta", "build_model_from_config",
    "random_token_array", "make_random_dataloaders",
]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .layers import StochasticTernaryLinear
from .transformer import StochasticTransformerModel
from .data import ChunkedDataset, create_dataloaders


# ──────────────────────────────────────────────────────────────
# Local delta rules
# ──────────────────────────────────────────────────────────────

def predictive_coding_delta(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
    """Rule C: temporal predictive coding.

    ``e = y(t) - y(t-1)`` (each position predicts its successor), then
    ``delta = -sign(x^T e)`` — the descent direction of ``0.5||e||^2``,
    matching the accumulator convention used by ``StochasticBitFlipLinear``.

    Args:
        x: layer input (batch, seq, in_features)
        y: layer output (batch, seq, out_features)

    Returns:
        (out_features, in_features) delta in {-1, 0, +1}, or None if seq < 2.
    """
    if y.size(-2) < 2:
        return None
    xf = x[:, :-1, :].reshape(-1, x.size(-1))
    e = (y[:, 1:, :] - y[:, :-1, :]).reshape(-1, y.size(-1))
    grad = e.t() @ xf  # (out, in)
    return -torch.sign(grad)


def forward_forward_delta(
    x_pos: torch.Tensor, y_pos: torch.Tensor,
    x_neg: torch.Tensor, y_neg: torch.Tensor,
) -> torch.Tensor:
    """Rule B: Forward-Forward goodness update.

    ``delta = sign(x_pos^T y_pos - x_neg^T y_neg)`` — increase goodness
    (sum of squared activations) on the positive pass, decrease on the
    negative pass.

    Args:
        x_pos/y_pos: input/output of the positive pass
        x_neg/y_neg: input/output of the corrupted negative pass

    Returns:
        (out_features, in_features) delta in {-1, 0, +1}.
    """
    xp = x_pos.reshape(-1, x_pos.size(-1))
    yp = y_pos.reshape(-1, y_pos.size(-1))
    xn = x_neg.reshape(-1, x_neg.size(-1))
    yn = y_neg.reshape(-1, y_neg.size(-1))
    grad = yp.t() @ xp - yn.t() @ xn  # (out, in)
    return torch.sign(grad)


def hebbian_delta(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Rule A: Hebbian correlation (EXPERIMENTAL).

    ``delta = sign(x^T y)``. No error signal, so consistent correlated input
    pushes weights toward saturation. Use with a high threshold.
    """
    xf = x.reshape(-1, x.size(-1))
    yf = y.reshape(-1, y.size(-1))
    grad = yf.t() @ xf  # (out, in)
    return torch.sign(grad)


def entropy_delta(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Rule D: local entropy minimization (EXPERIMENTAL).

    Uses a centered residual ``e = y - mean(y)`` as the error — pushing each
    channel away from the channel mean concentrates energy (lower normalized
    entropy). ``delta = -sign(x^T e)``.
    """
    xf = x.reshape(-1, x.size(-1))
    e = (y - y.mean(dim=-1, keepdim=True)).reshape(-1, y.size(-1))
    grad = e.t() @ xf  # (out, in)
    return -torch.sign(grad)


# Temporal fallback for rule 'p' modules that have no target signal.
_TEMPORAL_FALLBACK = {"c", "e"}


# ──────────────────────────────────────────────────────────────
# Bottleneck aux head (keeps auxiliary parameters tiny)
# ──────────────────────────────────────────────────────────────

class BottleneckHead(nn.Module):
    """Small next-token head used to give a block a local target.

    Architecture ``hidden -> latent -> vocab`` keeps the auxiliary parameter
    count small (e.g. 256 -> 32 -> 8192 ≈ 270K vs a full 256 -> 8192 ≈ 2.1M).
    """

    def __init__(self, hidden_dim: int, latent_dim: int, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.down = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.up = nn.Linear(latent_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map hidden states to vocab logits.

        Args:
            x: (batch, seq, hidden_dim)

        Returns:
            (batch, seq, vocab_size) logits.
        """
        return self.up(F.silu(self.down(x)))

    def extra_repr(self) -> str:
        return (f"hidden={self.hidden_dim}, latent={self.latent_dim}, "
                f"vocab={self.vocab_size}")


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

@dataclass
class DiscreteConfig:
    """Configuration for the discrete (gradient-free) trainer."""
    # Model
    vocab_size: int = 8192
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    max_seq_len: int = 2048

    # Rule
    rule: str = "c"  # c = predictive coding (default), b = forward-forward,
                     # p = target PC (needs --aux for intermediate blocks),
                     # h = hebbian (exp), e = entropy (exp)

    # Data
    block_size: int = 128
    batch_size: int = 16
    val_split: float = 0.05
    max_steps: int = 1000
    device: str = "cpu"

    # Bit-flip dynamics
    threshold: float = 20.0
    threshold_decay_to: float | None = None
    flip_every_n_steps: int = 5
    acc_decay: float = 0.99  # leaky accumulator: acc *= acc_decay each step

    # Embedding (FP32, trained with plain local SGD since it is not ternary)
    train_embedding: bool = True
    lr_embedding: float = 1e-4
    wd_embedding: float = 0.1  # decoupled weight decay keeps ||E|| bounded

    # Aux heads (rule 'p')
    aux_latent_dim: int = 32
    aux_lr: float = 3e-4

    # FF negative corruption
    ff_corrupt: float = 0.3

    # Logging / validation
    log_interval: int = 10
    eval_interval: int = 200
    eval_steps: int = 20

    # Misc
    seed: int = 0
    save_dir: str = "checkpoints_discrete"


# ──────────────────────────────────────────────────────────────
# Model / data helpers
# ──────────────────────────────────────────────────────────────

def build_model_from_config(config: DiscreteConfig) -> StochasticTransformerModel:
    """Build a fresh stochastic bit-flip model from a DiscreteConfig."""
    return StochasticTransformerModel(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.max_seq_len,
        threshold=config.threshold,
    )


def random_token_array(n_tokens: int, vocab_size: int, seed: int = 0) -> np.ndarray:
    """Generate a random uint16 token array for smoke tests / --random-data."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)


def make_random_dataloaders(config: DiscreteConfig, n_tokens: int = 5000):
    """Create train/val DataLoaders from synthetic random tokens."""
    tokens = random_token_array(n_tokens, config.vocab_size, seed=config.seed)
    return create_dataloaders(
        tokens, block_size=config.block_size, batch_size=config.batch_size,
        val_split=config.val_split, num_workers=0,
    )


# ──────────────────────────────────────────────────────────────
# Discrete trainer
# ──────────────────────────────────────────────────────────────

class DiscreteTrainer:
    """Gradient-free local training loop over a stochastic ternary model.

    Forward passes run under ``torch.no_grad()``. Per-layer local deltas are
    accumulated into each ``StochasticTernaryLinear.accumulator`` (with a
    leaky decay to prevent unbounded growth) and bits are flipped when the
    accumulator exceeds the threshold. No global optimizer is used for the
    ternary weights.
    """

    def __init__(
        self,
        config: DiscreteConfig,
        train_loader,
        val_loader=None,
        model: StochasticTransformerModel | None = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model if model is not None else build_model_from_config(config)
        self.model.to(self.device)
        self.model.eval()

        # Calibration: the tied lm_head shares the nn.Embedding weight which is
        # initialised with std ~1, giving raw logits that explode the CE. We
        # scale logits down to O(1) inside the trainer (weights untouched).
        self._logit_scale = 1.0 / math.sqrt(config.hidden_dim)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Aux heads per block (rule 'p')
        self.aux_heads = nn.ModuleList()
        if config.rule == "p":
            for _ in range(config.num_layers - 1):
                self.aux_heads.append(
                    BottleneckHead(config.hidden_dim, config.aux_latent_dim, config.vocab_size)
                )
            self.aux_heads.to(self.device)
        self._aux_enabled = config.rule == "p"

        # Capture machinery
        self._captured_pos: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._captured_neg: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._captured_blocks: dict[str, torch.Tensor] = {}
        self._h_norm: torch.Tensor | None = None
        self._phase = "pos"
        self._linear_map: dict[str, StochasticTernaryLinear] = {}
        self._install_hooks()

        # Rule dispatch
        self._single_pass_rules = {
            "c": predictive_coding_delta,
            "h": hebbian_delta,
            "e": entropy_delta,
        }
        if config.rule not in ("c", "b", "p", "h", "e"):
            raise ValueError(f"Unknown rule '{config.rule}'")

        # Target modules for rule 'p' (block-output projections)
        self._target_mods: set[str] = set()
        if config.rule == "p":
            for k in range(config.num_layers):
                self._target_mods.add(f"layers.{k}.attn.o_proj")
                self._target_mods.add(f"layers.{k}.ffn.down_proj")

        # History
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.learning_rates: list[float] = []
        self.step = 0
        self._nan_step_count = 0

        torch.manual_seed(config.seed)

    # ── hooks ─────────────────────────────────────────────────
    def _install_hooks(self):
        handles = []
        for name, m in self.model.named_modules():
            if isinstance(m, StochasticTernaryLinear):
                self._linear_map[name] = m
                handles.append(m.register_forward_hook(
                    partial(self._hook_linear, name)))
            elif name.startswith("layers.") and m.__class__.__name__ in (
                    "StochasticTransformerBlock", "StochasticMLABlock"):
                handles.append(m.register_forward_hook(
                    partial(self._hook_block, name)))
        handles.append(self.model.norm.register_forward_hook(self._hook_norm))
        self._handles = handles

    def _hook_linear(self, name, module, inp, out):
        x = inp[0].detach().float()
        y = out.detach().float()
        if self._phase == "pos":
            self._captured_pos[name] = (x, y)
        else:
            self._captured_neg[name] = (x, y)

    def _hook_block(self, name, module, inp, out):
        if self._phase != "pos":
            return
        self._captured_blocks[name] = out[0].detach().float()

    def _hook_norm(self, module, inp, out):
        self._h_norm = out.detach().float()

    # ── helpers ───────────────────────────────────────────────
    def _make_negative(self, x: torch.Tensor) -> torch.Tensor:
        """Corrupt input for the Forward-Forward negative pass."""
        V = self.config.vocab_size
        mask = torch.rand(x.shape, device=x.device) < self.config.ff_corrupt
        rnd = torch.randint(0, V, x.shape, device=x.device)
        return torch.where(mask, rnd, x)

    @staticmethod
    def _ce(logits: torch.Tensor, targets: torch.Tensor, scale: float = 1.0) -> float:
        lf = (logits.float() * scale).reshape(-1, logits.size(-1))
        tf = targets.reshape(-1)
        valid = tf >= 0
        if not valid.any():
            return float("nan")
        return float(F.cross_entropy(lf[valid], tf[valid]))

    def _scaled(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self._logit_scale

    def _soft_target_grad(self, logits: torch.Tensor, targets: torch.Tensor,
                          ) -> torch.Tensor:
        """(softmax(scaled logits) - onehot(targets)) masked to valid targets."""
        V = logits.size(-1)
        probs = torch.softmax(self._scaled(logits), dim=-1)
        valid = (targets >= 0) & (targets < V)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(-1, targets.clamp(0, V - 1).unsqueeze(-1), 1.0)
        g = (probs - onehot) * valid.unsqueeze(-1)
        return g

    # ── local accumulation ───────────────────────────────────
    def _accumulate(self, module: StochasticTernaryLinear, delta: torch.Tensor):
        with torch.no_grad():
            module.accumulator.mul_(self.config.acc_decay)
            module.accumulator.add_(delta.to(module.accumulator.device))

    def _feed_local_deltas(self):
        if self.config.rule == "b":
            for name, m in self._linear_map.items():
                if name not in self._captured_pos or name not in self._captured_neg:
                    continue
                d = forward_forward_delta(*self._captured_pos[name], *self._captured_neg[name])
                self._accumulate(m, d)
            return
        for name, m in self._linear_map.items():
            if name not in self._captured_pos:
                continue
            if self.config.rule == "p" and name in self._target_mods:
                continue  # handled by target deltas
            fn = self._single_pass_rules.get(self.config.rule, predictive_coding_delta)
            d = fn(*self._captured_pos[name])
            if d is not None:
                self._accumulate(m, d)

    def _apply_target_deltas(self, logits: torch.Tensor, targets: torch.Tensor):
        """Rule 'p': drive block-output projections with real CE error."""
        L = self.config.num_layers
        E = self.model.token_embedding.weight  # (V, C)
        g = self._soft_target_grad(logits, targets)  # (B, T, V)
        if self._h_norm is None:
            return
        # Gradient of CE w.r.t. the (normed) final hidden state.
        g_h_top = torch.einsum("btv,vc->btc", g, E.float())  # (B, T, C)

        for k in range(L):
            if k == L - 1:
                g_h = g_h_top
            elif self._aux_enabled:
                block_name = f"layers.{k}"
                if block_name not in self._captured_blocks:
                    continue
                h_k = self._captured_blocks[block_name]
                g_h = self._aux_head_error(k, h_k, targets)
            else:
                # No aux head for this block: fall back to temporal PC.
                self._target_temporal_fallback(k)
                continue

            for proj in ("o_proj", "down_proj"):
                mod_name = f"layers.{k}.attn.{proj}"
                if mod_name in self._captured_pos and mod_name in self._linear_map:
                    self._accumulate_target(g_h, self._linear_map[mod_name],
                                            self._captured_pos[mod_name][0])
                mod_name = f"layers.{k}.ffn.{proj}"
                if mod_name in self._captured_pos and mod_name in self._linear_map:
                    self._accumulate_target(g_h, self._linear_map[mod_name],
                                            self._captured_pos[mod_name][0])

    def _target_temporal_fallback(self, k: int):
        """Give the block's output projections a temporal PC signal."""
        for proj in ("o_proj", "down_proj"):
            for kind in ("attn", "ffn"):
                mod_name = f"layers.{k}.{kind}.{proj}"
                if mod_name not in self._captured_pos:
                    continue
                m = self._linear_map[mod_name]
                d = predictive_coding_delta(*self._captured_pos[mod_name])
                if d is not None:
                    self._accumulate(m, d)

    def _accumulate_target(self, g_h: torch.Tensor, module: StochasticTernaryLinear,
                           x_cap: torch.Tensor):
        """delta = -sign(x^T g_h) for a block-output projection."""
        xf = x_cap.reshape(-1, x_cap.size(-1))
        gf = g_h.reshape(-1, g_h.size(-1))
        grad = gf.t() @ xf  # (out, in)
        self._accumulate(module, -torch.sign(grad))

    def _aux_head_error(self, k: int, h_k: torch.Tensor, targets: torch.Tensor,
                        ) -> torch.Tensor:
        """Local CE error for block k via its bottleneck head.

        Backward runs only through the small head (memory O(latent + vocab)
        per block), then head parameters are updated with a plain SGD step.
        """
        head = self.aux_heads[k]
        with torch.enable_grad():
            hd = h_k.detach().requires_grad_(True)
            logits_k = head(hd)
            loss_k = F.cross_entropy(
                logits_k.reshape(-1, logits_k.size(-1)),
                targets.reshape(-1), ignore_index=-1,
            )
            head.zero_grad()
            loss_k.backward()
            g_h = hd.grad.detach()
        with torch.no_grad():
            for p in head.parameters():
                if p.grad is not None:
                    p.data.sub_(self.config.aux_lr * p.grad)
        return g_h

    def _update_embedding(self, logits: torch.Tensor, h: torch.Tensor,
                          targets: torch.Tensor):
        """Plain local SGD for the tied FP32 embedding via top CE gradient.

        The gradient is clipped per row and decoupled weight decay is applied
        so the embedding norm stays bounded (otherwise logits explode and CE
        diverges — the runaway seen in uncalibrated runs).
        """
        if h is None:
            return
        g = self._soft_target_grad(logits, targets)  # (B, T, V)
        grad_E = torch.einsum("btv,btc->vc", g, h.float())  # (V, C)
        # Per-row gradient clipping (max L2 norm 1).
        row_norm = grad_E.norm(dim=1, keepdim=True).clamp(min=1e-8)
        grad_E = grad_E / row_norm.clamp(max=1.0)
        with torch.no_grad():
            E = self.model.token_embedding.weight
            E.data.sub_(self.config.lr_embedding * grad_E)
            # Decoupled weight decay keeps ||E|| bounded.
            E.data.mul_(1.0 - self.config.lr_embedding * self.config.wd_embedding)

    # ── training ─────────────────────────────────────────────
    def train_step(self, batch) -> float:
        x, yt = batch
        x = x.to(self.device)
        yt = yt.to(self.device)

        self._captured_pos.clear()
        self._captured_neg.clear()
        self._captured_blocks.clear()
        self._h_norm = None

        self._phase = "pos"
        with torch.no_grad():
            logits, _, _ = self.model(x, yt)
            if self.config.rule == "b":
                x_neg = self._make_negative(x)
                self._phase = "neg"
                self.model(x_neg, None)
                self._phase = "pos"

        with torch.no_grad():
            self._feed_local_deltas()
            if self.config.rule == "p":
                self._apply_target_deltas(logits, yt)
            if self.config.train_embedding:
                self._update_embedding(logits, self._h_norm, yt)

        return self._ce(logits, yt, scale=self._logit_scale)

    def apply_flips(self):
        """Decay threshold if configured, then flip bits on all layers."""
        if self.config.threshold_decay_to is not None:
            progress = min(self.step / self.config.max_steps, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            t = (self.config.threshold_decay_to
                 + (self.config.threshold - self.config.threshold_decay_to) * cosine)
            self.model.set_thresholds(t)
        self.model.apply_bit_flips()

    def train(self, resume_step: int = 0):
        self.step = resume_step
        steps_remaining = self.config.max_steps - resume_step
        self.model.eval()
        from tqdm import tqdm
        pbar = tqdm(total=steps_remaining, desc=f"Discrete [{self.config.rule}]", unit="step")
        step = resume_step
        while step < self.config.max_steps:
            acc_loss = 0.0
            acc_n = 0
            for batch in self.train_loader:
                if step >= self.config.max_steps:
                    break
                loss = self.train_step(batch)
                if math.isfinite(loss):
                    acc_loss += loss
                    acc_n += 1
                else:
                    self._nan_step_count += 1
                step += 1
                self.step = step
                if self.config.flip_every_n_steps > 0 and step % self.config.flip_every_n_steps == 0:
                    self.apply_flips()
                if step % self.config.log_interval == 0:
                    avg = acc_loss / max(acc_n, 1)
                    self.train_losses.append(avg)
                    pbar.set_postfix(loss=f"{avg:.4f}")
                    acc_loss = 0.0
                    acc_n = 0
                pbar.update(1)
                if self.config.eval_interval > 0 and step % self.config.eval_interval == 0:
                    self.val_losses.append(self.validate())
        pbar.close()

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        total = 0.0
        n = 0
        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= self.config.eval_steps:
                break
            x, yt = batch
            x = x.to(self.device)
            yt = yt.to(self.device)
            logits, _, _ = self.model(x, yt)
            lf = (logits.float() * self._logit_scale).reshape(-1, logits.size(-1))
            tf = yt.reshape(-1)
            valid = tf >= 0
            if not valid.any():
                continue
            total += float(F.cross_entropy(lf[valid], tf[valid])) * valid.sum().item()
            n += valid.sum().item()
        if n == 0:
            return float("nan")
        return total / n

    def save_checkpoint(self, step: int):
        import json
        from pathlib import Path
        ckpt = {
            "step": step,
            "config": self.config.__dict__,
            "model_state_dict": self.model.state_dict(),
            "aux_heads": self.aux_heads.state_dict() if self._aux_enabled else None,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        path = Path(self.config.save_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path / f"checkpoint_{step:06d}.pt")
        history = {"train_losses": self.train_losses, "val_losses": self.val_losses}
        with open(path / "training_history.json", "w") as f:
            json.dump(history, f)
        print(f"Checkpoint saved to {path / f'checkpoint_{step:06d}.pt'}")
