"""Training pipeline for Ternary LLM.

Implements:
- AdamW optimizer with ternary-specific settings
- Cosine learning rate scheduler with warmup
- Gradient accumulation
- Checkpointing and logging
- Validation loop
"""
__all__ = [
    "TrainingConfig", "TernaryTrainer", "DMLAdamW",
]

import os
import json
import time
import math

_HAS_PSUTIL = False
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    pass
from pathlib import Path
from dataclasses import dataclass, field


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class DMLAdamW(torch.optim.Optimizer):
    """AdamW variant that avoids lerp_ (CPU fallback on DML).

    Uses only mul_ + add_ + addcdiv_ which are DML-native.
    Falls back to pow(2) + add_ if addcmul_ fails.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.1, amsgrad=False, *, foreach=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("DMLAdamW does not support sparse gradients")

                state = self.state[p]

                # Initialise state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Decoupled weight decay
                p.mul_(1 - lr * weight_decay)

                # Biased first moment update (avoid lerp_)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Biased second moment update (avoid lerp_)
                exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1 - beta2)

                # Bias correction
                step = state["step"]
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


@dataclass
class TrainingConfig:
    """Training configuration for Ternary LLM."""
    # Model (overridden by data pipeline from tokenizer metadata)
    vocab_size: int = 50257  # GPT-2 tokenizer (default)
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 512
    max_seq_len: int = 512

    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 10000
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # AdamW betas (ternary-specific: higher beta2)
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Validation
    eval_interval: int = 500
    eval_steps: int = 100

    # Checkpointing
    save_interval: int = 500
    save_dir: str = "checkpoints"

    # Logging
    log_interval: int = 10

    # Device
    device: str = ""
    dtype: str = "float32"
    hybrid_optimizer: bool = False  # Model on GPU, optimizer on CPU (avoids DML fallbacks)

    # Mode
    mode: str = "ste"  # "ste" or "stochastic"

    # Debug
    debug: bool = False  # Print MEM/TIME diagnostics

    # Stochastic Bit-Flip
    flip_every_n_steps: int = 5  # check threshold & flip bits every N optimizer steps
    threshold: float = 20.0  # Initial flip threshold (used for decay base)
    threshold_decay_to: float | None = None  # Decay threshold from initial to this by end (None = constant)

    # Quantization (STE)
    ternary_scale: float = 1.0  # Δ = scale x mean(|W|), lower -> more {-1,+1}, higher -> more 0
    # (default 1.0 since Exp 9: lower CE + fewer ±2 outliers vs old 0.7)
    per_channel: bool = False    # Per-channel vs per-tensor threshold

    # Exp 8: soft-to-hard quantization schedule (STE mode)
    soft_quant: bool = False             # enable sigmoid-surrogate warmup
    soft_quant_gamma_init: float = 2.0   # starting surrogate temperature
    soft_quant_gamma_max: float = 50.0   # final temperature (then hard STE)
    soft_quant_steps: int = 0            # warmup length (0 = 25% of max_steps)

    # STE robustness (rank-collapse fixes, finding #11)
    init_mode: str = "kaiming"   # "kaiming" or "balanced" (33/33/33 ternary init)
    ortho_reg: float = 0.0       # orthogonalization penalty weight on latent rows
    rank_monitor_interval: int = 500  # unique-rows report cadence (0 = off)
    rank_halt: bool = False      # halt training when any matrix collapses (unique_rows <= rows/4)
    save_best: bool = False      # keep best-by-val checkpoint (checkpoint_best.pt)

    # Data
    data_dir: str = "data"
    block_size: int = 128
    val_split: float = 0.05

    # ERC (Echo Residual Committer): two-timescale memory for the ternary core.
    # W_eff = Ternary(W_core) + R: the fast residual R absorbs new-domain
    # surprise at a high LR; periodic threshold commits (|R| >= Delta/2) carry
    # its energy into the slow latent core without overwriting it directly.
    erc: bool = False
    erc_lr: float = 0.01          # fast residual LR (core keeps --lr)
    erc_decay: float = 1.0        # leaky EMA on R per step (<1 fades old echoes)
    erc_commit_interval: int = 10 # commit R -> core every N optimizer steps
    erc_residual_dtype: str = "fp32"  # "fp32" or "fp16" (short-term memory)
    erc_freeze_core: bool = False  # freeze the latent core entirely (R-only learning)

    # Held-out slice evals (transfer tests): slice = old domain, domain = new
    eval_slice_path: str | None = None
    domain_eval_path: str | None = None
    eval_positions: int = 20000


class CosineScheduler:
    """Cosine learning rate scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr_scale = self._get_lr_scale()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * lr_scale

    def _get_lr_scale(self) -> float:
        step = self.step_count
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        decay = 0.5 * (1 - self.min_lr / self.base_lrs[0]) * (1 + math.cos(math.pi * progress))
        return self.min_lr / self.base_lrs[0] + decay


class TernaryTrainer:
    """Complete training loop for Ternary LLM."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.config = config

        # Resolve device
        if config.device == "cpu":
            self.device = torch.device("cpu")
        elif config.device:
            if config.device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif config.device == "directml":
                try:
                    import torch_directml
                    self.device = torch_directml.device()
                except ImportError:
                    print("WARNING: torch_directml not found, falling back to CPU")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(config.device)
        else:
            # Auto-detect: CUDA > DirectML > CPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                try:
                    import torch_directml
                    self.device = torch_directml.device()
                except Exception:
                    self.device = torch.device("cpu")
        print(f"Device: {self.device}")

        # C++ extension status
        from .quantization import _has_cpp
        if _has_cpp:
            print("C++ SIMD unpack: enabled (~1.2s/500M weights)")
        else:
            print("C++ SIMD unpack: disabled (build with: python build_cpp.py)")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ERC: convert every TernaryLinear into an ERCLinear (state-dict keys
        # unchanged) so the residual path exists before param groups are split.
        if config.erc:
            from .erc import enable_erc
            n_erc = enable_erc(self.model, residual_dtype=config.erc_residual_dtype)
            print(f"ERC: enabled on {n_erc:,} layers (residual dtype {config.erc_residual_dtype})")
            if n_erc == 0:
                print("WARNING: --erc set but no TernaryLinear found in the model")

        # ERC: split params into (slow core | fast residual) groups.
        # The residual gets its own LR and zero weight decay so it can move
        # fast without being dragged to 0 by decoupled decay.
        param_groups = self._build_param_groups()
        if config.erc and config.erc_freeze_core:
            for p in param_groups[0]["params"]:
                p.requires_grad_(False)
            print(f"ERC: latent core FROZEN (residual-only learning, R at lr {config.erc_lr})")

        # Hybrid optimizer: model on GPU, optimizer states on CPU
        self.hybrid = False
        if config.hybrid_optimizer and self.device != torch.device("cpu"):
            self.hybrid = True
            # Create CPU parameter clones for optimizer (deduplicate tied weights)
            self.cpu_params = []
            self._cpu_clone_of: dict[int, nn.Parameter] = {}
            for group in param_groups:
                cp_group = []
                for p in group["params"]:
                    cp = nn.Parameter(p.data.cpu().clone(), requires_grad=p.requires_grad)
                    cp_group.append(cp)
                    self._cpu_clone_of[id(p)] = cp
                self.cpu_params.append(cp_group)
            self.optimizer = torch.optim.AdamW(
                [dict(g, params=cp_group) for g, cp_group in zip(param_groups, self.cpu_params)],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                foreach=False,
            )
            print(f"Hybrid mode: model on {self.device}, optimizer on CPU")
        else:
            # Standard: optimizer on same device as model
            is_dml = self.device.type == "privateuseone"
            opt_cls = DMLAdamW if is_dml else torch.optim.AdamW
            self.optimizer = opt_cls(
                [dict(g, params=list(g["params"])) for g in param_groups],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
            )

        # ERC bookkeeping
        self.erc_total_commits = 0
        self.erc_committed_frac = 0.0
        self.slice_ces = []
        self.domain_ces = []
        self.slice_core_ces = []   # ERC: same slice with R zeroed (core-only)
        self.domain_core_ces = []

        # LR Scheduler
        self.scheduler = CosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr=config.min_lr,
        )

        # Mixed precision
        self.activation_dtype = None  # explicit dtype override
        self.autocast_dtype = None    # autocast dtype (unused, kept for compat)
        self.scaler = None
        if config.dtype in ("float16", "bfloat16"):
            bf16_ok = (
            config.dtype == "bfloat16"
            and self.device.type == "cuda"
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
            self.activation_dtype = torch.bfloat16 if bf16_ok else torch.float16
            if self.activation_dtype == torch.float16 and self.device.type == "cuda" \
               and config.mode != "stochastic":
                self.scaler = torch.amp.GradScaler("cuda")
            if self.activation_dtype == torch.float16 and self.device.type == "cuda" \
               and config.mode == "stochastic":
                print("  (GradScaler disabled for stochastic mode)")
            print(f"Activations: {str(self.activation_dtype).split('.')[-1]}")
        else:
            print(f"Full precision: float32")

        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

        # Logging
        self.train_losses = []
        self.train_log_steps = []  # step number at which each loss was logged
        self.val_losses = []
        self.learning_rates = []
        self._nan_step_count = 0

        # Timer accumulators (NaN-safe: always init here)
        self.fwd_time = 0.0
        self.bwd_time = 0.0
        self.micro_steps = 0
        self._has_grads = False

        # Best-checkpoint tracking
        self.best_val_loss = float("inf")
        self.best_step = -1
        self._rank_halted = False

        # Initial memory snapshot
        self.log_mem("init")

    def log_mem(self, tag: str):
        if not _HAS_PSUTIL:
            return
        ram_gb = psutil.Process().memory_info().rss / 1024**3
        msg = f"  Memory ({tag}): RAM={ram_gb:.1f}GB"
        if self.device.type == "cuda":
            vram_gb = torch.cuda.memory_allocated() / 1024**3
            msg += f" VRAM={vram_gb:.1f}GB"
        if self.config.debug:
            print(msg, flush=True)

    def _clear_cache(self):
        """Free cached memory from CUDA allocator; best-effort on DML."""
        import gc
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """Single training step with gradient accumulation."""
        import time
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        t0 = time.perf_counter()
        _, loss, _ = self.model(x, y, activation_dtype=self.activation_dtype)
        t1 = time.perf_counter()

        if self.config.ortho_reg > 0:
            loss = loss + self.config.ortho_reg * self._ortho_penalty()
        raw_loss = loss.item()
        if not math.isfinite(raw_loss):
            self.model.zero_grad(set_to_none=True)
            return raw_loss
        loss = loss / self.config.gradient_accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        t2 = time.perf_counter()
        self._clear_cache()

        self.fwd_time += t1 - t0
        self.bwd_time += t2 - t1
        self.micro_steps += 1
        self._has_grads = True

        return raw_loss

    def _build_param_groups(self) -> list[dict]:
        """Split model params into (core | ERC residual) optimizer groups.

        Core group: all params except residual tensors (lr = base, wd).
        Residual group: every ERCLinear.residual (lr = erc_lr, wd = 0).
        """
        from .erc import ERCLinear

        residual_ids = set()
        residual_params = []
        for m in self.model.modules():
            if isinstance(m, ERCLinear):
                residual_ids.add(id(m.residual))
                residual_params.append(m.residual)
        core_params = []
        seen = set()
        for p in self.model.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) not in residual_ids:
                core_params.append(p)
        if self.config.erc:
            if not residual_params:
                print("WARNING: --erc set but no ERCLinear found in the model")
            return [
                {"params": core_params, "lr": self.config.learning_rate,
                 "weight_decay": self.config.weight_decay},
                {"params": residual_params, "lr": self.config.erc_lr,
                 "weight_decay": 0.0},
            ]
        return [{"params": core_params, "lr": self.config.learning_rate,
                 "weight_decay": self.config.weight_decay}]

    def _unique_params(self):
        """Yield unique model parameters (skip tied weight duplicates)."""
        seen = set()
        for p in self.model.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                yield p

    def _ortho_penalty(self, margin: float = 0.3, k: int = 64) -> torch.Tensor:
        """L2 orthogonality penalty on sampled latent-weight row pairs.

        Mean over (|cosine| - margin)+ for random row pairs of each ternary
        matrix. Counteracts the rank-1 collapse of finding #11.
        """
        total = None
        count = 0
        for name, p in self.model.named_parameters():
            if "latent_weights" not in name or p.ndim != 2:
                continue
            out, inn = p.shape
            kk = min(out, k)
            idx = torch.randperm(out, device=p.device)[:kk]
            rows = p[idx]
            rows = rows / (rows.norm(dim=1, keepdim=True) + 1e-8)
            g = rows @ rows.T
            off = torch.triu(g, diagonal=1)
            term = (off.abs().clamp_min(margin) - margin).mean()
            total = term if total is None else total + term
            count += 1
        return total / max(count, 1) if total is not None else torch.zeros((), device=self.device)

    @torch.no_grad()
    def report_rank(self) -> bool:
        """Count unique ternary rows per matrix; warn on collapse.

        Collapse threshold: unique_rows <= rows/4 (finding #11: base checkpoint
        had unique_rows=1). Returns False if halt requested and any matrix is
        collapsed.
        """
        from .quantization import TernaryQuantizer, unpack_ternary_tensor

        worst = []
        halted = False
        for name, mod in self.model.named_modules():
            w = None
            if hasattr(mod, "latent_weights") and mod.latent_weights is not None:
                w = mod.latent_weights
            elif hasattr(mod, "packed_weights") and mod.packed_weights is not None:
                w = unpack_ternary_tensor(mod.packed_weights, (mod.out_features, mod.in_features))
            if w is None:
                continue
            if hasattr(mod, "latent_weights"):
                if getattr(mod, "per_channel", False):
                    delta = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-6) \
                        * getattr(mod, "ternary_scale", 1.0)
                    wt = (w / delta).clamp(-1, 1).round()
                else:
                    wt = TernaryQuantizer.apply(w.data)
            else:
                wt = w
            wt = wt.to("cpu")
            rows = wt.shape[0]
            n_unique = int(torch.unique(wt, dim=0).shape[0])
            frac = n_unique / rows
            worst.append((frac, name, n_unique, rows))
            if frac < 0.25:
                halted = True
        worst.sort()
        print("[Rank] " + " | ".join(f"{n}:{u}/{r}" for _, n, u, r in worst))
        if halted:
            print("[Rank] COLLAPSE detected" + (" — halting training" if self.config.rank_halt else ""))
        return not halted

    def _hybrid_sync_gradients(self):
        """Copy gradients from GPU model to CPU params."""
        for gp in self._unique_params():
            cp = self._cpu_clone_of.get(id(gp))
            if cp is None:
                continue
            if gp.grad is not None:
                cp.grad = gp.grad.cpu()
            else:
                cp.grad = None

    def _hybrid_sync_weights(self):
        """Copy updated weights from CPU params back to GPU model."""
        for gp in self._unique_params():
            cp = self._cpu_clone_of.get(id(gp))
            if cp is None:
                continue
            gp.data.copy_(cp.data.to(gp.device))

    def train_epoch(self, step_start: int) -> int:
        """Train for one epoch (or until data exhausted)."""
        self.model.train()
        total_loss = 0.0
        step = step_start
        micro_count = 0
        steps_remaining = self.config.max_steps - step_start
        pbar = tqdm(total=steps_remaining, desc=f"Training", unit="step")

        for batch in self.train_loader:
            if step >= self.config.max_steps:
                break

            loss = self.train_step(batch)
            if math.isfinite(loss):
                total_loss += loss
            else:
                self._nan_step_count += 1
            micro_count += 1

            if micro_count > 0 and micro_count % self.config.gradient_accumulation_steps == 0:
                # Timing breakdown
                if self.fwd_time > 0:
                    avg_fwd = self.fwd_time / self.micro_steps
                    avg_bwd = self.bwd_time / self.micro_steps
                    if self.config.debug:
                        tqdm.write(
                            f"  Timing: fwd={avg_fwd:.1f}s | bwd={avg_bwd:.1f}s | total={avg_fwd+avg_bwd:.1f}s"
                        )
                        # Per-layer timing (last micro-batch)
                        if hasattr(self.model, '_layer_times') and self.model._layer_times:
                            lt = self.model._layer_times
                            tqdm.write(f"  Layer timing: " + " | ".join(
                                f"L{i}={lt[i]:.3f}s" for i in range(len(lt))
                            ))
                self.fwd_time = 0.0
                self.bwd_time = 0.0
                self.micro_steps = 0

                # Skip optimizer step if all micro-batches were NaN
                opt_time = 0.0
                if self._has_grads:
                    self._has_grads = False
                    opt_t0 = time.perf_counter()
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._unique_params(), self.config.grad_clip
                    )

                    # Optimizer step
                    if self.hybrid:
                        self._hybrid_sync_gradients()
                        if self.scaler is not None:
                            self.scaler.step(self.optimizer)
                        else:
                            self.optimizer.step()
                        self._hybrid_sync_weights()
                    else:
                        if self.scaler is not None:
                            self.scaler.step(self.optimizer)
                        else:
                            self.optimizer.step()
                    if self.scaler is not None:
                        self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.hybrid:
                        self.model.zero_grad(set_to_none=True)
                    opt_time = time.perf_counter() - opt_t0

                    # Free cached allocator memory after optimizer step
                    self._clear_cache()

                # ERC two-timescale loop: leaky decay of the fast residual +
                # periodic threshold commits into the slow ternary core.
                if self.config.erc and step > 0:
                    from .erc import commit_erc, decay_erc
                    if self.config.erc_decay < 1.0:
                        decay_erc(self.model, self.config.erc_decay)
                    if self.config.erc_commit_interval > 0 and step % self.config.erc_commit_interval == 0:
                        n_committed, n_positions = commit_erc(self.model)
                        self.erc_total_commits += n_committed
                        self.erc_committed_frac = n_committed / max(1, n_positions)
                        if n_committed > 0:
                            tqdm.write(
                                f"  ERC commit: {n_committed:,}/{n_positions:,} "
                                f"positions ({n_committed/max(1,n_positions)*100:.3f}%) "
                                f"carried into core")

                # Stochastic Bit-Flip: apply accumulated flips every N steps
                if (self.config.mode == "stochastic"
                    and step > 0
                    and step % self.config.flip_every_n_steps == 0):
                    if self.config.debug:
                        tqdm.write(f"  Optimizer step: {opt_time:.1f}s")
                    # Apply threshold decay if configured
                    if self.config.threshold_decay_to is not None:
                        progress = min(step / self.config.max_steps, 1.0)
                        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                        t = self.config.threshold_decay_to + (self.config.threshold - self.config.threshold_decay_to) * cosine
                        self.model.set_thresholds(t)
                    self.log_mem("before apply_bit_flips")
                    flip_t0 = time.perf_counter()
                    self.model.apply_bit_flips()
                    flip_time = time.perf_counter() - flip_t0
                    self.log_mem("after apply_bit_flips")
                    if self.config.debug:
                        tqdm.write(f"  Bit-flip time: {flip_time:.1f}s")
                elif self.config.debug:
                    tqdm.write(f"  Optimizer step: {opt_time:.1f}s")

                # LR scheduler
                self.scheduler.step()
                step += 1
                pbar.update(1)

                # Exp 8: soft-to-hard gamma schedule (log-linear warmup then
                # switch to hard STE at the end of the warmup window)
                if self.config.soft_quant and self.config.mode == "ste":
                    warm = self.config.soft_quant_steps
                    if warm <= 0:
                        warm = max(1, int(self.config.max_steps * 0.25))
                    if step < warm:
                        p = max(0.0, step / warm)
                        g_init = self.config.soft_quant_gamma_init
                        g_max = self.config.soft_quant_gamma_max
                        gamma = g_init * (g_max / g_init) ** p  # log-linear
                    else:
                        gamma = None  # hard round + STE (matches baseline)
                    for m in self.model.modules():
                        if hasattr(m, "set_soft_gamma"):
                            m.set_soft_gamma(gamma)

                # Logging
                if step % self.config.log_interval == 0:
                    n_micro = self.config.gradient_accumulation_steps * self.config.log_interval
                    n_valid = n_micro - self._nan_step_count
                    avg_loss = total_loss / max(n_valid, 1) if n_valid > 0 else float("nan")
                    lr = self.optimizer.param_groups[0]["lr"]
                    if math.isfinite(avg_loss):
                        self.train_losses.append(avg_loss)
                        self.train_log_steps.append(step)
                    self.learning_rates.append(lr)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                    total_loss = 0.0
                    self._nan_step_count = 0

                # Validation
                if step % self.config.eval_interval == 0:
                    val_loss = self.validate()
                    self.val_losses.append(val_loss)
                    if self.config.save_best and math.isfinite(val_loss) and val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_step = step
                        self.save_checkpoint(step, best=True)
                    # Held-out slice evals (transfer tests: old-domain slice +
                    # new-domain slice, mirroring train_baseline_backprop.py)
                    if self.config.eval_slice_path:
                        self.eval_slices(step)

                # Rank monitor (unique ternary rows per matrix)
                if (self.config.rank_monitor_interval > 0
                        and step % self.config.rank_monitor_interval == 0
                        and not self.report_rank()):
                    self._rank_halted = True
                    break

                # Checkpoint
                if step % self.config.save_interval == 0:
                    self.save_checkpoint(step)

        return step

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= self.config.eval_steps:
                break

            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            _, loss, _ = self.model(x, y)

            total_loss += loss.item()
            num_batches += 1

        avg_val_loss = total_loss / max(num_batches, 1)
        print(f"Validation: step {self.scheduler.step_count}  loss={avg_val_loss:.4f}")
        return avg_val_loss

    @torch.no_grad()
    def eval_slices(self, step: int) -> None:
        """Held-out slice CE on the old domain (slice) and new domain (domain).

        Mirrors train_baseline_backprop.py: raw uint16 .bin token slices,
        causal full-context within block_size chunks. Writes eval_history.json.

        Under ERC each metric is measured twice: with the residual R active
        (behavior) and with R temporarily zeroed (core-only). Core-only CE
        isolates genuine forgetting of the ternary core from the residual's
        fast behavioral shift.
        """
        import numpy as np

        def _ce(path: str, zero_r: bool = False) -> float:
            data = np.fromfile(path, dtype=np.uint16)
            limit = min(len(data) - 1, self.config.eval_positions)
            total = 0.0
            n = 0
            bs = self.config.block_size
            saved = []
            if zero_r:
                from .erc import ERCLinear
                for m in self.model.modules():
                    if isinstance(m, ERCLinear):
                        saved.append((m, m.residual.detach().clone()))
                        with torch.no_grad():
                            m.residual.mul_(0.0)
            try:
                for start in range(0, limit, bs):
                    end = min(limit, start + bs)
                    x = torch.tensor(data[start:end].astype(np.int64)[None], dtype=torch.long,
                                     device=self.device)
                    y = torch.tensor(data[start + 1:end + 1].astype(np.int64)[None], dtype=torch.long,
                                     device=self.device)
                    logits, _, _ = self.model(x, None)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="mean")
                    total += loss.item() * (end - start)
                    n += end - start
                return total / max(n, 1)
            finally:
                for m, r in saved:
                    with torch.no_grad():
                        m.residual.copy_(r)

        self.model.eval()
        line = f"[step {step}]"
        if self.config.eval_slice_path:
            ce_slice = _ce(self.config.eval_slice_path)
            self.slice_ces.append([int(step), float(ce_slice)])
            line += f" sliceCE(old)={ce_slice:.4f}"
            if self.config.erc:
                ce_core = _ce(self.config.eval_slice_path, zero_r=True)
                self.slice_core_ces.append([int(step), float(ce_core)])
                line += f" sliceCEcore={ce_core:.4f}"
        if self.config.domain_eval_path:
            ce_dom = _ce(self.config.domain_eval_path)
            self.domain_ces.append([int(step), float(ce_dom)])
            line += f" domainCE(new)={ce_dom:.4f}"
            if self.config.erc:
                ce_core = _ce(self.config.domain_eval_path, zero_r=True)
                self.domain_core_ces.append([int(step), float(ce_core)])
                line += f" domainCEcore={ce_core:.4f}"
        if line != f"[step {step}]":
            print(line)
            self._write_eval_history()

    def _write_eval_history(self) -> None:
        if not (self.slice_ces or self.domain_ces):
            return
        history = {
            "slice_ces": self.slice_ces,
            "domain_ces": self.domain_ces,
            "slice_core_ces": self.slice_core_ces,
            "domain_core_ces": self.domain_core_ces,
        }
        path = Path(self.config.save_dir) / "eval_history.json"
        with open(path, "w") as f:
            json.dump(history, f)


    def _quantize_optimizer_to_fp16(self, opt_state: dict) -> dict:
        """Convert optimizer state tensors from FP32 to FP16 for smaller checkpoints."""
        quantized = {}
        for k, v in opt_state.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                quantized[k] = v.half()
            elif isinstance(v, dict):
                quantized[k] = self._quantize_optimizer_to_fp16(v)
            else:
                quantized[k] = v
        return quantized

    def _dequantize_optimizer_to_fp32(self, opt_state: dict) -> dict:
        """Convert optimizer state tensors from FP16 back to FP32."""
        dequantized = {}
        for k, v in opt_state.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                dequantized[k] = v.float()
            elif isinstance(v, dict):
                dequantized[k] = self._dequantize_optimizer_to_fp32(v)
            else:
                dequantized[k] = v
        return dequantized

    def save_checkpoint(self, step: int, best: bool = False):
        """Save model checkpoint and keep only the 3 most recent.

        STE mode: latent weights -> FP16, optimizer states -> FP16.
        Stochastic mode: packed ternary + accumulators saved directly.
        best=True writes checkpoint_best.pt (kept until a better one).
        """
        is_stochastic = self.config.mode == "stochastic"

        if is_stochastic:
            state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                if "accumulator" in k and v.is_floating_point():
                    state_dict[k] = v.half()
        else:
            # STE: convert latent to FP16 on device (fast), keep rest as-is
            state_dict = {}
            for k, v in self.model.state_dict().items():
                if "latent_weights" in k:
                    state_dict[k] = v.half()
                else:
                    state_dict[k] = v

        # ERC: full-carry the residual into the latent core BEFORE saving, then
        # drop the residual keys. The saved checkpoint is therefore a plain
        # ternary model — ERC-free. R is committed, not lost.
        if self.config.erc and not is_stochastic:
            from ternary_llm.erc import commit_erc_state_dict
            n = commit_erc_state_dict(state_dict, self.config.__dict__)
            self.erc_total_commits += n
            print(f"ERC full-carry on save: {n:,} residual positions baked into core")

        # Quantize optimizer states to FP16
        opt_state = self.optimizer.state_dict()
        opt_state["state"] = {k: self._quantize_optimizer_to_fp16(v) for k, v in opt_state["state"].items()}

        checkpoint = {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": opt_state,
            "config": self.config.__dict__,
            "train_losses": self.train_losses,
            "train_log_steps": self.train_log_steps,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "optimizer_fp16": True,
            "mode": self.config.mode,
        }
        if not is_stochastic:
            checkpoint["latent_fp16"] = True

        # Move all tensors to CPU before saving (torch.save is 8x faster on CPU tensors)
        def _to_cpu(obj):
            if isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, torch.Tensor):
                return obj.cpu()
            return obj

        path = Path(self.config.save_dir) / ("checkpoint_best.pt" if best else f"checkpoint_{step:06d}.pt")
        torch.save(_to_cpu(checkpoint), path)
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"Checkpoint saved to {path} ({size_mb:.0f} MB)" + (f" [best val {self.best_val_loss:.4f}]" if best else ""))

        if best:
            return

        # Cleanup: keep only 3 most recent checkpoints
        checkpoints = sorted(Path(self.config.save_dir).glob("checkpoint_*.pt"))
        checkpoints = [c for c in checkpoints if "best" not in c.name]
        while len(checkpoints) > 3:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"Removed old checkpoint: {oldest.name}")

        # Save training history at each checkpoint (crash-safe)
        history = {
            "train_losses": self.train_losses,
            "train_log_steps": self.train_log_steps,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
        }
        history_path = Path(self.config.save_dir) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)

        self._write_eval_history()

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Handles FP16 latent weights (STE), FP16 accumulators (stochastic),
        packed ternary (old), FP16 optimizer, and old FP32 formats.
        """
        from .quantization import unpack_ternary

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raw_state = checkpoint["model_state_dict"]
        ckpt_mode = checkpoint.get("mode", "ste")

        if ckpt_mode == "stochastic":
            # Stochastic: FP16 accumulators -> FP32, packed weights stay uint8
            state_dict = {}
            for k, v in raw_state.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                    state_dict[k] = v.float()
                else:
                    state_dict[k] = v
        elif checkpoint.get("latent_fp16", False):
            # STE new format: FP16 latent weights -> FP32
            state_dict = {}
            for k, v in raw_state.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                    state_dict[k] = v.float()
                else:
                    state_dict[k] = v
        # Old format: packed ternary -> unpack (legacy compatibility)
        elif checkpoint.get("ternary_packed", False):
            state_dict = {}
            for k, v in raw_state.items():
                if isinstance(v, dict) and "packed" in v:
                    state_dict[k] = unpack_ternary(v["packed"], tuple(v["shape"]))
                else:
                    state_dict[k] = v
        else:
            state_dict = raw_state

        # strict=False: v7 checkpoints carry outlier_signs buffers that older
        # checkpoints lack (defaults to empty = no outliers)
        self.model.load_state_dict(state_dict, strict=False)

        # Dequantize optimizer states from FP16 to FP32 if needed
        opt_state = checkpoint["optimizer_state_dict"]
        if checkpoint.get("optimizer_fp16", False):
            opt_state["state"] = {k: self._dequantize_optimizer_to_fp32(v) for k, v in opt_state["state"].items()}

        if self.hybrid:
            # Reconstruct cpu_params from model weights (avoids saving duplicate)
            for gp in self._unique_params():
                cp = self._cpu_clone_of.get(id(gp))
                if cp is not None:
                    cp.data.copy_(gp.data.cpu())
            # Load optimizer state into cpu_params (group mismatch = fresh restart)
            try:
                self.optimizer.load_state_dict(opt_state)
            except (RuntimeError, ValueError) as e:
                print(f"WARNING: optimizer state not loaded ({e}); restarting optimizer")
            # Sync to GPU model
            self._hybrid_sync_weights()
        else:
            # Group mismatch (ERC on/off across resume, e.g. Phase-1 checkpoint
            # trained without residuals into an ERC model): restart optimizer.
            try:
                self.optimizer.load_state_dict(opt_state)
            except (RuntimeError, ValueError) as e:
                print(f"WARNING: optimizer state not loaded ({e}); restarting optimizer")
            # Move optimizer state to device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.model.to(self.device)
        # Restore loss history
        self.train_losses = checkpoint.get("train_losses", [])
        self.train_log_steps = checkpoint.get("train_log_steps", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.learning_rates = checkpoint.get("learning_rates", [])
        if self.config.save_best and self.val_losses:
            self.best_val_loss = min(self.val_losses)
        print(f"Checkpoint loaded from {path} (step {checkpoint['step']})")
        print(f"Restored {len(self.train_losses)} train losses, {len(self.val_losses)} val losses")
        return checkpoint["step"]

    def train(self, resume_step: int = 0):
        """Main training loop."""
        print("\nTernary LLM Training")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.config.dtype}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Vocab size: {self.config.vocab_size}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Grad accum: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"LR: {self.config.learning_rate} -> {self.config.min_lr}")
        self._nan_step_count = 0

        # Validate first batch
        try:
            sample_batch = next(iter(self.train_loader))
            x, y = sample_batch
            max_id = x.max().item()
            if max_id >= self.config.vocab_size:
                print(f"WARNING: Data contains token ID {max_id} >= vocab_size {self.config.vocab_size}!")
                print(f"Clamping will be applied. Re-prepare data with current tokenizer to fix.")
            else:
                print(f"Data OK: max token ID = {max_id} < vocab_size = {self.config.vocab_size}")
        except Exception:
            pass

        step = resume_step
        start_time = time.time()

        # Fast-forward scheduler
        self.scheduler.step_count = resume_step
        lr_scale = self.scheduler._get_lr_scale()
        for group, base_lr in zip(self.scheduler.optimizer.param_groups, self.scheduler.base_lrs):
            group["lr"] = base_lr * lr_scale
        if resume_step > 0:
            print(f"Resumed from step {resume_step}, LR = {self.scheduler.optimizer.param_groups[0]['lr']:.6f}")

        while step < self.config.max_steps:
            new_step = self.train_epoch(step)
            if new_step == step:
                print("WARNING: No training progress (loader empty or all batches dropped). "
                      "Check batch size vs dataset size. Stopping.")
                break
            step = new_step

            if self._rank_halted:
                print(f"\nHalted by rank monitor at step {step}")
                break

            if step < self.config.max_steps:
                # Save checkpoint at intervals
                if step % self.config.save_interval == 0:
                    self.save_checkpoint(step)

        # Final save
        if not self._rank_halted:
            self.save_checkpoint(step)
        elif self.config.save_best and self.best_step > 0:
            print(f"Best checkpoint kept: step {self.best_step}, val {self.best_val_loss:.4f}")

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
        if self.train_losses:
            print(f"Final train loss: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            print(f"Final val loss: {self.val_losses[-1]:.4f}")

        # Save training history
        history = {
            "train_losses": self.train_losses,
            "train_log_steps": self.train_log_steps,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
        }
        history_path = Path(self.config.save_dir) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)
