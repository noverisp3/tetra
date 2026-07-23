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
    device: str = "cpu"
    dtype: str = "float32"
    hybrid_optimizer: bool = False  # Model on GPU, optimizer on CPU (avoids DML fallbacks)

    # Mode
    mode: str = "ste"  # "ste" or "stochastic"

    # Debug
    debug: bool = False  # Print MEM/TIME diagnostics

    # Stochastic Bit-Flip
    flip_every_n_steps: int = 5  # check threshold & flip bits every N optimizer steps

    # Quantization (STE)
    ternary_scale: float = 0.7  # Δ = scale x mean(|W|), lower -> more {-1,+1}, higher -> more 0
    per_channel: bool = False    # Per-channel vs per-tensor threshold

    # Data
    data_dir: str = "data"
    block_size: int = 128
    val_split: float = 0.05


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
        if config.device and config.device != "cpu":
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

        # Hybrid optimizer: model on GPU, optimizer states on CPU
        self.hybrid = False
        if config.hybrid_optimizer and self.device != torch.device("cpu"):
            self.hybrid = True
            # Create CPU parameter clones for optimizer (deduplicate tied weights)
            seen = set()
            self.cpu_params = []
            for p in model.parameters():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                cp = nn.Parameter(p.data.cpu().clone(), requires_grad=p.requires_grad)
                self.cpu_params.append(cp)
            self.optimizer = torch.optim.AdamW(
                self.cpu_params,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                foreach=False,
            )
            print(f"Hybrid mode: model on {self.device}, optimizer on CPU")
        else:
            # Standard: optimizer on same device as model
            is_dml = self.device.type == "privateuseone"
            opt_cls = DMLAdamW if is_dml else torch.optim.AdamW
            self.optimizer = opt_cls(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
            )

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
            if self.activation_dtype == torch.float16 and self.device.type == "cuda":
                self.scaler = torch.amp.GradScaler("cuda")
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

    def _unique_params(self):
        """Yield unique model parameters (skip tied weight duplicates)."""
        seen = set()
        for p in self.model.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                yield p

    def _hybrid_sync_gradients(self):
        """Copy gradients from GPU model to CPU params."""
        for gp, cp in zip(self._unique_params(), self.cpu_params):
            if gp.grad is not None:
                cp.grad = gp.grad.cpu()
            else:
                cp.grad = None

    def _hybrid_sync_weights(self):
        """Copy updated weights from CPU params back to GPU model."""
        for gp, cp in zip(self._unique_params(), self.cpu_params):
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

                # Stochastic Bit-Flip: apply accumulated flips every N steps
                if (self.config.mode == "stochastic"
                    and step > 0
                    and step % self.config.flip_every_n_steps == 0):
                    if self.config.debug:
                        tqdm.write(f"  Optimizer step: {opt_time:.1f}s")
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

    def save_checkpoint(self, step: int):
        """Save model checkpoint and keep only the 3 most recent.

        STE mode: latent weights -> FP16, optimizer states -> FP16.
        Stochastic mode: packed ternary + accumulators saved directly.
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

        path = Path(self.config.save_dir) / f"checkpoint_{step:06d}.pt"
        torch.save(_to_cpu(checkpoint), path)
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"Checkpoint saved to {path} ({size_mb:.0f} MB)")

        # Cleanup: keep only 3 most recent checkpoints
        checkpoints = sorted(Path(self.config.save_dir).glob("checkpoint_*.pt"))
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

        self.model.load_state_dict(state_dict)

        # Dequantize optimizer states from FP16 to FP32 if needed
        opt_state = checkpoint["optimizer_state_dict"]
        if checkpoint.get("optimizer_fp16", False):
            opt_state["state"] = {k: self._dequantize_optimizer_to_fp32(v) for k, v in opt_state["state"].items()}

        if self.hybrid:
            # Reconstruct cpu_params from model weights (avoids saving duplicate)
            for gp, cp in zip(self.model.parameters(), self.cpu_params):
                cp.data.copy_(gp.data.cpu())
            # Load optimizer state into cpu_params
            self.optimizer.load_state_dict(opt_state)
            # Sync to GPU model
            self._hybrid_sync_weights()
        else:
            self.optimizer.load_state_dict(opt_state)
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
            step = self.train_epoch(step)

            if step < self.config.max_steps:
                # Save checkpoint at intervals
                if step % self.config.save_interval == 0:
                    self.save_checkpoint(step)

        # Final save
        self.save_checkpoint(step)

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
