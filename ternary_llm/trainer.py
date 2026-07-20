"""Training pipeline for Ternary LLM.

Implements:
- AdamW optimizer with ternary-specific settings
- Cosine learning rate scheduler with warmup
- Gradient accumulation
- Checkpointing and logging
- Validation loop
"""
import os
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Training configuration for Ternary LLM."""
    # Model (overridden by data pipeline from tokenizer metadata)
    vocab_size: int = 8192  # custom BPE tokenizer
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

    # Quantization (STE)
    ternary_scale: float = 0.7  # Δ = scale × mean(|W|), lower → more {-1,+1}, higher → more 0
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
        return self.min_lr / self.base_lrs[0] + 0.5 * (1 - self.min_lr / self.base_lrs[0]) * (1 + math.cos(math.pi * progress))


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
                    print("  WARNING: torch_directml not found, falling back to CPU")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(config.device)
        else:
            # Auto-detect: prefer CPU (faster for ternary models without CUDA)
            try:
                import torch_directml
                self.device = torch_directml.device()
            except Exception:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")
        print(f"  Device: {self.device}")

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
            )
            print(f"  Hybrid mode: model on {self.device}, optimizer on CPU")
        else:
            # Standard: optimizer on same device as model
            self.optimizer = torch.optim.AdamW(
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

        # Mixed precision (GPU only - CPU uses float32)
        self.use_amp = False
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

        # Logging
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """Single training step with gradient accumulation."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        if self.use_amp:
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16,
            ):
                _, loss = self.model(x, y)
                loss = loss / self.config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        else:
            _, loss = self.model(x, y)
            raw_loss = loss.item()
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

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
            total_loss += loss
            micro_count += 1

            if micro_count % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping (unique params only)
                torch.nn.utils.clip_grad_norm_(
                    self._unique_params(), self.config.grad_clip
                )

                # Optimizer step
                if self.hybrid:
                    self._hybrid_sync_gradients()
                    self.optimizer.step()
                    self._hybrid_sync_weights()
                elif self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.hybrid:
                    self.model.zero_grad(set_to_none=True)

                # LR scheduler
                self.scheduler.step()
                step += 1
                pbar.update(1)

                # Logging
                if step % self.config.log_interval == 0:
                    avg_loss = total_loss / self.config.log_interval
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.train_losses.append(avg_loss)
                    self.learning_rates.append(lr)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                    total_loss = 0.0

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

            _, loss = self.model(x, y)

            total_loss += loss.item()
            num_batches += 1

        avg_val_loss = total_loss / max(num_batches, 1)
        print(f"  [VAL] Step {self.scheduler.step_count:5d} | Loss: {avg_val_loss:.4f}")
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

        STE mode: latent weights → FP16, optimizer states → FP16.
        Stochastic mode: packed ternary + accumulators saved directly.
        Packed ternary (2-bit) always included for inference export.
        """
        from .quantization import pack_ternary, ternary_quantize

        is_stochastic = self.config.mode == "stochastic"
        state_dict = {}
        inference_data = {}

        if is_stochastic:
            # Stochastic: packed weights + accumulators are already compact
            state_dict = self.model.state_dict()
            inference_data = {
                k: {"packed": v.cpu(), "shape": list(v.shape) if "packed" in k else None}
                for k, v in state_dict.items() if "packed_weights" in k
            }
            # Downcast accumulators to FP16 for storage
            for k, v in state_dict.items():
                if "accumulator" in k and v.is_floating_point():
                    state_dict[k] = v.half()
        else:
            # STE: move all to CPU first (single bulk transfer), then process
            sd = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in sd.items():
                if "latent_weights" in k:
                    state_dict[k] = v.half()
                    inference_data[k] = {"packed": pack_ternary(ternary_quantize(v)), "shape": list(v.shape)}
                else:
                    state_dict[k] = v

        # Quantize optimizer states to FP16
        opt_state = self.optimizer.state_dict()
        opt_state["state"] = {k: self._quantize_optimizer_to_fp16(v) for k, v in opt_state["state"].items()}

        checkpoint = {
            "step": step,
            "model_state_dict": state_dict,
            "inference_ternary": inference_data,
            "optimizer_state_dict": opt_state,
            "config": self.config.__dict__,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "optimizer_fp16": True,
            "mode": self.config.mode,
        }
        if not is_stochastic:
            checkpoint["latent_fp16"] = True

        path = Path(self.config.save_dir) / f"checkpoint_{step:06d}.pt"
        torch.save(checkpoint, path)
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  [SAVE] Checkpoint saved to {path} ({size_mb:.0f} MB)")

        # Cleanup: keep only 3 most recent checkpoints
        checkpoints = sorted(Path(self.config.save_dir).glob("checkpoint_*.pt"))
        while len(checkpoints) > 3:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"  [DEL] Removed old checkpoint: {oldest.name}")

        # Save training history at each checkpoint (crash-safe)
        history = {
            "train_losses": self.train_losses,
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

        checkpoint = torch.load(path, map_location="cpu")
        raw_state = checkpoint["model_state_dict"]
        ckpt_mode = checkpoint.get("mode", "ste")

        if ckpt_mode == "stochastic":
            # Stochastic: FP16 accumulators → FP32, packed weights stay uint8
            state_dict = {}
            for k, v in raw_state.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                    state_dict[k] = v.float()
                else:
                    state_dict[k] = v
        elif checkpoint.get("latent_fp16", False):
            # STE new format: FP16 latent weights → FP32
            state_dict = {}
            for k, v in raw_state.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                    state_dict[k] = v.float()
                else:
                    state_dict[k] = v
        # Old format: packed ternary → unpack (legacy compatibility)
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
        self.val_losses = checkpoint.get("val_losses", [])
        self.learning_rates = checkpoint.get("learning_rates", [])
        print(f"  [LOAD] Checkpoint loaded from {path} (step {checkpoint['step']})")
        print(f"         Restored {len(self.train_losses)} train losses, {len(self.val_losses)} val losses")
        return checkpoint["step"]

    def train(self, resume_step: int = 0):
        """Main training loop."""
        print("=" * 60)
        print("Ternary LLM Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Dtype: {self.config.dtype}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Grad accum: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"LR: {self.config.learning_rate} -> {self.config.min_lr}")
        print("=" * 60)

        step = resume_step
        start_time = time.time()

        # Fast-forward scheduler
        self.scheduler.step_count = resume_step
        lr_scale = self.scheduler._get_lr_scale()
        for group, base_lr in zip(self.scheduler.optimizer.param_groups, self.scheduler.base_lrs):
            group["lr"] = base_lr * lr_scale
        if resume_step > 0:
            print(f"  Resumed from step {resume_step}, LR = {self.scheduler.optimizer.param_groups[0]['lr']:.6f}")

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
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
        }
        history_path = Path(self.config.save_dir) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)
