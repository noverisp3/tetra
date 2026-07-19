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
    gradient_accumulation_steps: int = 1  # effective batch = 16
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
    save_interval: int = 200
    save_dir: str = "checkpoints"

    # Logging
    log_interval: int = 10

    # Device
    device: str = "cpu"
    dtype: str = "float32"

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

        # Optimizer (AdamW with ternary-specific settings)
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
        self.model.train()
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
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def train_epoch(self, step_start: int) -> int:
        """Train for one epoch (or until data exhausted)."""
        self.model.train()
        total_loss = 0.0
        step = step_start

        remaining = self.config.max_steps - step_start
        pbar = tqdm(self.train_loader, desc=f"Training", total=min(len(self.train_loader), remaining // max(1, self.config.gradient_accumulation_steps)))

        for batch_idx, batch in enumerate(pbar):
            if step >= self.config.max_steps:
                break

            loss = self.train_step(batch)
            total_loss += loss

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # LR scheduler
                self.scheduler.step()
                step += 1

                # Logging
                if step % self.config.log_interval == 0:
                    avg_loss = total_loss / self.config.log_interval
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.train_losses.append(avg_loss)
                    self.learning_rates.append(lr)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", step=f"{step}/{self.config.max_steps}")
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

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        path = Path(self.config.save_dir) / f"checkpoint_{step:06d}.pt"
        torch.save(checkpoint, path)
        print(f"  [SAVE] Checkpoint saved to {path}")

        # Also save as latest
        latest_path = Path(self.config.save_dir) / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Move optimizer state to device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.model.to(self.device)
        print(f"  [LOAD] Checkpoint loaded from {path} (step {checkpoint['step']})")
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
