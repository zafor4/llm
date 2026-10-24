"""Production-grade Trainer for Small GPT Causal Language Models."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import SmallGPT


@dataclass
class TrainingArgs:
    """Hyperparameters and runtime arguments for SmallGPT training."""

    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    batch_size: int = 16
    eval_interval: int = 100
    eval_steps: int = 20
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = torch.cuda.is_available()
    seed: int = 42


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine learning rate scheduler with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def configure_optimizers(model: SmallGPT, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str):
    """Separate parameters that receive weight decay from biases and LayerNorms."""
    decay_params = []
    nodecay_params = []
    
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer


class GPTTrainer:
    """Trainer managing execution, evaluation, metrics logging, and checkpointing."""

    def __init__(
        self,
        model: SmallGPT,
        args: TrainingArgs,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(args.device)
        self.model.to(self.device)

        self.optimizer = configure_optimizers(
            model=self.model,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            betas=(args.beta1, args.beta2),
            device_type=self.device.type,
        )

        min_ratio = args.min_learning_rate / args.learning_rate
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            min_lr_ratio=min_ratio,
        )

        self.scaler = torch.amp.GradScaler(enabled=(args.mixed_precision and self.device.type == "cuda"))
        self.history: List[Dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.stop_requested = False

        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def request_stop(self) -> None:
        self.stop_requested = True

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Compute average validation loss and perplexity across eval_steps."""
        self.model.eval()
        total_loss = 0.0
        steps = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.args.mixed_precision):
                _, loss, _ = self.model(x, targets=y)
            total_loss += loss.item()
            steps += 1
            if steps >= self.args.eval_steps:
                break

        avg_loss = total_loss / max(1, steps)
        perplexity = math.exp(min(avg_loss, 20.0))  # Cap to prevent numerical overflow
        self.model.train()
        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    def train(
        self,
        on_log: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Main training loop across step budget."""
        self.model.train()
        step = 0
        train_iter = iter(self.train_loader)
        start_time = time.time()
        tokens_processed = 0
        val_metrics = {"val_loss": 0.0, "val_perplexity": 0.0}

        print(f"Starting training on {self.device.type} ({self.model.get_num_params():,} parameters)...")

        while step < self.args.max_steps:
            if self.stop_requested or (should_stop and should_stop()):
                break
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)

            x, y = x.to(self.device), y.to(self.device)
            tokens_processed += x.numel()

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=self.device.type, enabled=self.args.mixed_precision):
                _, loss, _ = self.model(x, targets=y)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.args.grad_clip > 0.0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            else:
                grad_norm = 0.0

            # Optimizer step & scaler update
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            step += 1

            if on_log:
                on_log(
                    {
                        "step": step,
                        "train_loss": loss.item(),
                        "val_loss": val_metrics.get("val_loss", 0.0),
                        "val_perplexity": val_metrics.get("val_perplexity", 0.0),
                        "tokens_per_sec": tokens_processed / max(0.001, time.time() - start_time),
                        "partial": True,
                    }
                )

            # Periodic Evaluation & Checkpoint Logging
            if step % self.args.eval_interval == 0 or step == self.args.max_steps:
                val_metrics = self.evaluate()
                elapsed = time.time() - start_time
                throughput = tokens_processed / max(0.001, elapsed)
                current_lr = self.scheduler.get_last_lr()[0]
                peak_gpu_mb = (
                    torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                    if self.device.type == "cuda"
                    else 0.0
                )

                log_entry = {
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": val_metrics["val_loss"],
                    "val_perplexity": val_metrics["val_perplexity"],
                    "lr": current_lr,
                    "tokens_per_sec": throughput,
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, (int, float, torch.Tensor)) else 0.0,
                    "elapsed_sec": elapsed,
                    "peak_gpu_mb": peak_gpu_mb,
                }
                self.history.append(log_entry)
                if on_log:
                    on_log({**log_entry, "partial": False})

                print(
                    f"Step {step:4d}/{self.args.max_steps} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val PPL: {val_metrics['val_perplexity']:.2f} | "
                    f"Throughput: {throughput:.0f} tok/s | "
                    f"Peak VRAM: {peak_gpu_mb:.1f} MB"
                )

                # Save best model checkpoint
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt", step=step, val_loss=self.best_val_loss)

        # Save final checkpoint and history
        self.save_checkpoint("final_model.pt", step=step, val_loss=val_metrics["val_loss"])
        with open(self.checkpoint_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {
            "final_step": step,
            "best_val_loss": self.best_val_loss,
            "best_perplexity": math.exp(min(self.best_val_loss, 20.0)),
            "total_tokens": tokens_processed,
            "total_time_sec": time.time() - start_time,
            "history": self.history,
        }

    def save_checkpoint(self, filename: str, step: int, val_loss: float) -> None:
        save_path = self.checkpoint_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config.to_dict(),
            "step": step,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, save_path)
