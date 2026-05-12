"""
Training loop orchestration for VADASR.

Single Responsibility: Manage the training lifecycle — forward pass,
backward pass, optimization, checkpointing, and logging.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..models.vadasr_model import VADASRModel
from .loss import VADASRLoss
from .scheduler import WarmupCosineScheduler

logger = logging.getLogger(__name__)


class Trainer:
    """VADASR training orchestrator.

    Parameters
    ----------
    model : VADASRModel
        The composed model.
    criterion : VADASRLoss
        Combined loss function.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    scheduler : WarmupCosineScheduler
        LR scheduler.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    cfg : dict
        Training config section.
    device : torch.device
        Target device.
    """

    def __init__(
        self,
        model: VADASRModel,
        criterion: VADASRLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        self.use_amp = cfg.get("use_amp", True)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_accum = cfg.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = cfg.get("max_grad_norm", 5.0)
        self.max_epochs = cfg.get("max_epochs", 100)

        # Checkpointing
        self.ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = cfg.get("save_every_n_epochs", 5)
        self.keep_top_k = cfg.get("keep_top_k", 3)

        # Early stopping
        self.patience = cfg.get("patience", 10)
        self.best_metric = float("inf")
        self.epochs_without_improvement = 0

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.ckpt_dir / "logs"))
        self.global_step = 0

    def train(self) -> None:
        """Run the full training loop."""
        logger.info("Starting training for %d epochs", self.max_epochs)

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)
            elapsed = time.time() - t0

            # Logging
            logger.info(
                "Epoch %d/%d | Train Loss: %.4f (vad=%.4f, ctc=%.4f) | "
                "Val Loss: %.4f | Time: %.1fs",
                epoch, self.max_epochs,
                train_metrics["total"], train_metrics["vad"],
                train_metrics["ctc"], val_metrics["total"], elapsed,
            )

            self.writer.add_scalars("loss/train", train_metrics, epoch)
            self.writer.add_scalars("loss/val", val_metrics, epoch)
            self.writer.add_scalar(
                "lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            # Checkpointing
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, val_metrics["total"])

            # Early stopping
            current_metric = val_metrics["total"]
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, current_metric, is_best=True)
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch, self.patience,
                )
                break

        self.writer.close()
        logger.info("Training complete. Best metric: %.4f", self.best_metric)

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        running = {"total": 0.0, "vad": 0.0, "ctc": 0.0}
        n_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            waveform = batch["waveform"].to(self.device)
            wav_lengths = batch["wav_lengths"].to(self.device)
            token_ids = batch["token_ids"].to(self.device)
            token_lengths = batch["token_lengths"].to(self.device)
            has_voice = batch["has_voice"].to(self.device)

            with autocast(enabled=self.use_amp):
                output = self.model(waveform, wav_lengths)
                losses = self.criterion(
                    gate_logits=output.gate_logits,
                    ctc_log_probs=output.ctc_log_probs,
                    ctc_lengths=output.ctc_lengths,
                    token_ids=token_ids,
                    token_lengths=token_lengths,
                    has_voice=has_voice,
                )
                loss = losses["total"] / self.grad_accum

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            running["total"] += losses["total"].item()
            running["vad"] += losses["vad"].item()
            running["ctc"] += losses["ctc"].item()
            n_batches += 1

        return {k: v / max(1, n_batches) for k, v in running.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        running = {"total": 0.0, "vad": 0.0, "ctc": 0.0}
        n_batches = 0

        for batch in self.val_loader:
            waveform = batch["waveform"].to(self.device)
            wav_lengths = batch["wav_lengths"].to(self.device)
            token_ids = batch["token_ids"].to(self.device)
            token_lengths = batch["token_lengths"].to(self.device)
            has_voice = batch["has_voice"].to(self.device)

            output = self.model(waveform, wav_lengths)
            losses = self.criterion(
                gate_logits=output.gate_logits,
                ctc_log_probs=output.ctc_log_probs,
                ctc_lengths=output.ctc_lengths,
                token_ids=token_ids,
                token_lengths=token_lengths,
                has_voice=has_voice,
            )

            running["total"] += losses["total"].item()
            running["vad"] += losses["vad"].item()
            running["ctc"] += losses["ctc"].item()
            n_batches += 1

        return {k: v / max(1, n_batches) for k, v in running.items()}

    def _save_checkpoint(
        self, epoch: int, metric: float, is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "global_step": self.global_step,
        }

        path = self.ckpt_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(state, path)
        logger.info("Saved checkpoint: %s (metric=%.4f)", path, metric)

        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(state, best_path)
            logger.info("Saved best checkpoint: %s", best_path)

        # Cleanup old checkpoints (keep top-K)
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self) -> None:
        """Keep only the top-K checkpoints by filename (most recent)."""
        ckpts = sorted(
            self.ckpt_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in ckpts[self.keep_top_k:]:
            old.unlink()

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint and return the epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_metric = ckpt.get("best_metric", float("inf"))
        self.global_step = ckpt.get("global_step", 0)
        epoch = ckpt["epoch"]
        logger.info("Loaded checkpoint from epoch %d", epoch)
        return epoch
