#!/usr/bin/env python3
"""
Train the VADASR gated early exit model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --debug --max_samples 10 --max_epochs 50
    python scripts/train.py --config configs/default.yaml --resume checkpoints/best.pt
    python scripts/train.py --config configs/default.yaml --nemo_weights path/to/conformer.nemo
    python scripts/train.py --config configs/default.yaml --nemo_weights path/to/model.pth --freeze_conformer
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.dataset import VADASRDataset
from src.data.collator import VADASRCollator
from src.data.augmentation import AugmentationPipeline
from src.models.vadasr_model import VADASRModel
from src.training.loss import VADASRLoss
from src.training.scheduler import WarmupCosineScheduler
from src.training.trainer import Trainer
from src.models.nemo_loader import load_nemo_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VADASR model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode (small dataset)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples per split (for debugging)",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None,
        help="Override max epochs",
    )
    parser.add_argument(
        "--nemo_weights", type=str, default=None,
        help="Path to NeMo .nemo/.ckpt/.pth file for pretrained Conformer weights",
    )
    parser.add_argument(
        "--freeze_conformer", action="store_true",
        help="Freeze Conformer encoder (phase-1: only train MarbleNet + gate + CTC head)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    if args.max_epochs:
        train_cfg["max_epochs"] = args.max_epochs

    # Seed
    set_seed(train_cfg.get("seed", 42))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- Tokenizer ----
    logger.info("Loading tokenizer...")
    tokenizer = BPETokenizer.from_config(cfg["tokenizer"])
    logger.info("Vocab size: %d, Blank ID: %d", tokenizer.vocab_size, tokenizer.blank_id)

    # ---- Augmentation ----
    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])

    # ---- Datasets ----
    logger.info("Loading datasets...")
    manifest_dir = Path(data_cfg.get("manifest_dir", "data/manifest"))
    train_manifest = manifest_dir / "combined_train.jsonl"
    val_manifest   = manifest_dir / "combined_val.jsonl"

    train_dataset = VADASRDataset.from_manifest(
        manifest=train_manifest,
        tokenizer=tokenizer,
        augmentation=augmentation,
        sample_rate=cfg["features"]["sample_rate"],
        max_audio_len_sec=data_cfg.get("max_audio_len_sec", 15.0),
        min_audio_len_sec=data_cfg.get("min_audio_len_sec", 0.5),
    )

    val_dataset = VADASRDataset.from_manifest(
        manifest=val_manifest,
        tokenizer=tokenizer,
        augmentation=None,  # no augmentation for validation
        sample_rate=cfg["features"]["sample_rate"],
        max_audio_len_sec=data_cfg.get("max_audio_len_sec", 15.0),
        min_audio_len_sec=data_cfg.get("min_audio_len_sec", 0.5),
    )

    # Limit samples in debug mode
    if args.debug and args.max_samples:
        train_dataset._indices = train_dataset._indices[:args.max_samples]
        val_dataset._indices = val_dataset._indices[:args.max_samples]

    collator = VADASRCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=collator,
    )

    logger.info("Train: %d samples, Val: %d samples", len(train_dataset), len(val_dataset))

    # ---- Model ----
    logger.info("Building model...")
    model = VADASRModel.from_config(cfg, vocab_size=tokenizer.vocab_size)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    # Parameter breakdown
    for name, module in [
        ("MelExtractor", model.mel_extractor),
        ("MarbleNet", model.marblenet),
        ("VADGate", model.vad_gate),
        ("Conformer", model.conformer),
        ("CTCHead", model.ctc_head),
    ]:
        n = sum(p.numel() for p in module.parameters())
        logger.info("  %-15s: %s params", name, f"{n:,}")

    # ---- Loss ----
    criterion = VADASRLoss.from_config(train_cfg, blank_id=tokenizer.blank_id)

    # ---- Load NeMo pretrained weights (optional) ----
    if args.nemo_weights:
        logger.info("Loading NeMo pretrained weights...")
        nemo_diag = load_nemo_weights(
            model,
            args.nemo_weights,
            load_conformer=True,
            load_ctc_head=True,
            freeze_loaded=args.freeze_conformer,
            device=device,
        )
        arch = nemo_diag["nemo_arch"]
        logger.info(
            "NeMo model: %s (d_model=%d, n_layers=%d)",
            arch["model_type"], arch["d_model"], arch["n_layers"],
        )
        logger.info(
            "Loaded: %d conformer + %d ctc params, %d skipped",
            nemo_diag["conformer_loaded"],
            nemo_diag["ctc_loaded"],
            len(nemo_diag["skipped"]),
        )
    elif args.freeze_conformer:
        # Freeze without loading NeMo weights (e.g., resuming from own checkpoint)
        logger.info("Freezing Conformer encoder (no NeMo weights)")
        for param in model.conformer.parameters():
            param.requires_grad = False

    # ---- Optimizer (separate LR groups for VAD vs ASR) ----
    vad_lr_scale = train_cfg.get("vad_lr_scale", 0.1)
    base_lr = train_cfg["learning_rate"]
    weight_decay = train_cfg.get("weight_decay", 0.0001)

    # Group 1: VAD branch (MarbleNet + VADGate) — converges fast, use lower LR
    vad_params = list(model.marblenet.parameters()) + list(model.vad_gate.parameters())
    # Group 2: ASR branch (Conformer + CTC Head) — needs higher LR
    asr_params = list(model.conformer.parameters()) + list(model.ctc_head.parameters())

    param_groups = [
        {"params": vad_params, "lr": base_lr * vad_lr_scale, "name": "vad"},
        {"params": asr_params, "lr": base_lr, "name": "asr"},
    ]
    logger.info(
        "Optimizer LR groups: VAD=%.6f, ASR=%.6f",
        base_lr * vad_lr_scale, base_lr,
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=weight_decay,
    )

    # ---- Scheduler ----
    total_steps = len(train_loader) * train_cfg["max_epochs"]
    scheduler = WarmupCosineScheduler.from_config(
        train_cfg, optimizer, total_steps
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=train_cfg,
        device=device,
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ---- Train ----
    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
