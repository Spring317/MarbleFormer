#!/usr/bin/env python3
"""
Train the VADASR gated early exit model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --debug --max_samples 10 --max_epochs 50
    python scripts/train.py --config configs/default.yaml --resume checkpoints/best.pt
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
    speech_manifest_train = Path("data/speech_train.jsonl")
    speech_manifest_test = Path("data/speech_test.jsonl")
    noise_manifest = Path(data_cfg.get("noise_manifest", "data/noise/manifest.jsonl"))
    noise_dir = data_cfg.get("noise_dir", "data/noise/audio")

    train_dataset = VADASRDataset.from_manifests(
        speech_manifest=speech_manifest_train,
        noise_manifest=noise_manifest,
        noise_dir=noise_dir,
        tokenizer=tokenizer,
        augmentation=augmentation,
        sample_rate=cfg["features"]["sample_rate"],
        max_audio_len_sec=data_cfg.get("max_audio_len_sec", 15.0),
        min_audio_len_sec=data_cfg.get("min_audio_len_sec", 0.5),
        speech_noise_ratio=data_cfg.get("speech_noise_ratio", 0.7),
    )

    val_dataset = VADASRDataset.from_manifests(
        speech_manifest=speech_manifest_test,
        noise_manifest=noise_manifest,
        noise_dir=noise_dir,
        tokenizer=tokenizer,
        augmentation=None,  # no augmentation for validation
        sample_rate=cfg["features"]["sample_rate"],
        max_audio_len_sec=data_cfg.get("max_audio_len_sec", 15.0),
        min_audio_len_sec=data_cfg.get("min_audio_len_sec", 0.5),
        speech_noise_ratio=0.5,  # balanced for validation
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

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0001),
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
