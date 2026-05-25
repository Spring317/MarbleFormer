#!/usr/bin/env python3
"""
Evaluate a trained VADASR model.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --config configs/default.yaml --checkpoint best.pt --threshold_search
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.dataset import VADASRDataset
from src.data.collator import VADASRCollator
from src.models.vadasr_model import VADASRModel
from src.evaluation.evaluator import Evaluator
from src.models.nemo_loader import load_nemo_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VADASR model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--nemo_weights", type=str, default=None,
        help="Path to NeMo .nemo/.ckpt/.pth file for pretrained Conformer weights",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Gate threshold override",
    )
    parser.add_argument(
        "--threshold_search", action="store_true",
        help="Run automatic threshold search",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Data split to evaluate",
    )
    args = parser.parse_args()

    if not args.checkpoint and not args.nemo_weights:
        parser.error("Must provide either --checkpoint or --nemo_weights")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Tokenizer
    tokenizer = BPETokenizer.from_config(cfg["tokenizer"])

    # Dataset
    data_cfg = cfg["data"]
    manifest_dir = Path(data_cfg.get("manifest_dir", "data/manifest"))
    test_dataset = VADASRDataset.from_manifests(
        speech_manifest=manifest_dir / f"speech_{args.split}.jsonl",
        noise_manifest=Path(data_cfg.get("noise_manifest", "data/manifest/noise.jsonl")),
        tokenizer=tokenizer,
        sample_rate=cfg["features"]["sample_rate"],
        max_audio_len_sec=data_cfg.get("max_audio_len_sec", 15.0),
        min_audio_len_sec=data_cfg.get("min_audio_len_sec", 0.5),
        speech_noise_ratio=0.5,
    )

    collator = VADASRCollator()
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collator,
    )

    logger.info("Test set: %d samples", len(test_dataset))

    # Model
    model = VADASRModel.from_config(cfg, vocab_size=tokenizer.vocab_size)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint: %s (epoch %d)", args.checkpoint, ckpt.get("epoch", -1))
        
    if args.nemo_weights:
        logger.info("Loading NeMo pretrained weights...")
        nemo_diag = load_nemo_weights(
            model,
            args.nemo_weights,
            load_conformer=True,
            load_ctc_head=True,
            freeze_loaded=False,
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

    # Evaluator
    evaluator = Evaluator(model=model, tokenizer=tokenizer, device=device)

    if args.threshold_search:
        eval_cfg = cfg.get("evaluation", {})
        thr_range = tuple(eval_cfg.get("threshold_range", [0.3, 0.7]))
        steps = eval_cfg.get("threshold_steps", 9)
        best_thr, metrics = evaluator.threshold_search(
            test_loader, threshold_range=thr_range, steps=steps
        )
        logger.info("=" * 60)
        logger.info("BEST THRESHOLD: %.3f", best_thr)
    else:
        metrics = evaluator.evaluate(test_loader, threshold=args.threshold)

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("VAD Metrics:")
    logger.info("  Precision : %.4f", metrics.vad.precision)
    logger.info("  Recall    : %.4f", metrics.vad.recall)
    logger.info("  F1        : %.4f", metrics.vad.f1)
    logger.info("  Accuracy  : %.4f", metrics.vad.accuracy)
    logger.info("")
    logger.info("ASR Metrics:")
    logger.info("  WER       : %.4f (%.2f%%)", metrics.asr.wer, metrics.asr.wer * 100)
    logger.info("  CER       : %.4f (%.2f%%)", metrics.asr.cer, metrics.asr.cer * 100)
    logger.info("  Samples   : %d", metrics.asr.num_samples)
    logger.info("")
    logger.info("Efficiency:")
    logger.info("  Avg Inference : %.2f ms", metrics.efficiency.avg_inference_ms)
    logger.info("  RTF           : %.4f", metrics.efficiency.rtf)
    logger.info("  Exit Rate     : %.2f%%", metrics.efficiency.exit_rate * 100)


if __name__ == "__main__":
    main()
