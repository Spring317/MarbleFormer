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


def _profile_flops(
    model: VADASRModel,
    waveform: torch.Tensor,
    wav_lengths: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> int | None:
    if not hasattr(torch, "profiler"):
        logger.warning("torch.profiler not available; skipping FLOPs/TOPS.")
        return None

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and hasattr(torch.profiler.ProfilerActivity, "CUDA"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    original_threshold = model.vad_gate.threshold
    model.vad_gate.threshold = threshold
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with torch.inference_mode():
                _ = model.inference(waveform, wav_lengths)
    except Exception as e:
        logger.warning("FLOPs profiling failed: %s", e)
        return None
    finally:
        model.vad_gate.threshold = original_threshold

    total_flops = 0
    for evt in prof.key_averages():
        flops = getattr(evt, "flops", None)
        if flops:
            total_flops += flops
    if total_flops == 0:
        logger.warning("Profiler returned 0 FLOPs; check torch build.")
        return None
    return total_flops


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
    parser.add_argument(
        "--export_transcripts", type=str, default=None,
        help="Path to save transcripts in plain text format",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=None,
        help="Override evaluation batch size (e.g., 1 for fairness checks)",
    )
    parser.add_argument(
        "--profile_flops", action="store_true",
        help="Profile FLOPs/TOPS on a single batch",
    )
    args = parser.parse_args()

    if not args.checkpoint and not args.nemo_weights:
        parser.error("Must provide either --checkpoint or --nemo_weights")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Tokenizer (optional — evaluation can run without it for VAD-only metrics)
    tokenizer = None
    try:
        tokenizer = BPETokenizer.from_config(cfg["tokenizer"])
        logger.info("Tokenizer loaded: vocab_size=%d", tokenizer.vocab_size)
    except Exception as e:
        logger.warning("Tokenizer not available (%s) — WER/CER will be skipped.", e)

    # Dataset
    data_cfg = cfg["data"]
    manifest_dir = Path(data_cfg.get("manifest_dir", "data/manifest"))
    # Map --split to the unified manifest name (test → combined_test.jsonl)
    split_name = args.split
    manifest_file = manifest_dir / f"combined_{split_name}.jsonl"
    if not manifest_file.exists():
        # Fallback for legacy split naming (e.g. speech_test.jsonl)
        manifest_file = manifest_dir / f"speech_{split_name}.jsonl"
        logger.warning(
            "combined_%s.jsonl not found, falling back to %s",
            split_name, manifest_file.name,
        )
    test_dataset = VADASRDataset.from_manifest(
        manifest=manifest_file,
        tokenizer=tokenizer,
        sample_rate=cfg["features"]["sample_rate"],
        max_audio_len_sec=data_cfg.get("max_audio_len_sec", 15.0),
        min_audio_len_sec=data_cfg.get("min_audio_len_sec", 0.5),
    )

    collator = VADASRCollator()
    eval_batch_size = args.eval_batch_size or cfg["evaluation"]["batch_size"]
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
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

    # Optional FLOPs/TOPS profiling on a single batch
    flops_early = None
    flops_full = None
    prof_audio_s = None
    if args.profile_flops:
        try:
            prof_batch = next(iter(test_loader))
        except StopIteration:
            logger.warning("No data available for FLOPs profiling.")
        else:
            waveform = prof_batch["waveform"].to(device)
            wav_lengths = prof_batch["wav_lengths"].to(device)
            prof_audio_s = wav_lengths.sum().item() / cfg["features"]["sample_rate"]

            flops_early = _profile_flops(
                evaluator.model, waveform, wav_lengths, device, threshold=2.0
            )
            flops_full = _profile_flops(
                evaluator.model, waveform, wav_lengths, device, threshold=-1.0
            )

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
        metrics = evaluator.evaluate(
            test_loader,
            threshold=args.threshold,
            save_transcripts_path=args.export_transcripts,
        )

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

    if flops_early is not None and flops_full is not None and prof_audio_s:
        logger.info("")
        logger.info("Compute (profile batch):")
        logger.info("  Audio Seconds : %.3f", prof_audio_s)
        early_gflops = flops_early / 1e9
        full_gflops = flops_full / 1e9
        early_tops = (flops_early / max(1e-6, prof_audio_s)) / 1e12
        full_tops = (flops_full / max(1e-6, prof_audio_s)) / 1e12
        logger.info("  Early-Exit    : %.3f GFLOPs | %.6f TOPS@1x", early_gflops, early_tops)
        logger.info("  Full ASR      : %.3f GFLOPs | %.6f TOPS@1x", full_gflops, full_tops)

    if args.export_transcripts:
        logger.info("Saved transcripts: %s", args.export_transcripts)


if __name__ == "__main__":
    main()
