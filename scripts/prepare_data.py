#!/usr/bin/env python3
"""
Prepare data manifests for VADASR training.

Downloads BUD500 from HuggingFace (streaming) and generates JSONL
manifests for both speech and noise datasets.

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
    python scripts/prepare_data.py --config configs/default.yaml --max_samples 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import soundfile as sf
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_data")


def prepare_bud500(
    dataset_name: str,
    output_dir: Path,
    split: str = "train",
    max_samples: int | None = None,
) -> Path:
    """Download BUD500 and create a speech manifest.

    Returns the path to the generated JSONL manifest.
    """
    from datasets import load_dataset

    manifest_path = output_dir / f"speech_{split}.jsonl"
    wav_dir = output_dir / f"speech_{split}_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        logger.info("Speech manifest already exists: %s", manifest_path)
        return manifest_path

    logger.info("Loading BUD500 split=%s (streaming)...", split)
    ds = load_dataset(dataset_name, split=split, streaming=True)

    count = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for sample in ds:
            if max_samples and count >= max_samples:
                break

            audio = sample["audio"]
            text = sample.get("transcription", "")
            if not text or not text.strip():
                continue

            # Save waveform
            wav_path = wav_dir / f"{count:08d}.wav"
            if not wav_path.exists():
                import numpy as np
                waveform = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                sf.write(str(wav_path), waveform, sr)

            entry = {
                "audio_filepath": str(wav_path),
                "text": text.strip(),
                "duration": len(audio["array"]) / audio["sampling_rate"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

            if count % 1000 == 0:
                logger.info("Processed %d speech samples...", count)

    logger.info(
        "Created speech manifest: %s (%d samples)", manifest_path, count
    )
    return manifest_path


def prepare_noise_manifest(
    noise_dir: Path,
    output_path: Path,
) -> Path:
    """Scan a directory of noise audio files and create a JSONL manifest.

    Expected structure:
        noise_dir/
            *.wav, *.flac, *.ogg, etc.
            subdirs/
                *.wav
    """
    if output_path.exists():
        logger.info("Noise manifest already exists: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    extensions = {".wav", ".flac", ".ogg", ".mp3"}

    with open(output_path, "w", encoding="utf-8") as f:
        for audio_file in sorted(noise_dir.rglob("*")):
            if audio_file.suffix.lower() not in extensions:
                continue
            try:
                info = sf.info(str(audio_file))
                entry = {
                    "audio_filepath": audio_file.name,
                    "duration": info.duration,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                logger.warning("Skipping %s: %s", audio_file, e)

    logger.info(
        "Created noise manifest: %s (%d files)", output_path, count
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VADASR data")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max speech samples to download (for debugging)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"],
        help="Dataset splits to prepare",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Prepare speech data (BUD500)
    for split in args.splits:
        prepare_bud500(
            dataset_name=data_cfg["bud500_name"],
            output_dir=output_dir,
            split=split,
            max_samples=args.max_samples,
        )

    # Prepare noise manifest
    noise_dir = Path(data_cfg.get("noise_dir", "data/noise/audio"))
    noise_manifest = Path(data_cfg.get("noise_manifest", "data/noise/manifest.jsonl"))

    if noise_dir.exists():
        prepare_noise_manifest(noise_dir, noise_manifest)
    else:
        logger.warning(
            "Noise directory not found: %s — skipping noise manifest. "
            "Place noise audio files there and re-run.",
            noise_dir,
        )


if __name__ == "__main__":
    main()
