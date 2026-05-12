#!/usr/bin/env python3
"""
Prepare data manifests for VADASR training.

- Scans a LOCAL BUD500 dataset directory (already downloaded)
- Downloads & resamples Freesound background noise by category
- Generates JSONL manifests for speech and noise data

Usage:
    # Generate manifests from local data (no download)
    python scripts/prepare_data.py --config configs/default.yaml

    # Download Freesound background noise first, then generate manifests
    python scripts/prepare_data.py --config configs/default.yaml \
        --download_freesound --freesound_api_key YOUR_KEY

    # Limit speech samples for debugging
    python scripts/prepare_data.py --config configs/default.yaml --max_samples 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import urllib.request
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_data")


# ============================================================================
# Freesound background noise categories — used by NeMo / MarbleNet
# ============================================================================
FREESOUND_NOISE_CATEGORIES: list[str] = [
    "air-conditioner",
    "car-horn",
    "children-playing",
    "dog-bark",
    "drilling",
    "engine-idling",
    "gun-shot",
    "jackhammer",
    "siren",
    "street-music",
    "rain",
    "thunderstorm",
    "wind",
    "crowd-noise",
    "traffic",
    "typing",
    "clock-ticking",
    "water-drops",
    "washing-machine",
    "vacuum-cleaner",
    "helicopter",
    "chainsaw",
    "crackling-fire",
    "hand-saw",
    "insects",
    "ocean-waves",
    "footsteps",
    "door-knock",
    "birds",
    "white-noise",
]

TARGET_SAMPLE_RATE: int = 16000


# ============================================================================
# 1. Local BUD500 speech data — NO downloading
# ============================================================================

def prepare_local_bud500(
    speech_data_root: Path,
    out_dir: Path,
    split: str = "train",
    max_samples: int | None = None,
) -> Path:
    """Scan a local BUD500 directory and create a speech manifest.

    Expected structure (any of these are supported):
        speech_data_root/
            dataset_dict.json     # HuggingFace load_from_disk format (.arrow)
            train/
                *.arrow + state.json
            test/
                *.arrow + state.json
        OR
            train/
                *.parquet
        OR
            train/
                *.jsonl | *.json  (NeMo manifest)
        OR
            train/
                *.wav | *.flac    (raw audio + sidecar .txt)

    The function auto-detects the format in priority order:
    Arrow → Parquet → JSONL → raw audio.

    Parameters
    ----------
    speech_data_root : Path
        Root directory of the locally-downloaded BUD500 dataset.
    out_dir : Path
        Directory to write the manifest JSONL file.
    split : str
        Dataset split name ("train", "test", "validation").
    max_samples : int | None
        Cap on number of samples (for debugging).

    Returns
    -------
    Path to the generated manifest.
    """
    manifest_path = out_dir / f"speech_{split}.jsonl"

    if manifest_path.exists():
        n_lines = sum(1 for _ in open(manifest_path))
        if n_lines > 0:
            logger.info(
                "Speech manifest already exists: %s (%d samples)",
                manifest_path, n_lines,
            )
            return manifest_path
        else:
            logger.warning(
                "Speech manifest %s exists but is empty — regenerating.",
                manifest_path,
            )
            manifest_path.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine search directory ---
    split_dir = speech_data_root / split
    if split_dir.is_dir():
        search_root = split_dir
    else:
        search_root = speech_data_root
        logger.warning(
            "No '%s' subdirectory found in %s — scanning entire root.",
            split, speech_data_root,
        )

    # --- Check for HuggingFace Arrow format (load_from_disk) ---
    # This is the format saved by datasets.save_to_disk() and used by
    # HuggingFace datasets cache. It contains .arrow files + state.json.
    arrow_files = sorted(search_root.glob("*.arrow"))
    has_dataset_dict = (speech_data_root / "dataset_dict.json").exists()
    has_state_json = (search_root / "state.json").exists()

    if arrow_files and (has_dataset_dict or has_state_json):
        return _manifest_from_arrow(
            speech_data_root, out_dir, split, max_samples
        )

    # --- Check for parquet files (HuggingFace cache format) ---
    parquet_files = sorted(search_root.glob("*.parquet"))
    if parquet_files:
        return _manifest_from_parquet(
            parquet_files, out_dir, split, max_samples
        )

    # --- Check for JSONL manifest (NeMo format) ---
    existing_manifests = list(search_root.glob("*.jsonl")) + \
                         list(search_root.glob("*.json"))
    # Filter out dataset_info.json / state.json / dataset_dict.json
    existing_manifests = [
        p for p in existing_manifests
        if p.name not in {"dataset_info.json", "state.json", "dataset_dict.json"}
    ]
    if existing_manifests:
        return _manifest_from_nemo_jsonl(
            existing_manifests, out_dir, split, max_samples
        )

    # --- Scan for raw audio files ---
    return _manifest_from_audio_dir(
        search_root, out_dir, split, max_samples
    )


def _manifest_from_arrow(
    dataset_root: Path,
    out_dir: Path,
    split: str,
    max_samples: int | None,
) -> Path:
    """Generate manifest from HuggingFace Arrow format (load_from_disk).

    This handles the format produced by datasets.save_to_disk(), which
    contains .arrow files + state.json + dataset_dict.json.

    Structure:
        dataset_root/
            dataset_dict.json
            train/
                data-00000-of-00105.arrow
                ...
                state.json
                dataset_info.json
    """
    from datasets import load_from_disk

    manifest_path = out_dir / f"speech_{split}.jsonl"
    wav_dir = out_dir / f"speech_{split}_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading HuggingFace Arrow dataset from %s (split='%s')...",
                dataset_root, split)

    ds_dict = load_from_disk(str(dataset_root))

    # Handle both DatasetDict and single Dataset
    if hasattr(ds_dict, 'keys'):
        if split in ds_dict:
            ds = ds_dict[split]
        else:
            available = list(ds_dict.keys())
            logger.error(
                "Split '%s' not found. Available splits: %s", split, available
            )
            # Fall back to first available split
            ds = ds_dict[available[0]]
            logger.warning("Using split '%s' instead.", available[0])
    else:
        ds = ds_dict

    total = len(ds)
    logger.info("  Found %d samples in split '%s'", total, split)

    count = 0
    skipped = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(ds):
            if max_samples and count >= max_samples:
                break

            # BUD500 uses "transcription" as the text field
            text = sample.get("transcription", sample.get("text", ""))
            if not text or not text.strip():
                skipped += 1
                continue

            audio = sample.get("audio")
            if audio is None:
                skipped += 1
                continue

            # Extract audio data
            waveform_array = audio.get("array") if isinstance(audio, dict) else None
            sr = audio.get("sampling_rate", TARGET_SAMPLE_RATE) if isinstance(audio, dict) else TARGET_SAMPLE_RATE

            if waveform_array is None:
                skipped += 1
                continue

            # Save waveform to wav
            wav_path = wav_dir / f"{count:08d}.wav"
            if not wav_path.exists():
                import numpy as np
                waveform = np.array(waveform_array, dtype=np.float32)
                sf.write(str(wav_path), waveform, sr)

            duration = len(waveform_array) / sr
            entry = {
                "audio_filepath": str(wav_path.resolve()),
                "text": text.strip(),
                "duration": round(duration, 4),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

            if count % 5000 == 0:
                logger.info(
                    "  Processed %d / %d samples (%.1f%%)...",
                    count, total, 100 * (i + 1) / total,
                )

    logger.info(
        "Created speech manifest (Arrow): %s (%d samples, %d skipped)",
        manifest_path, count, skipped,
    )
    return manifest_path


def _manifest_from_parquet(
    parquet_files: list[Path],
    out_dir: Path,
    split: str,
    max_samples: int | None,
) -> Path:
    """Generate manifest from HuggingFace parquet files (local cache)."""
    from datasets import load_dataset

    manifest_path = out_dir / f"speech_{split}.jsonl"
    wav_dir = out_dir / f"speech_{split}_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %d parquet file(s) for split '%s'...",
                len(parquet_files), split)

    data_files = {split: [str(p) for p in parquet_files]}
    ds = load_dataset("parquet", data_files=data_files, split=split)

    count = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for sample in ds:
            if max_samples and count >= max_samples:
                break

            audio = sample.get("audio")
            text = sample.get("transcription", sample.get("text", ""))
            if not text or not text.strip():
                continue

            # Save waveform to wav
            wav_path = wav_dir / f"{count:08d}.wav"
            if not wav_path.exists():
                import numpy as np
                waveform = np.array(audio["array"], dtype=np.float32)
                sr = audio.get("sampling_rate", TARGET_SAMPLE_RATE)
                sf.write(str(wav_path), waveform, sr)

            duration = len(audio["array"]) / audio.get(
                "sampling_rate", TARGET_SAMPLE_RATE
            )
            entry = {
                "audio_filepath": str(wav_path),
                "text": text.strip(),
                "duration": duration,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

            if count % 5000 == 0:
                logger.info("  Processed %d speech samples...", count)

    logger.info("Created speech manifest: %s (%d samples)",
                manifest_path, count)
    return manifest_path


def _manifest_from_nemo_jsonl(
    jsonl_files: list[Path],
    out_dir: Path,
    split: str,
    max_samples: int | None,
) -> Path:
    """Copy/filter an existing NeMo-style JSONL manifest."""
    manifest_path = out_dir / f"speech_{split}.jsonl"

    count = 0
    with open(manifest_path, "w", encoding="utf-8") as out_f:
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if max_samples and count >= max_samples:
                        break
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    try:
                        entry = json.loads(line_stripped)
                    except json.JSONDecodeError:
                        try:
                            import ast
                            entry = ast.literal_eval(line_stripped)
                        except (SyntaxError, ValueError):
                            logger.warning("Skipping malformed line in %s", jsonl_file.name)
                            continue

                    # Ensure required fields
                    if "audio_filepath" not in entry:
                        continue
                    text = entry.get("text", entry.get("transcription", ""))
                    if not text or not text.strip():
                        continue
                    entry["text"] = text.strip()
                    out_f.write(
                        json.dumps(entry, ensure_ascii=False) + "\n"
                    )
                    count += 1

    logger.info("Created speech manifest from JSONL: %s (%d samples)",
                manifest_path, count)
    return manifest_path


def _manifest_from_audio_dir(
    search_root: Path,
    out_dir: Path,
    split: str,
    max_samples: int | None,
) -> Path:
    """Scan directory for audio files with sidecar text files."""
    manifest_path = out_dir / f"speech_{split}.jsonl"
    extensions = {".wav", ".flac", ".ogg", ".mp3"}

    count = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for audio_file in sorted(search_root.rglob("*")):
            if audio_file.suffix.lower() not in extensions:
                continue
            if max_samples and count >= max_samples:
                break

            # Look for sidecar text file: same name with .txt
            txt_file = audio_file.with_suffix(".txt")
            text = ""
            if txt_file.exists():
                text = txt_file.read_text(encoding="utf-8").strip()

            # Also check for transcript in parent's manifest
            if not text:
                text = f"[unlabeled:{audio_file.name}]"
                logger.debug("No transcript for %s", audio_file)

            try:
                info = sf.info(str(audio_file))
                entry = {
                    "audio_filepath": str(audio_file),
                    "text": text,
                    "duration": info.duration,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                logger.warning("Skipping %s: %s", audio_file, e)

    logger.info("Created speech manifest from audio dir: %s (%d samples)",
                manifest_path, count)
    return manifest_path


# ============================================================================
# 2. Freesound background noise — download + resample
# ============================================================================

def download_freesound_category(
    api_key: str,
    query: str,
    download_dir: Path,
    max_sounds: int = 100,
    page_size: int = 50,
) -> int:
    """Download sounds for a single Freesound category/query.

    Uses the Freesound API v2 text search endpoint with token auth.
    Downloads preview-quality MP3 files (no OAuth2 required).

    Parameters
    ----------
    api_key : str
        Freesound API key.
    query : str
        Search query (e.g., "rain", "traffic", "white-noise").
    download_dir : Path
        Directory to save downloaded files.
    max_sounds : int
        Maximum number of sounds to download per category.
    page_size : int
        API page size (max 150).

    Returns
    -------
    int — number of files downloaded.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://freesound.org/apiv2/search/text/"

    downloaded = 0
    page = 1

    while downloaded < max_sounds:
        params = urllib.parse.urlencode({
            "query": query.replace("-", " "),
            "fields": "id,name,previews,duration",
            "page_size": min(page_size, max_sounds - downloaded),
            "page": page,
            "token": api_key,
            "filter": "duration:[1 TO 30]",  # 1-30 seconds
        })
        url = f"{base_url}?{params}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.warning(
                "  API request failed for '%s' page %d: %s", query, page, e
            )
            break

        results = data.get("results", [])
        if not results:
            break

        for sound in results:
            if downloaded >= max_sounds:
                break

            sound_id = sound["id"]
            name = sound.get("name", str(sound_id))
            # Sanitize filename
            safe_name = "".join(
                c if c.isalnum() or c in "-_." else "_" for c in name
            )
            out_path = download_dir / f"{query}_{sound_id}_{safe_name}.mp3"

            if out_path.exists():
                downloaded += 1
                continue

            # Get preview URL (no OAuth needed)
            previews = sound.get("previews", {})
            preview_url = previews.get(
                "preview-hq-mp3",
                previews.get("preview-lq-mp3", ""),
            )
            if not preview_url:
                continue

            try:
                urllib.request.urlretrieve(preview_url, str(out_path))
                downloaded += 1
            except Exception as e:
                logger.debug("  Failed to download %s: %s", name, e)

        # Check for next page
        if data.get("next") is None:
            break
        page += 1

    return downloaded


def download_freesound_all_categories(
    api_key: str,
    download_dir: Path,
    categories: list[str] | None = None,
    max_per_category: int = 100,
) -> int:
    """Download Freesound sounds across all noise categories.

    Parameters
    ----------
    api_key : str
        Freesound API key.
    download_dir : Path
        Base download directory.
    categories : list[str] | None
        Category queries. Defaults to FREESOUND_NOISE_CATEGORIES.
    max_per_category : int
        Max sounds per category.

    Returns
    -------
    int — total files downloaded.
    """
    if categories is None:
        categories = FREESOUND_NOISE_CATEGORIES

    download_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    logger.info("Downloading Freesound noise data (%d categories, "
                "max %d per category)...", len(categories), max_per_category)

    for i, category in enumerate(categories, 1):
        logger.info(
            "  [%d/%d] Category: '%s'", i, len(categories), category
        )
        cat_dir = download_dir / category
        n = download_freesound_category(
            api_key=api_key,
            query=category,
            download_dir=cat_dir,
            max_sounds=max_per_category,
        )
        logger.info("    Downloaded %d sounds", n)
        total += n

    logger.info("Total Freesound downloads: %d files", total)
    return total


def resample_audio_dir(
    input_dir: Path,
    output_dir: Path,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> int:
    """Resample all audio files in a directory to target sample rate.

    Converts any format (MP3, FLAC, OGG, WAV) → 16kHz mono WAV.
    Uses sox if available, falls back to Python-based resampling.

    Parameters
    ----------
    input_dir : Path
        Directory containing downloaded audio files.
    output_dir : Path
        Directory to write resampled WAV files.
    target_sr : int
        Target sample rate (default 16000).

    Returns
    -------
    int — number of files resampled.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".wav", ".flac", ".ogg", ".mp3"}

    # Check if sox is available (preferred — faster, handles edge cases)
    has_sox = _check_sox_available()
    if has_sox:
        logger.info("Using sox for resampling")
    else:
        logger.info("sox not found — using Python (torchaudio) resampling")

    count = 0
    for audio_file in sorted(input_dir.rglob("*")):
        if audio_file.suffix.lower() not in extensions:
            continue

        out_name = audio_file.stem + ".wav"
        # Preserve category subdirectory structure
        relative = audio_file.relative_to(input_dir)
        out_path = output_dir / relative.parent / out_name

        if out_path.exists():
            count += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if has_sox:
                _resample_with_sox(audio_file, out_path, target_sr)
            else:
                _resample_with_python(audio_file, out_path, target_sr)
            count += 1
        except Exception as e:
            logger.warning("  Resample failed for %s: %s", audio_file, e)

    logger.info("Resampled %d files to %d Hz in %s", count, target_sr,
                output_dir)
    return count


def _check_sox_available() -> bool:
    """Check if sox command is available."""
    try:
        subprocess.run(
            ["sox", "--version"],
            capture_output=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _resample_with_sox(
    input_path: Path, output_path: Path, target_sr: int
) -> None:
    """Resample a single file using sox."""
    subprocess.run(
        [
            "sox", str(input_path), "-r", str(target_sr),
            "-c", "1",  # mono
            "-b", "16",  # 16-bit
            str(output_path),
        ],
        capture_output=True, check=True, timeout=30,
    )


def _resample_with_python(
    input_path: Path, output_path: Path, target_sr: int
) -> None:
    """Resample a single file using torchaudio."""
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(input_path))

    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    torchaudio.save(str(output_path), waveform, target_sr)


# ============================================================================
# 3. Noise manifest generation
# ============================================================================

def prepare_noise_manifest(
    noise_dir: Path,
    output_path: Path,
) -> Path:
    """Scan a directory of noise audio files and create a JSONL manifest.

    Stores absolute paths so the dataset loader can find files regardless
    of the working directory.

    Parameters
    ----------
    noise_dir : Path
        Root directory containing noise audio files (may have subdirs).
    output_path : Path
        Path to write the JSONL manifest.

    Returns
    -------
    Path to the generated manifest.
    """
    if output_path.exists():
        n_lines = sum(1 for _ in open(output_path))
        logger.info(
            "Noise manifest already exists: %s (%d files)",
            output_path, n_lines,
        )
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    extensions = {".wav", ".flac", ".ogg", ".mp3"}

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for audio_file in sorted(noise_dir.rglob("*")):
            if audio_file.suffix.lower() not in extensions:
                continue
            try:
                info = sf.info(str(audio_file))
                entry = {
                    "audio_filepath": str(audio_file.resolve()),
                    "duration": info.duration,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                logger.warning("Skipping %s: %s", audio_file, e)

    logger.info("Created noise manifest: %s (%d files)", output_path, count)
    return output_path


# ============================================================================
# 4. Manifest statistics & balancing
# ============================================================================

@dataclass
class ManifestStats:
    """Statistics from a JSONL manifest."""
    num_samples: int = 0
    total_duration_sec: float = 0.0

    @property
    def total_hours(self) -> float:
        return self.total_duration_sec / 3600.0

    @property
    def avg_duration_sec(self) -> float:
        return self.total_duration_sec / max(1, self.num_samples)


def count_manifest_stats(manifest_path: Path) -> ManifestStats:
    """Count samples and total duration from a JSONL manifest.

    Parameters
    ----------
    manifest_path : Path
        Path to a JSONL manifest file with 'duration' fields.

    Returns
    -------
    ManifestStats with num_samples and total_duration_sec.
    """
    stats = ManifestStats()
    if not manifest_path.exists():
        return stats

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            duration = entry.get("duration", 0.0)
            stats.num_samples += 1
            stats.total_duration_sec += float(duration)

    return stats


# --- Synthetic noise generators ---
# These are used to fill the gap when downloaded Freesound data
# is insufficient to match the speech dataset volume.

SYNTHETIC_NOISE_TYPES: list[str] = [
    "white",
    "pink",
    "brown",
    "babble",
    "hum",
    "fan",
]


def _generate_white_noise(n_samples: int) -> "np.ndarray":
    """Gaussian white noise."""
    import numpy as np
    return np.random.randn(n_samples).astype(np.float32) * 0.3


def _generate_pink_noise(n_samples: int) -> "np.ndarray":
    """Pink noise (1/f spectrum) via Voss-McCartney algorithm."""
    import numpy as np
    n_rows = 16
    n_cols = n_samples
    array = np.full((n_rows, n_cols), np.nan)
    array[0, :] = np.random.randn(n_cols)
    array[:, 0] = np.random.randn(n_rows)
    cols = np.random.geometric(0.5, n_cols)
    cols[cols >= n_rows] = 0
    rows = np.random.randint(0, n_rows, n_cols)
    for i in range(1, n_cols):
        array[:, i] = array[:, i - 1]
        array[rows[i], i] = np.random.randn()
    result = np.nansum(array, axis=0)
    result = result / np.max(np.abs(result) + 1e-8) * 0.3
    return result.astype(np.float32)


def _generate_brown_noise(n_samples: int) -> "np.ndarray":
    """Brownian (red) noise — cumulative sum of white noise."""
    import numpy as np
    white = np.random.randn(n_samples).astype(np.float32)
    brown = np.cumsum(white)
    brown = brown / np.max(np.abs(brown) + 1e-8) * 0.3
    return brown.astype(np.float32)


def _generate_babble_noise(n_samples: int) -> "np.ndarray":
    """Simulated babble: sum of multiple frequency-shifted noise streams."""
    import numpy as np
    n_voices = random.randint(5, 15)
    babble = np.zeros(n_samples, dtype=np.float32)
    for _ in range(n_voices):
        freq = random.uniform(100, 800)
        t = np.arange(n_samples) / TARGET_SAMPLE_RATE
        voice = np.sin(2 * np.pi * freq * t + random.uniform(0, 2 * np.pi))
        # Modulate with random envelope
        envelope = np.random.uniform(0.1, 0.5) * (
            1 + 0.5 * np.sin(2 * np.pi * random.uniform(0.5, 3) * t)
        )
        babble += (voice * envelope).astype(np.float32)
    babble = babble / np.max(np.abs(babble) + 1e-8) * 0.3
    return babble.astype(np.float32)


def _generate_hum_noise(n_samples: int) -> "np.ndarray":
    """Electrical hum (50/60Hz + harmonics)."""
    import numpy as np
    base_freq = random.choice([50.0, 60.0])
    t = np.arange(n_samples) / TARGET_SAMPLE_RATE
    hum = np.zeros(n_samples, dtype=np.float32)
    for harmonic in range(1, random.randint(4, 8)):
        amplitude = 1.0 / harmonic
        hum += amplitude * np.sin(
            2 * np.pi * base_freq * harmonic * t
            + random.uniform(0, 2 * np.pi)
        )
    hum = hum / np.max(np.abs(hum) + 1e-8) * 0.25
    return hum.astype(np.float32)


def _generate_fan_noise(n_samples: int) -> "np.ndarray":
    """Simulated fan/HVAC — band-limited noise with low-frequency rumble."""
    import numpy as np
    # Broadband component
    noise = np.random.randn(n_samples).astype(np.float32)
    # Simple low-pass via moving average
    kernel_size = random.randint(3, 15)
    kernel = np.ones(kernel_size) / kernel_size
    filtered = np.convolve(noise, kernel, mode="same")
    # Add low rumble
    t = np.arange(n_samples) / TARGET_SAMPLE_RATE
    rumble = 0.2 * np.sin(
        2 * np.pi * random.uniform(20, 80) * t
    )
    result = (filtered + rumble).astype(np.float32)
    result = result / np.max(np.abs(result) + 1e-8) * 0.25
    return result.astype(np.float32)


_NOISE_GENERATORS = {
    "white": _generate_white_noise,
    "pink": _generate_pink_noise,
    "brown": _generate_brown_noise,
    "babble": _generate_babble_noise,
    "hum": _generate_hum_noise,
    "fan": _generate_fan_noise,
}


def generate_synthetic_noise(
    output_dir: Path,
    target_samples: int,
    target_duration_sec: float,
    existing_noise_samples: int = 0,
    existing_noise_duration: float = 0.0,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[int, float]:
    """Generate synthetic noise files to match a target count and duration.

    Produces a mix of white, pink, brown, babble, hum, and fan noise
    clips until the combined (existing + generated) count and duration
    meet the targets.

    Parameters
    ----------
    output_dir : Path
        Directory to write generated WAV files.
    target_samples : int
        Target total number of noise samples.
    target_duration_sec : float
        Target total noise duration in seconds.
    existing_noise_samples : int
        Number of noise samples already available.
    existing_noise_duration : float
        Total duration of existing noise in seconds.
    sample_rate : int
        Output sample rate.

    Returns
    -------
    (n_generated, generated_duration_sec)
    """
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    needed_samples = max(0, target_samples - existing_noise_samples)
    needed_duration = max(0.0, target_duration_sec - existing_noise_duration)

    if needed_samples == 0 and needed_duration <= 0:
        logger.info("  Existing noise already meets targets — no generation needed.")
        return 0, 0.0

    # Compute average clip duration to distribute evenly
    if needed_samples > 0:
        avg_clip_sec = needed_duration / needed_samples
    else:
        avg_clip_sec = 5.0  # default 5s clips

    # Clamp clip duration to reasonable range
    avg_clip_sec = max(1.0, min(avg_clip_sec, 15.0))

    noise_types = list(_NOISE_GENERATORS.keys())
    generated_count = 0
    generated_duration = 0.0

    logger.info(
        "  Generating synthetic noise: need %d samples, %.1f hours",
        needed_samples, needed_duration / 3600,
    )
    logger.info(
        "  Average clip duration: %.1f sec", avg_clip_sec,
    )

    while generated_count < needed_samples or generated_duration < needed_duration:
        # Pick a random noise type
        noise_type = random.choice(noise_types)
        generator = _NOISE_GENERATORS[noise_type]

        # Vary clip duration ±30%
        clip_duration = avg_clip_sec * random.uniform(0.7, 1.3)
        clip_duration = max(1.0, min(clip_duration, 15.0))
        n_audio_samples = int(clip_duration * sample_rate)

        # Generate
        waveform = generator(n_audio_samples)

        # Random amplitude scaling for variety
        scale = random.uniform(0.1, 1.0)
        waveform = waveform * scale

        # Save
        out_path = output_dir / f"synth_{noise_type}_{generated_count:06d}.wav"
        sf.write(str(out_path), waveform, sample_rate)

        generated_count += 1
        generated_duration += clip_duration

        if generated_count % 1000 == 0:
            logger.info(
                "    Generated %d clips (%.1f hours so far)...",
                generated_count, generated_duration / 3600,
            )

    logger.info(
        "  Synthetic noise generation complete: %d clips, %.2f hours",
        generated_count, generated_duration / 3600,
    )
    return generated_count, generated_duration


def balance_noise_to_speech(
    speech_manifest: Path,
    noise_manifest: Path,
    noise_dir: Path,
    synthetic_dir: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> None:
    """Ensure noise data matches speech data in sample count and hours.

    1. Count speech samples and total hours from the speech manifest.
    2. Count existing noise samples and total hours.
    3. If noise is insufficient, generate synthetic noise to fill the gap.
    4. Append new entries to the noise manifest.

    Parameters
    ----------
    speech_manifest : Path
        Path to the speech JSONL manifest.
    noise_manifest : Path
        Path to the noise JSONL manifest (will be appended to).
    noise_dir : Path
        Directory containing existing noise files.
    synthetic_dir : Path
        Directory to write generated synthetic noise files.
    sample_rate : int
        Target sample rate.
    """
    # --- Count speech ---
    speech_stats = count_manifest_stats(speech_manifest)
    logger.info(
        "  Speech stats: %d samples, %.2f hours (avg %.1fs/clip)",
        speech_stats.num_samples,
        speech_stats.total_hours,
        speech_stats.avg_duration_sec,
    )

    # --- Count existing noise ---
    noise_stats = count_manifest_stats(noise_manifest)
    logger.info(
        "  Noise stats:  %d samples, %.2f hours (avg %.1fs/clip)",
        noise_stats.num_samples,
        noise_stats.total_hours,
        noise_stats.avg_duration_sec,
    )

    # --- Check if we need more noise ---
    samples_deficit = speech_stats.num_samples - noise_stats.num_samples
    duration_deficit = (
        speech_stats.total_duration_sec - noise_stats.total_duration_sec
    )

    if samples_deficit <= 0 and duration_deficit <= 0:
        logger.info(
            "  ✓ Noise already balanced (≥ speech in both count and duration)"
        )
        return

    logger.info(
        "  Deficit: %d samples, %.2f hours → generating synthetic noise",
        max(0, samples_deficit),
        max(0, duration_deficit) / 3600,
    )

    # --- Generate synthetic noise ---
    n_gen, dur_gen = generate_synthetic_noise(
        output_dir=synthetic_dir,
        target_samples=max(0, samples_deficit),
        target_duration_sec=max(0.0, duration_deficit),
        existing_noise_samples=0,  # deficit already computed
        existing_noise_duration=0.0,
        sample_rate=sample_rate,
    )

    if n_gen == 0:
        return

    # --- Append synthetic entries to noise manifest ---
    logger.info("  Appending %d synthetic entries to noise manifest...", n_gen)
    extensions = {".wav"}
    appended = 0
    with open(noise_manifest, "a", encoding="utf-8") as f:
        for audio_file in sorted(synthetic_dir.rglob("*")):
            if audio_file.suffix.lower() not in extensions:
                continue
            try:
                info = sf.info(str(audio_file))
                entry = {
                    "audio_filepath": str(audio_file.resolve()),
                    "duration": info.duration,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                appended += 1
            except Exception as e:
                logger.warning("Skipping %s: %s", audio_file, e)

    # --- Final count ---
    final_stats = count_manifest_stats(noise_manifest)
    logger.info(
        "  ✓ Balanced noise manifest: %d samples, %.2f hours "
        "(appended %d synthetic clips)",
        final_stats.num_samples,
        final_stats.total_hours,
        appended,
    )


# ============================================================================
# 5. Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare VADASR data manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan local BUD500 + existing noise, generate manifests only
  python scripts/prepare_data.py --config configs/default.yaml

  # Also download Freesound background noise
  python scripts/prepare_data.py --config configs/default.yaml \\
      --download_freesound --freesound_api_key YOUR_KEY

  # Quick test with 1000 speech samples
  python scripts/prepare_data.py --config configs/default.yaml --max_samples 1000
        """,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max speech samples per split (for debugging)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"],
        help="Dataset splits to prepare",
    )
    # --- Freesound download ---
    parser.add_argument(
        "--download_freesound", action="store_true",
        help="Download Freesound background noise by category",
    )
    parser.add_argument(
        "--freesound_api_key", type=str, default=None,
        help="Freesound API key (get one at https://freesound.org/apiv2/apply/)",
    )
    parser.add_argument(
        "--max_per_category", type=int, default=100,
        help="Max Freesound sounds per noise category (default: 100)",
    )
    parser.add_argument(
        "--freesound_categories", nargs="+", default=None,
        help="Override noise categories (default: built-in 30 categories)",
    )
    # --- Resample ---
    parser.add_argument(
        "--skip_resample", action="store_true",
        help="Skip the resampling step (if data is already 16kHz WAV)",
    )
    # --- Force regeneration ---
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-generation of manifests even if they exist",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # ---- Directory layout ----
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)

    speech_data_root = Path(data_cfg.get(
        "bud500_local_dir", "data/bud500"
    ))
    background_data_root = Path(data_cfg.get(
        "noise_dir", "data/background_noise"
    ))
    resampled_noise_dir = Path(data_cfg.get(
        "noise_resampled_dir", "data/background_noise_resampled"
    ))
    out_dir = Path(data_cfg.get("manifest_dir", "data/manifest"))
    noise_manifest = Path(data_cfg.get(
        "noise_manifest", "data/manifest/noise.jsonl"
    ))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Force cleanup
    if args.force:
        for p in out_dir.glob("*.jsonl"):
            p.unlink()
            logger.info("Removed existing manifest: %s", p)

    # ================================================================
    # Step 1: Speech manifests (local BUD500 — no download)
    # ================================================================
    print("=" * 60)
    print("STEP 1: Prepare speech manifests (local BUD500)")
    print("=" * 60)

    if not speech_data_root.exists():
        logger.error(
            "BUD500 data directory not found: %s\n"
            "  Please ensure the dataset is downloaded and set\n"
            "  'data.bud500_local_dir' in the config YAML.\n"
            "  Expected structure:\n"
            "    %s/\n"
            "      train/  (*.wav or *.parquet)\n"
            "      test/   (*.wav or *.parquet)",
            speech_data_root, speech_data_root,
        )
        sys.exit(1)

    for split in args.splits:
        prepare_local_bud500(
            speech_data_root=speech_data_root,
            out_dir=out_dir,
            split=split,
            max_samples=args.max_samples,
        )

    # ================================================================
    # Step 2: Download Freesound noise (optional)
    # ================================================================
    if args.download_freesound:
        print("\n" + "=" * 60)
        print("STEP 2: Download Freesound background noise")
        print("=" * 60)

        if not args.freesound_api_key:
            logger.error(
                "Freesound API key required! Get one at:\n"
                "  https://freesound.org/apiv2/apply/\n"
                "Then run with: --freesound_api_key YOUR_KEY"
            )
            sys.exit(1)

        freesound_raw_dir = data_folder / "freesound_raw"
        download_freesound_all_categories(
            api_key=args.freesound_api_key,
            download_dir=freesound_raw_dir,
            categories=args.freesound_categories,
            max_per_category=args.max_per_category,
        )

        # Resample to 16kHz mono WAV
        if not args.skip_resample:
            print("\n  Resampling to 16kHz mono WAV...")
            resample_audio_dir(
                input_dir=freesound_raw_dir,
                output_dir=background_data_root,
                target_sr=TARGET_SAMPLE_RATE,
            )
        else:
            # If skipping resample, use raw dir as noise dir
            background_data_root = freesound_raw_dir

    # ================================================================
    # Step 3: Noise manifest (from existing downloads)
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Prepare noise manifest")
    print("=" * 60)

    if background_data_root.exists():
        # If resampled dir exists, prefer it
        if resampled_noise_dir.exists() and any(
            resampled_noise_dir.rglob("*.wav")
        ):
            prepare_noise_manifest(resampled_noise_dir, noise_manifest)
        else:
            prepare_noise_manifest(background_data_root, noise_manifest)
    else:
        # Create an empty noise manifest so balancing can append to it
        noise_manifest.parent.mkdir(parents=True, exist_ok=True)
        if not noise_manifest.exists():
            noise_manifest.touch()
        logger.warning(
            "No downloaded noise found at %s — will generate synthetic noise.",
            background_data_root,
        )

    # ================================================================
    # Step 4: Count speech stats & balance noise to match
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Balance noise to match speech (count + hours)")
    print("=" * 60)

    # Use the training split manifest as the reference for balancing
    train_speech_manifest = out_dir / "speech_train.jsonl"
    if train_speech_manifest.exists():
        speech_stats = count_manifest_stats(train_speech_manifest)
        print(f"\n  BUD500 speech (train):")
        print(f"    Samples       : {speech_stats.num_samples:>10,}")
        print(f"    Total hours   : {speech_stats.total_hours:>10.2f}")
        print(f"    Avg clip (sec): {speech_stats.avg_duration_sec:>10.1f}")

        noise_stats = count_manifest_stats(noise_manifest)
        print(f"\n  Noise (before balancing):")
        print(f"    Samples       : {noise_stats.num_samples:>10,}")
        print(f"    Total hours   : {noise_stats.total_hours:>10.2f}")

        synthetic_dir = data_folder / "synthetic_noise"
        balance_noise_to_speech(
            speech_manifest=train_speech_manifest,
            noise_manifest=noise_manifest,
            noise_dir=background_data_root,
            synthetic_dir=synthetic_dir,
            sample_rate=TARGET_SAMPLE_RATE,
        )
    else:
        logger.warning(
            "No speech_train.jsonl found — skipping noise balancing."
        )

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Speech data root      : {speech_data_root}")
    print(f"  Background noise root : {background_data_root}")
    print(f"  Synthetic noise dir   : {data_folder / 'synthetic_noise'}")
    print(f"  Manifest output dir   : {out_dir}")
    print()
    for manifest in sorted(out_dir.glob("*.jsonl")):
        stats = count_manifest_stats(manifest)
        print(
            f"  {manifest.name:30s} : {stats.num_samples:>8,} samples, "
            f"{stats.total_hours:>8.2f} hours"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
