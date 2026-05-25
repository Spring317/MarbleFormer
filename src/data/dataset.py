"""
Unified dataset for VADASR — loads from a single combined manifest.

Each entry in the manifest has the format:
    {
        "audio_filepath": "/abs/path/to/audio.wav",
        "text":           "transcription or empty string",
        "duration":       3.45,
        "is_speech":      true | false
    }

The VAD encoder + gate learns from ``is_speech``.
The Conformer CTC head learns from ``text`` (tokenized on-the-fly with BPE).
Tokenization is skipped silently if the tokenizer is not provided.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VADASRDataset(Dataset):
    """Unified speech + noise dataset loaded from a single JSONL manifest.

    Each manifest entry must have:
        - ``audio_filepath`` (str): absolute path to audio file.
        - ``text``           (str): transcription for speech; empty for noise.
        - ``duration``       (float): duration in seconds.
        - ``is_speech``      (bool): True for speech, False for noise/non-speech.

    Parameters
    ----------
    data : list[dict]
        Pre-loaded list of manifest entries.
    sample_rate : int
        Target sample rate (audio is resampled if necessary).
    max_audio_len_sec : float
        Clips longer than this are truncated.
    min_audio_len_sec : float
        Clips shorter than this are zero-padded.
    tokenizer : Any | None
        Tokenizer with ``encode(text) -> list[int]``. If None, token_ids
        will always be an empty list (graceful degradation — the CTC loss
        will simply be skipped for those batches by the trainer).
    augmentation : Any | None
        Optional augmentation pipeline applied to the waveform.
    """

    def __init__(
        self,
        data: list[dict],
        sample_rate: int = 16000,
        max_audio_len_sec: float = 15.0,
        min_audio_len_sec: float = 0.5,
        tokenizer: Any = None,
        augmentation: Any = None,
    ) -> None:
        self.data = data
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_len_sec * sample_rate)
        self.min_samples = int(min_audio_len_sec * sample_rate)
        self.tokenizer = tokenizer
        self.augmentation = augmentation

        if tokenizer is None:
            logger.warning(
                "VADASRDataset: no tokenizer provided — token_ids will be "
                "empty for all samples. CTC loss will be skipped."
            )

        n_speech = sum(1 for e in data if e.get("is_speech", False))
        n_noise = len(data) - n_speech
        logger.info(
            "Dataset loaded: %d total (%d speech, %d noise)",
            len(data), n_speech, n_noise,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        entry = self.data[idx]

        audio_path = entry["audio_filepath"]
        text = entry.get("text", "")
        is_speech = bool(entry.get("is_speech", False))

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono [T]
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # Truncate / pad to length bounds
        if waveform.size(0) > self.max_samples:
            waveform = waveform[:self.max_samples]
        if waveform.size(0) < self.min_samples:
            pad_len = self.min_samples - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        wav_length = waveform.size(0)

        # Augmentation (training only)
        if self.augmentation is not None:
            waveform = self.augmentation(waveform)

        # Tokenize text — only for speech samples and only if tokenizer exists
        token_ids: list[int] = []
        if is_speech and self.tokenizer is not None and text and text.strip():
            token_ids = self.tokenizer.encode(text.strip())

        return {
            "waveform":  waveform,       # [T]
            "wav_length": wav_length,    # int
            "text":      text,           # raw transcript (or "")
            "token_ids": token_ids,      # list[int], empty for noise
            "has_voice": is_speech,      # bool — VAD label
        }

    # ------------------------------------------------------------------
    # Factory: load from a single unified JSONL manifest
    # ------------------------------------------------------------------

    @classmethod
    def from_manifest(
        cls,
        manifest: str | Path,
        tokenizer: Any = None,
        augmentation: Any = None,
        sample_rate: int = 16000,
        max_audio_len_sec: float = 15.0,
        min_audio_len_sec: float = 0.5,
        max_samples: int | None = None,
    ) -> "VADASRDataset":
        """Load from a single combined JSONL manifest.

        The manifest is expected to have been produced by ``prepare_data.py``
        Step 5 (combined_train/val/test.jsonl) with the unified format::

            {"audio_filepath": "...", "text": "...",
             "duration": 3.4, "is_speech": true}

        Parameters
        ----------
        manifest : path
            Path to the JSONL file.
        tokenizer : optional
            BPE tokenizer.  Pass None to disable tokenization.
        augmentation : optional
            Augmentation pipeline.
        max_samples : int | None
            Cap on the number of samples loaded (useful for debugging).
        """
        manifest = Path(manifest)
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        data: list[dict] = []
        skipped = 0
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                path = entry.get("audio_filepath", "")
                if not Path(path).exists():
                    skipped += 1
                    continue
                data.append(entry)
                if max_samples and len(data) >= max_samples:
                    break

        if skipped > 0:
            logger.warning(
                "Skipped %d entries with missing audio files in %s",
                skipped, manifest.name,
            )

        return cls(
            data=data,
            sample_rate=sample_rate,
            max_audio_len_sec=max_audio_len_sec,
            min_audio_len_sec=min_audio_len_sec,
            tokenizer=tokenizer,
            augmentation=augmentation,
        )

    # ------------------------------------------------------------------
    # Legacy factory: kept for backward compatibility
    # ------------------------------------------------------------------

    @classmethod
    def from_manifests(
        cls,
        speech_manifest: str | Path,
        noise_manifest: str | Path,
        speech_dir: str | Path = "",
        noise_dir: str | Path = "",
        tokenizer: Any = None,
        augmentation: Any = None,
        sample_rate: int = 16000,
        max_audio_len_sec: float = 15.0,
        min_audio_len_sec: float = 0.5,
        speech_noise_ratio: float = 0.7,
        **kwargs,
    ) -> "VADASRDataset":
        """Legacy two-manifest loader (speech + noise separately).

        Kept for backward compatibility.  New code should use
        ``from_manifest()`` with the unified combined_*.jsonl format.
        """
        logger.warning(
            "from_manifests() is deprecated. Use from_manifest() with the "
            "unified combined_train/val/test.jsonl manifests instead."
        )

        data: list[dict] = []

        # Load speech entries
        speech_manifest = Path(speech_manifest)
        if speech_manifest.exists():
            with open(speech_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    path = entry.get("audio_filepath", "")
                    if speech_dir:
                        path = str(Path(speech_dir) / path)
                    if Path(path).exists():
                        data.append({
                            "audio_filepath": path,
                            "text": entry.get("text", ""),
                            "duration": entry.get("duration", 0.0),
                            "is_speech": entry.get("is_speech", True),
                        })

        # Load noise entries (sampled to match speech_noise_ratio)
        noise_entries: list[dict] = []
        noise_manifest_path = Path(noise_manifest)
        if noise_manifest_path.exists():
            with open(noise_manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    path = entry.get("audio_filepath", "")
                    if noise_dir:
                        path = str(Path(noise_dir) / path)
                    if Path(path).exists():
                        noise_entries.append({
                            "audio_filepath": path,
                            "text": "",
                            "duration": entry.get("duration", 0.0),
                            "is_speech": False,
                        })

        n_speech = len(data)
        if speech_noise_ratio > 0 and noise_entries:
            target_noise = int(n_speech * (1 - speech_noise_ratio) / speech_noise_ratio)
            target_noise = min(target_noise, len(noise_entries))
            noise_entries = noise_entries[:target_noise]

        data.extend(noise_entries)
        random.shuffle(data)

        return cls(
            data=data,
            sample_rate=sample_rate,
            max_audio_len_sec=max_audio_len_sec,
            min_audio_len_sec=min_audio_len_sec,
            tokenizer=tokenizer,
            augmentation=augmentation,
        )
