"""
Unified dataset for VADASR — combines BUD500 speech + noise data.

Liskov Substitution: Implements standard torch Dataset interface.
Single Responsibility: Load, preprocess, and yield audio samples with
VAD labels.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class VADASRDataset(Dataset):
    """Unified speech + noise dataset.

    Combines BUD500 (speech with transcription) and noise data
    (silence / non-speech) into a single dataset with dual labels:
    ``has_voice`` (bool) and ``text`` (str).

    Parameters
    ----------
    speech_data : list[dict]
        List of speech samples: {"audio_path": str, "text": str}.
    noise_data : list[dict]
        List of noise samples: {"audio_path": str}.
    sample_rate : int
        Target sample rate.
    max_audio_len_sec : float
        Maximum audio length in seconds (longer clips are truncated).
    min_audio_len_sec : float
        Minimum audio length in seconds (shorter clips are skipped).
    speech_noise_ratio : float
        Fraction of speech samples per epoch (0.7 = 70% speech).
    tokenizer : Any
        Tokenizer with ``encode(text) -> list[int]``.
    augmentation : Any | None
        Optional augmentation pipeline.
    """

    def __init__(
        self,
        speech_data: list[dict],
        noise_data: list[dict],
        sample_rate: int = 16000,
        max_audio_len_sec: float = 15.0,
        min_audio_len_sec: float = 0.5,
        speech_noise_ratio: float = 0.7,
        tokenizer: Any = None,
        augmentation: Any = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_len_sec * sample_rate)
        self.min_samples = int(min_audio_len_sec * sample_rate)
        self.tokenizer = tokenizer
        self.augmentation = augmentation

        # Build combined dataset with balanced sampling
        self.speech_data = speech_data
        self.noise_data = noise_data

        # Compute epoch size based on ratio
        n_speech = len(speech_data)
        n_noise = len(noise_data)

        if speech_noise_ratio > 0 and n_noise > 0:
            target_noise = int(n_speech * (1 - speech_noise_ratio) / speech_noise_ratio)
            target_noise = min(target_noise, n_noise)
        else:
            target_noise = n_noise

        self._indices: list[tuple[str, int]] = []
        for i in range(n_speech):
            self._indices.append(("speech", i))
        for i in range(target_noise):
            self._indices.append(("noise", i % n_noise))

        random.shuffle(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        source, data_idx = self._indices[idx]

        if source == "speech":
            item = self.speech_data[data_idx]
            audio_path = item["audio_path"]
            text = item.get("text", "")
            has_voice = True
        else:
            item = self.noise_data[data_idx]
            audio_path = item["audio_path"]
            text = ""
            has_voice = False

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # [T]

        # Truncate / validate length
        if waveform.size(0) > self.max_samples:
            waveform = waveform[:self.max_samples]
        if waveform.size(0) < self.min_samples:
            # Pad short audio
            pad_len = self.min_samples - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        wav_length = waveform.size(0)

        # Apply augmentation
        if self.augmentation is not None:
            waveform = self.augmentation(waveform)

        # Tokenize text
        token_ids = []
        if self.tokenizer is not None and text:
            token_ids = self.tokenizer.encode(text)

        return {
            "waveform": waveform,
            "wav_length": wav_length,
            "text": text,
            "token_ids": token_ids,
            "has_voice": has_voice,
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_hf_and_manifest(
        cls,
        hf_dataset,
        noise_manifest: str | Path,
        noise_dir: str | Path,
        tokenizer: Any = None,
        augmentation: Any = None,
        cache_dir: str | Path = "data/cache",
        **kwargs,
    ) -> "VADASRDataset":
        """Build from HuggingFace dataset + noise manifest.

        Parameters
        ----------
        hf_dataset
            HuggingFace dataset split with 'audio' and 'transcription'.
        noise_manifest : str | Path
            Path to JSONL manifest for noise files.
        noise_dir : str | Path
            Base directory for noise audio files.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Process speech data from HuggingFace
        speech_data = []
        for i, sample in enumerate(hf_dataset):
            audio = sample["audio"]
            text = sample.get("transcription", "")

            # Save audio to cache for consistent loading
            wav_path = cache_dir / f"speech_{i:08d}.wav"
            if not wav_path.exists():
                waveform = torch.tensor(
                    audio["array"], dtype=torch.float32
                ).unsqueeze(0)
                sr = audio["sampling_rate"]
                torchaudio.save(str(wav_path), waveform, sr)

            speech_data.append({"audio_path": str(wav_path), "text": text})

        # Load noise manifest
        noise_data = []
        noise_manifest = Path(noise_manifest)
        noise_dir = Path(noise_dir)
        if noise_manifest.exists():
            with open(noise_manifest, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    audio_path = noise_dir / entry["audio_filepath"]
                    if audio_path.exists():
                        noise_data.append({"audio_path": str(audio_path)})

        return cls(
            speech_data=speech_data,
            noise_data=noise_data,
            tokenizer=tokenizer,
            augmentation=augmentation,
            **kwargs,
        )

    @classmethod
    def from_manifests(
        cls,
        speech_manifest: str | Path,
        noise_manifest: str | Path,
        speech_dir: str | Path = "",
        noise_dir: str | Path = "",
        tokenizer: Any = None,
        augmentation: Any = None,
        **kwargs,
    ) -> "VADASRDataset":
        """Build from two JSONL manifests (speech + noise)."""
        speech_data = []
        speech_manifest = Path(speech_manifest)
        if speech_manifest.exists():
            with open(speech_manifest, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    path = entry["audio_filepath"]
                    if speech_dir:
                        path = str(Path(speech_dir) / path)
                    speech_data.append({
                        "audio_path": path,
                        "text": entry.get("text", ""),
                    })

        noise_data = []
        noise_manifest_path = Path(noise_manifest)
        if noise_manifest_path.exists():
            with open(noise_manifest_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    path = entry["audio_filepath"]
                    if noise_dir:
                        path = str(Path(noise_dir) / path)
                    noise_data.append({"audio_path": path})

        return cls(
            speech_data=speech_data,
            noise_data=noise_data,
            tokenizer=tokenizer,
            augmentation=augmentation,
            **kwargs,
        )
