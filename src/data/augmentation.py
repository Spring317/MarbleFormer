"""
Audio augmentation pipeline.

Open/Closed: New augmentation transforms can be added as classes
without modifying existing code or the pipeline itself.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import torch
import torchaudio


class AudioTransform(ABC):
    """Abstract base for audio transforms."""

    @abstractmethod
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        ...


class SpeedPerturb(AudioTransform):
    """Tempo perturbation without pitch shift.

    Parameters
    ----------
    rates : list[float]
        Possible speed factors (e.g., [0.9, 1.0, 1.1]).
    sample_rate : int
        Audio sample rate.
    """

    def __init__(
        self, rates: list[float] | None = None, sample_rate: int = 16000
    ) -> None:
        self.rates = rates or [0.9, 0.95, 1.0, 1.05, 1.1]
        self.sample_rate = sample_rate

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        rate = random.choice(self.rates)
        if rate == 1.0:
            return waveform
        # Use torchaudio's speed effect via resampling approximation
        orig_len = waveform.size(0)
        new_sr = int(self.sample_rate * rate)
        resampler = torchaudio.transforms.Resample(new_sr, self.sample_rate)
        # Treat waveform as if recorded at new_sr, resample to original sr
        perturbed = resampler(waveform.unsqueeze(0)).squeeze(0)
        # Truncate or pad to original length
        if perturbed.size(0) > orig_len:
            perturbed = perturbed[:orig_len]
        elif perturbed.size(0) < orig_len:
            pad = orig_len - perturbed.size(0)
            perturbed = torch.nn.functional.pad(perturbed, (0, pad))
        return perturbed


class AddNoise(AudioTransform):
    """Mix Gaussian noise at random SNR.

    Parameters
    ----------
    snr_range : tuple[float, float]
        Min and max SNR in dB.
    """

    def __init__(self, snr_range: tuple[float, float] = (5.0, 20.0)) -> None:
        self.snr_min, self.snr_max = snr_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        snr_db = random.uniform(self.snr_min, self.snr_max)
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return waveform + noise


class SpecAugment(AudioTransform):
    """Time and frequency masking (applied to waveform domain).

    Note: This is a simplified waveform-domain version. For full
    SpecAugment, the masking is applied after mel extraction in the
    model's forward pass or as a tensor transform.
    """

    def __init__(self, zero_fraction: float = 0.02) -> None:
        self.zero_fraction = zero_fraction

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        n_samples = waveform.size(0)
        n_mask = int(n_samples * self.zero_fraction)
        if n_mask > 0:
            start = random.randint(0, n_samples - n_mask)
            waveform = waveform.clone()
            waveform[start:start + n_mask] = 0.0
        return waveform


class AugmentationPipeline:
    """Composable augmentation pipeline.

    Parameters
    ----------
    transforms : list[AudioTransform]
        List of transforms to apply sequentially.
    probability : float
        Probability of applying the entire pipeline (default 0.8).
    """

    def __init__(
        self,
        transforms: list[AudioTransform] | None = None,
        probability: float = 0.8,
    ) -> None:
        self.transforms = transforms or []
        self.probability = probability

    @classmethod
    def from_config(cls, cfg: dict) -> "AugmentationPipeline":
        """Build from an ``augmentation`` config dict."""
        if not cfg.get("enabled", True):
            return cls(transforms=[], probability=0.0)

        transforms: list[AudioTransform] = []

        if cfg.get("speed_perturb", {}).get("enabled", False):
            transforms.append(
                SpeedPerturb(rates=cfg["speed_perturb"].get("rates"))
            )

        if cfg.get("noise_mix", {}).get("enabled", False):
            snr = cfg["noise_mix"].get("snr_range", [5, 20])
            transforms.append(AddNoise(snr_range=tuple(snr)))

        if cfg.get("spec_augment", {}).get("enabled", False):
            transforms.append(SpecAugment())

        return cls(transforms=transforms)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.probability:
            return waveform
        for transform in self.transforms:
            waveform = transform(waveform)
        return waveform
