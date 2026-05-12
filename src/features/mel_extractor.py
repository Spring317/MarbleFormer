"""
Mel-spectrogram feature extractor.

Single Responsibility: Extract log-mel spectrogram features from raw audio
waveforms, matching the configuration shared by MarbleNet and Conformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T


class MelSpectrogramExtractor(nn.Module):
    """Differentiable log-mel spectrogram extractor.

    Converts raw 16 kHz waveforms into log-mel spectrograms suitable for
    both the MarbleNet encoder (VAD) and the Conformer encoder (ASR).

    Parameters
    ----------
    sample_rate : int
        Expected audio sample rate (default 16000).
    n_mels : int
        Number of mel filter-bank channels (default 80).
    n_fft : int
        FFT window size (default 512).
    hop_length : int
        Hop size in samples (default 160 → 10 ms at 16 kHz).
    win_length : int
        Window size in samples (default 400 → 25 ms at 16 kHz).
    fmin : float
        Minimum frequency for mel filter-bank.
    fmax : float
        Maximum frequency for mel filter-bank.
    normalize : bool
        If True, apply per-utterance CMVN (zero-mean, unit-variance).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            power=2.0,
        )
        # Stable log transform: amplitude_to_DB gives dB-scale values
        self.log_transform = T.AmplitudeToDB(stype="power", top_db=80)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "MelSpectrogramExtractor":
        """Construct from a ``features`` config dict."""
        return cls(
            sample_rate=cfg.get("sample_rate", 16000),
            n_mels=cfg.get("n_mels", 80),
            n_fft=cfg.get("n_fft", 512),
            hop_length=cfg.get("hop_length", 160),
            win_length=cfg.get("win_length", 400),
            fmin=cfg.get("fmin", 0.0),
            fmax=cfg.get("fmax", 8000.0),
            normalize=cfg.get("normalize", True),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract log-mel features.

        Parameters
        ----------
        waveform : Tensor [B, T_samples]
            Raw audio waveform at ``sample_rate``.
        lengths : Tensor [B]
            Number of valid samples per utterance.

        Returns
        -------
        features : Tensor [B, n_mels, T_frames]
            Log-mel spectrogram.
        feat_lengths : Tensor [B]
            Number of valid frames per utterance.
        """
        # Mel spectrogram: [B, n_mels, T_frames]
        mel = self.mel_spec(waveform)
        log_mel = self.log_transform(mel)

        # Compute output frame lengths from input sample lengths
        # torchaudio MelSpectrogram uses center=True by default:
        #   T_frames = 1 + floor(T_samples / hop_length)
        hop = self.mel_spec.hop_length
        feat_lengths = (lengths.float() / hop).floor().long() + 1

        # Clamp to actual feature length
        max_frames = log_mel.size(2)
        feat_lengths = feat_lengths.clamp(max=max_frames)

        if self.normalize:
            log_mel = self._cmvn(log_mel, feat_lengths)

        return log_mel, feat_lengths

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cmvn(
        features: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Per-utterance Cepstral Mean and Variance Normalization.

        Parameters
        ----------
        features : Tensor [B, C, T]
        lengths  : Tensor [B]

        Returns
        -------
        Tensor [B, C, T] — normalized features.
        """
        batch_size = features.size(0)
        for i in range(batch_size):
            valid_len = lengths[i].item()
            if valid_len > 1:
                segment = features[i, :, :valid_len]
                mean = segment.mean(dim=-1, keepdim=True)
                std = segment.std(dim=-1, keepdim=True).clamp(min=1e-6)
                features[i, :, :valid_len] = (segment - mean) / std
        return features
