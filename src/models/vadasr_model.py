"""
VADASR Model — Composed gated early exit model.

Dependency Inversion: All sub-modules are injected via constructor.
Open/Closed: New encoders or gate strategies can be swapped without
modifying this class.

Architecture:
  MelExtractor → MarbleNet → VADGate → [early exit] → Conformer → CTCHead
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..features.mel_extractor import MelSpectrogramExtractor
from .marblenet_encoder import MarbleNetEncoder
from .conformer_encoder import ConformerEncoder
from .vad_gate import VADGate
from .ctc_head import CTCHead


@dataclass
class VADASROutput:
    """Structured output from the VADASR model."""

    gate_prob: torch.Tensor         # [B] — speech probability
    ctc_log_probs: Optional[torch.Tensor]  # [B, T, V+1] or None
    ctc_lengths: Optional[torch.Tensor]    # [B] or None
    has_voice: torch.Tensor         # [B] bool — gate decisions


class VADASRModel(nn.Module):
    """Gated early exit VAD-ASR model.

    Composes MarbleNet (VAD encoder) + VAD Gate + Conformer (ASR encoder)
    + CTC Head into a unified model with early exit capability.

    Parameters
    ----------
    mel_extractor : MelSpectrogramExtractor
        Audio feature extractor.
    marblenet : MarbleNetEncoder
        VAD encoder (1D separable conv blocks).
    vad_gate : VADGate
        Binary gate for early exit decision.
    conformer : ConformerEncoder
        ASR encoder (self-attention + conv blocks).
    ctc_head : CTCHead
        CTC projection layer.
    """

    def __init__(
        self,
        mel_extractor: MelSpectrogramExtractor,
        marblenet: MarbleNetEncoder,
        vad_gate: VADGate,
        conformer: ConformerEncoder,
        ctc_head: CTCHead,
    ) -> None:
        super().__init__()
        self.mel_extractor = mel_extractor
        self.marblenet = marblenet
        self.vad_gate = vad_gate
        self.conformer = conformer
        self.ctc_head = ctc_head

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict, vocab_size: int) -> "VADASRModel":
        """Build the full model from a config dict."""
        mel = MelSpectrogramExtractor.from_config(cfg["features"])
        marble = MarbleNetEncoder.from_config(cfg["marblenet"])
        gate = VADGate.from_config(cfg["gate"], input_dim=marble.output_channels)
        conformer = ConformerEncoder.from_config(cfg["conformer"])
        ctc = CTCHead.from_config(
            cfg["ctc"],
            encoder_dim=conformer.encoder_dim,
            vocab_size=vocab_size,
        )
        return cls(mel, marble, gate, conformer, ctc)

    # ------------------------------------------------------------------
    # Training forward (both branches always execute)
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,
        wav_lengths: torch.Tensor,
    ) -> VADASROutput:
        """Training forward pass — both branches execute for gradient flow.

        Parameters
        ----------
        waveform : Tensor [B, T_samples]
            Raw audio waveforms.
        wav_lengths : Tensor [B]
            Number of valid samples per utterance.

        Returns
        -------
        VADASROutput with gate_prob, ctc_log_probs, ctc_lengths, has_voice.
        """
        # Feature extraction
        mel_features, feat_lengths = self.mel_extractor(waveform, wav_lengths)

        # MarbleNet encoder (VAD front-end)
        marble_out, marble_lengths = self.marblenet(mel_features, feat_lengths)

        # VAD gate
        gate_prob = self.vad_gate(marble_out, marble_lengths)
        has_voice = self.vad_gate.decide(gate_prob)

        # Conformer encoder (always runs during training)
        conformer_out, ctc_lengths = self.conformer(marble_out, marble_lengths)

        # CTC head
        ctc_log_probs = self.ctc_head(conformer_out)

        return VADASROutput(
            gate_prob=gate_prob,
            ctc_log_probs=ctc_log_probs,
            ctc_lengths=ctc_lengths,
            has_voice=has_voice,
        )

    # ------------------------------------------------------------------
    # Inference forward (gated early exit)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def inference(
        self,
        waveform: torch.Tensor,
        wav_lengths: torch.Tensor,
    ) -> VADASROutput:
        """Inference forward pass with gated early exit.

        If the gate predicts no speech, the Conformer and CTC head are
        skipped entirely — saving compute.

        Parameters
        ----------
        waveform : Tensor [B, T_samples]
        wav_lengths : Tensor [B]

        Returns
        -------
        VADASROutput — ctc_log_probs is None for non-speech samples.
        """
        mel_features, feat_lengths = self.mel_extractor(waveform, wav_lengths)
        marble_out, marble_lengths = self.marblenet(mel_features, feat_lengths)
        gate_prob = self.vad_gate(marble_out, marble_lengths)
        has_voice = self.vad_gate.decide(gate_prob)

        # Early exit: if no sample in the batch has voice
        if not has_voice.any():
            return VADASROutput(
                gate_prob=gate_prob,
                ctc_log_probs=None,
                ctc_lengths=None,
                has_voice=has_voice,
            )

        # Only process samples with detected voice
        voice_mask = has_voice
        voice_indices = voice_mask.nonzero(as_tuple=True)[0]

        marble_voice = marble_out[voice_indices]
        lengths_voice = marble_lengths[voice_indices]

        conformer_out, ctc_lengths = self.conformer(
            marble_voice, lengths_voice
        )
        ctc_log_probs = self.ctc_head(conformer_out)

        # Reconstruct full-batch output (None for non-speech)
        batch_size = waveform.size(0)
        max_t = ctc_log_probs.size(1)
        vocab_dim = ctc_log_probs.size(2)

        full_ctc = torch.zeros(
            batch_size, max_t, vocab_dim,
            device=waveform.device, dtype=ctc_log_probs.dtype,
        )
        full_lengths = torch.zeros(
            batch_size, device=waveform.device, dtype=torch.long
        )

        full_ctc[voice_indices] = ctc_log_probs
        full_lengths[voice_indices] = ctc_lengths

        return VADASROutput(
            gate_prob=gate_prob,
            ctc_log_probs=full_ctc,
            ctc_lengths=full_lengths,
            has_voice=has_voice,
        )
