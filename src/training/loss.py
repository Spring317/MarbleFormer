"""
Dual loss for VADASR — BCE (gate) + CTC (decoder).

Single Responsibility: Compute the combined loss with masking.
CTC loss is zeroed out for noise-only samples where has_voice=False.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VADASRLoss(nn.Module):
    """Combined VAD + ASR loss.

    Parameters
    ----------
    lambda_vad : float
        Weight for the VAD binary cross-entropy loss.
    lambda_ctc : float
        Weight for the CTC loss.
    blank_id : int
        CTC blank token index.
    """

    def __init__(
        self,
        lambda_vad: float = 1.0,
        lambda_ctc: float = 1.0,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        self.lambda_vad = lambda_vad
        self.lambda_ctc = lambda_ctc
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    @classmethod
    def from_config(cls, cfg: dict, blank_id: int) -> "VADASRLoss":
        return cls(
            lambda_vad=cfg.get("lambda_vad", 1.0),
            lambda_ctc=cfg.get("lambda_ctc", 1.0),
            blank_id=blank_id,
        )

    def forward(
        self,
        gate_logits: torch.Tensor,
        ctc_log_probs: torch.Tensor,
        ctc_lengths: torch.Tensor,
        token_ids: torch.Tensor,
        token_lengths: torch.Tensor,
        has_voice: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Parameters
        ----------
        gate_logits   : [B]         — predicted speech logits (pre-sigmoid)
        ctc_log_probs : [B, T, V+1] — CTC log probabilities
        ctc_lengths   : [B]         — valid CTC output lengths
        token_ids     : [B, S]      — target token sequences
        token_lengths : [B]         — target sequence lengths
        has_voice     : [B]         — ground truth VAD labels (bool)

        Returns
        -------
        dict with 'total', 'vad', 'ctc' loss tensors.
        """
        # --- VAD Loss (Binary Cross-Entropy with Logits) ---
        vad_target = has_voice.float()
        vad_loss = self.bce_loss(gate_logits, vad_target)

        # --- CTC Loss (masked for noise samples) ---
        voice_mask = has_voice
        if voice_mask.any():
            voice_indices = voice_mask.nonzero(as_tuple=True)[0]

            # CTC expects [T, B, V+1]
            ctc_input = ctc_log_probs[voice_indices].permute(1, 0, 2)
            ctc_target = token_ids[voice_indices]
            input_lengths = ctc_lengths[voice_indices]
            target_lengths = token_lengths[voice_indices]

            # Clamp lengths to valid range — critical for preventing NaN
            # input_lengths must not exceed actual T dimension of ctc_input
            max_T = ctc_input.size(0)
            input_lengths = input_lengths.clamp(min=1, max=max_T)

            # target_lengths must not exceed actual S dimension of ctc_target
            max_S = ctc_target.size(1)
            target_lengths = target_lengths.clamp(min=1, max=max_S)

            # CTC requires input_lengths >= target_lengths for valid alignment.
            # Clamp target_lengths to be at most input_lengths to prevent NaN.
            target_lengths = torch.min(target_lengths, input_lengths)

            # Ensure log_probs are valid (no -inf or NaN from log_softmax)
            # This can happen when softmax produces exact zeros in FP16
            ctc_input = ctc_input.clamp(min=-100.0)

            ctc_loss = self.ctc_loss(
                ctc_input, ctc_target,
                input_lengths, target_lengths,
            )

            # Final safety net: if CTC still produces NaN (e.g. from
            # degenerate length combinations), replace with zero
            if not torch.isfinite(ctc_loss):
                logger.warning(
                    "CTC loss is non-finite (%.4f), replacing with 0. "
                    "input_lengths range: [%d, %d], "
                    "target_lengths range: [%d, %d], max_T: %d",
                    ctc_loss.item(),
                    input_lengths.min().item(), input_lengths.max().item(),
                    target_lengths.min().item(), target_lengths.max().item(),
                    max_T,
                )
                ctc_loss = torch.tensor(0.0, device=gate_logits.device,
                                        requires_grad=True)
        else:
            ctc_loss = torch.tensor(0.0, device=gate_logits.device)

        total = self.lambda_vad * vad_loss + self.lambda_ctc * ctc_loss

        return {
            "total": total,
            "vad": vad_loss,
            "ctc": ctc_loss,
        }
