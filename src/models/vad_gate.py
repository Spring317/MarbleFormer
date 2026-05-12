"""
VAD Gate — learnable binary gate for early exit.

Single Responsibility: Produce a speech probability from MarbleNet
encoder features. At inference time, if p < threshold → early exit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VADGate(nn.Module):
    """Binary voice activity detection gate.

    Takes the MarbleNet encoder output, applies global average pooling
    over the time dimension, and projects to a scalar speech probability.

    Parameters
    ----------
    input_dim : int
        Number of input channels from MarbleNet encoder.
    hidden_dim : int
        Hidden layer dimension.
    threshold : float
        Decision boundary for inference (default 0.5).
    temperature : float
        Sigmoid temperature scaling (default 1.0).
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def from_config(cls, cfg: dict, input_dim: int) -> "VADGate":
        return cls(
            input_dim=input_dim,
            hidden_dim=cfg.get("hidden_dim", 128),
            threshold=cfg.get("threshold", 0.5),
            temperature=cfg.get("temperature", 1.0),
        )

    def forward(
        self, encoder_out: torch.Tensor, feat_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute speech probability.

        Parameters
        ----------
        encoder_out : Tensor [B, C, T]
            MarbleNet encoder output.
        feat_lengths : Tensor [B]
            Valid frame counts.

        Returns
        -------
        prob : Tensor [B]
            Speech probability in [0, 1].
        """
        # Masked global average pooling
        batch_size = encoder_out.size(0)
        pooled = []
        for i in range(batch_size):
            valid_len = feat_lengths[i].item()
            valid_len = max(1, min(valid_len, encoder_out.size(2)))
            pooled.append(encoder_out[i, :, :valid_len].mean(dim=-1))
        pooled_tensor = torch.stack(pooled, dim=0)  # [B, C]

        logit = self.classifier(pooled_tensor).squeeze(-1)  # [B]
        prob = torch.sigmoid(logit / self.temperature)
        return prob

    def decide(self, prob: torch.Tensor) -> torch.Tensor:
        """Binary decision: is there speech?

        Returns
        -------
        Tensor [B] of bool — True if speech detected.
        """
        return prob >= self.threshold
