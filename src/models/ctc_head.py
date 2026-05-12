"""
CTC Head — linear projection to vocabulary for CTC decoding.

Single Responsibility: Project Conformer encoder output to token logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CTCHead(nn.Module):
    """CTC projection head.

    Parameters
    ----------
    encoder_dim : int
        Conformer encoder output dimension.
    vocab_size : int
        Number of tokens (excluding blank).
    blank_id : int
        CTC blank token index (default: vocab_size, i.e. last index).
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        vocab_size: int = 4000,
        blank_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_id = blank_id if blank_id is not None else vocab_size
        # Output: vocab_size + 1 (tokens + blank)
        self.projection = nn.Linear(encoder_dim, vocab_size + 1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    @classmethod
    def from_config(
        cls, cfg: dict, encoder_dim: int, vocab_size: int
    ) -> "CTCHead":
        return cls(
            encoder_dim=encoder_dim,
            vocab_size=vocab_size,
            blank_id=cfg.get("blank_id"),
        )

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Project to log-probabilities over vocabulary.

        Parameters
        ----------
        encoder_out : Tensor [B, T, encoder_dim]

        Returns
        -------
        log_probs : Tensor [B, T, vocab_size + 1]
        """
        return self.log_softmax(self.projection(encoder_out))
