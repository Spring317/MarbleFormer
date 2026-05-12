"""
BPE Tokenizer wrapper.

Interface Segregation: Exposes only the minimal encode/decode interface
needed by the training and inference pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import sentencepiece as spm


# ---------------------------------------------------------------------------
# Abstract interface (for Dependency Inversion)
# ---------------------------------------------------------------------------

class TokenizerProtocol(Protocol):
    """Minimal tokenizer contract used by the rest of the codebase."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def blank_id(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


# ---------------------------------------------------------------------------
# Concrete SentencePiece implementation
# ---------------------------------------------------------------------------

class BPETokenizer:
    """SentencePiece BPE tokenizer.

    Parameters
    ----------
    model_path : str | Path
        Path to a trained ``.model`` file.
    """

    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {model_path}"
            )
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(model_path))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "BPETokenizer":
        """Construct from a ``tokenizer`` config dict."""
        return cls(model_path=cfg["model_path"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (excluding CTC blank)."""
        return self._sp.GetPieceSize()

    @property
    def blank_id(self) -> int:
        """CTC blank token ID — by convention the last index."""
        return self._sp.GetPieceSize()

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Convert text to a list of token IDs.

        Parameters
        ----------
        text : str
            Input text string.

        Returns
        -------
        list[int]
            Token IDs. Empty list for empty text.
        """
        if not text or not text.strip():
            return []
        return self._sp.EncodeAsIds(text.strip())

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to text.

        Parameters
        ----------
        ids : list[int]
            Token IDs (blank tokens should be filtered before calling).

        Returns
        -------
        str
            Decoded text string.
        """
        # Filter out any stray blank IDs or out-of-range tokens
        valid_ids = [
            tid for tid in ids
            if 0 <= tid < self._sp.GetPieceSize()
        ]
        if not valid_ids:
            return ""
        return self._sp.DecodeIds(valid_ids)
