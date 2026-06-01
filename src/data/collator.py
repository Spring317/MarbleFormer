"""
Batch collator for VADASR — handles variable-length padding.

Single Responsibility: Pad waveforms and token sequences to uniform
batch dimensions.
"""

from __future__ import annotations

import torch


class VADASRCollator:
    """Collate variable-length samples into padded batches.

    Pads waveforms to the longest in the batch, pads token_ids
    for CTC loss computation, and stacks scalar fields.
    """

    def __call__(self, batch: list[dict]) -> dict:
        """
        Parameters
        ----------
        batch : list of dicts from VADASRDataset.__getitem__

        Returns
        -------
        dict with keys:
            waveform     : Tensor [B, max_T_samples]
            wav_lengths  : Tensor [B]
            token_ids    : Tensor [B, max_token_len]
            token_lengths: Tensor [B]
            has_voice    : Tensor [B] bool
            texts        : list[str]
        """
        waveforms = [s["waveform"] for s in batch]
        wav_lengths = torch.tensor(
            [s["wav_length"] for s in batch], dtype=torch.long
        )
        token_id_lists = [s["token_ids"] for s in batch]
        has_voice = torch.tensor(
            [s["has_voice"] for s in batch], dtype=torch.bool
        )
        texts = [s["text"] for s in batch]
        audio_paths = [s.get("audio_path", "") for s in batch]

        # Pad waveforms
        max_wav_len = max(w.size(0) for w in waveforms)
        padded_waveforms = torch.zeros(len(batch), max_wav_len)
        for i, w in enumerate(waveforms):
            padded_waveforms[i, :w.size(0)] = w

        # Pad token IDs
        token_lengths = torch.tensor(
            [len(t) for t in token_id_lists], dtype=torch.long
        )
        max_token_len = max(1, max(len(t) for t in token_id_lists))
        padded_tokens = torch.zeros(
            len(batch), max_token_len, dtype=torch.long
        )
        for i, t in enumerate(token_id_lists):
            if len(t) > 0:
                padded_tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)

        return {
            "waveform": padded_waveforms,
            "wav_lengths": wav_lengths,
            "token_ids": padded_tokens,
            "token_lengths": token_lengths,
            "has_voice": has_voice,
            "texts": texts,
            "audio_paths": audio_paths,
        }
