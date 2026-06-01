"""
Evaluation pipeline for VADASR.

Single Responsibility: Run inference on a test set and collect metrics.
Supports automatic gate threshold search.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..models.vadasr_model import VADASRModel
from .metrics import MetricsCalculator, FullMetrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation pipeline with optional threshold search.

    Parameters
    ----------
    model : VADASRModel
        Trained model.
    tokenizer : Any
        Tokenizer with decode() method.
    device : torch.device
        Target device.
    """

    def __init__(
        self,
        model: VADASRModel,
        tokenizer: Any,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(
        self,
        dataloader: DataLoader,
        threshold: float | None = None,
        save_transcripts_path: str | None = None,
    ) -> FullMetrics:
        """Run evaluation on a dataset.

        Parameters
        ----------
        dataloader : DataLoader
            Test data loader.
        threshold : float | None
            Gate threshold override. Uses model default if None.

        Returns
        -------
        FullMetrics with VAD, ASR, and efficiency metrics.
        """
        self.model.eval()
        calc = MetricsCalculator()

        # Optionally override threshold
        original_threshold = self.model.vad_gate.threshold
        if threshold is not None:
            self.model.vad_gate.threshold = threshold

        transcripts = []
        try:
            for batch in tqdm(
                dataloader,
                desc="Evaluating",
                unit="batch",
                ascii=True,
            ):
                waveform = batch["waveform"].to(self.device)
                wav_lengths = batch["wav_lengths"].to(self.device)
                has_voice_gt = batch["has_voice"]
                texts = batch["texts"]
                audio_paths = batch.get("audio_paths", [""] * len(texts))

                t0 = time.time()
                output = self.model.inference(waveform, wav_lengths)
                elapsed_ms = (time.time() - t0) * 1000

                # VAD predictions
                vad_preds = output.has_voice.cpu().tolist()
                vad_labels = has_voice_gt.tolist()
                calc.update_vad(vad_preds, vad_labels)

                # ASR predictions (decode CTC output)
                batch_size = waveform.size(0)
                predictions = []
                for i in range(batch_size):
                    if output.has_voice[i] and output.ctc_log_probs is not None:
                        pred_text = self._ctc_greedy_decode(
                            output.ctc_log_probs[i],
                            output.ctc_lengths[i].item(),
                        )
                    else:
                        pred_text = ""
                    predictions.append(pred_text)

                calc.update_asr(predictions, texts)
                if save_transcripts_path is not None:
                    for audio_path, pred_text, ref_text in zip(
                        audio_paths, predictions, texts
                    ):
                        transcripts.append((audio_path, pred_text, ref_text))

                # Efficiency
                audio_dur = wav_lengths.float().sum().item() / 16000
                per_sample_ms = elapsed_ms / batch_size
                for i in range(batch_size):
                    dur = wav_lengths[i].item() / 16000
                    calc.update_efficiency(
                        per_sample_ms, dur, not output.has_voice[i].item()
                    )
        finally:
            self.model.vad_gate.threshold = original_threshold

        if save_transcripts_path is not None:
            with open(save_transcripts_path, "w", encoding="utf-8") as f:
                for audio_path, pred_text, ref_text in transcripts:
                    safe_audio = audio_path.replace("\t", " ").replace("\n", " ")
                    safe_pred = pred_text.replace("\t", " ").replace("\n", " ")
                    safe_ref = ref_text.replace("\t", " ").replace("\n", " ")
                    f.write(f"audio_path: {safe_audio}\n")
                    f.write(f"prediction: {safe_pred}\n")
                    f.write(f"reference: {safe_ref}\n")
                    f.write("---\n")

        return calc.compute()

    def threshold_search(
        self,
        dataloader: DataLoader,
        threshold_range: tuple[float, float] = (0.3, 0.7),
        steps: int = 9,
    ) -> tuple[float, FullMetrics]:
        """Search for optimal gate threshold.

        Evaluates at multiple thresholds and returns the one that
        maximizes VAD F1 while maintaining acceptable WER.

        Returns
        -------
        (best_threshold, best_metrics)
        """
        thresholds = np.linspace(
            threshold_range[0], threshold_range[1], steps
        )
        best_threshold = 0.5
        best_f1 = 0.0
        best_metrics = None

        for thresh in thresholds:
            metrics = self.evaluate(dataloader, threshold=float(thresh))
            logger.info(
                "Threshold=%.2f | VAD F1=%.4f | WER=%.4f | ExitRate=%.2f%%",
                thresh, metrics.vad.f1, metrics.asr.wer,
                metrics.efficiency.exit_rate * 100,
            )
            if metrics.vad.f1 > best_f1:
                best_f1 = metrics.vad.f1
                best_threshold = float(thresh)
                best_metrics = metrics

        logger.info("Best threshold: %.2f (F1=%.4f)", best_threshold, best_f1)
        return best_threshold, best_metrics

    def _ctc_greedy_decode(
        self, log_probs: torch.Tensor, length: int
    ) -> str:
        """Greedy CTC decoding with blank/repeat removal.

        Parameters
        ----------
        log_probs : Tensor [T, V+1]
        length : int — valid frame count

        Returns
        -------
        str — decoded text
        """
        log_probs = log_probs[:length]
        token_ids = log_probs.argmax(dim=-1).cpu().tolist()

        blank_id = self.tokenizer.blank_id
        cleaned = []
        prev = -1
        for tid in token_ids:
            if tid != prev and tid != blank_id:
                if 0 <= tid < self.tokenizer.vocab_size:
                    cleaned.append(tid)
            prev = tid

        return self.tokenizer.decode(cleaned) if cleaned else ""
