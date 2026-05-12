"""
Metrics calculator for VADASR.

Single Responsibility: Compute VAD (precision/recall/F1),
ASR (WER/CER), and efficiency (RTF, exit rate) metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jiwer


@dataclass
class VADMetrics:
    """Voice Activity Detection metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0


@dataclass
class ASRMetrics:
    """Automatic Speech Recognition metrics."""
    wer: float = 0.0
    cer: float = 0.0
    num_samples: int = 0


@dataclass
class EfficiencyMetrics:
    """Inference efficiency metrics."""
    avg_inference_ms: float = 0.0
    exit_rate: float = 0.0  # fraction of samples that exit early
    rtf: float = 0.0        # real-time factor


@dataclass
class FullMetrics:
    """Combined evaluation metrics."""
    vad: VADMetrics = field(default_factory=VADMetrics)
    asr: ASRMetrics = field(default_factory=ASRMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)


class MetricsCalculator:
    """Stateful metrics accumulator."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._vad_preds: list[bool] = []
        self._vad_labels: list[bool] = []
        self._asr_preds: list[str] = []
        self._asr_refs: list[str] = []
        self._inference_times: list[float] = []
        self._audio_durations: list[float] = []
        self._early_exits: int = 0
        self._total_samples: int = 0

    def update_vad(
        self, predictions: list[bool], labels: list[bool]
    ) -> None:
        self._vad_preds.extend(predictions)
        self._vad_labels.extend(labels)

    def update_asr(
        self, predictions: list[str], references: list[str]
    ) -> None:
        # Only compare non-empty references (speech samples)
        for pred, ref in zip(predictions, references):
            if ref.strip():
                self._asr_preds.append(pred)
                self._asr_refs.append(ref)

    def update_efficiency(
        self,
        inference_time_ms: float,
        audio_duration_sec: float,
        exited_early: bool,
    ) -> None:
        self._inference_times.append(inference_time_ms)
        self._audio_durations.append(audio_duration_sec)
        self._total_samples += 1
        if exited_early:
            self._early_exits += 1

    def compute(self) -> FullMetrics:
        """Compute all accumulated metrics."""
        return FullMetrics(
            vad=self._compute_vad(),
            asr=self._compute_asr(),
            efficiency=self._compute_efficiency(),
        )

    def _compute_vad(self) -> VADMetrics:
        if not self._vad_preds:
            return VADMetrics()

        preds = np.array(self._vad_preds)
        labels = np.array(self._vad_labels)

        tp = ((preds == True) & (labels == True)).sum()
        fp = ((preds == True) & (labels == False)).sum()
        fn = ((preds == False) & (labels == True)).sum()

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2 * precision * recall / max(1e-8, precision + recall)
        )
        accuracy = (preds == labels).mean()

        return VADMetrics(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
        )

    def _compute_asr(self) -> ASRMetrics:
        if not self._asr_preds or not self._asr_refs:
            return ASRMetrics()

        wer = jiwer.wer(self._asr_refs, self._asr_preds)
        cer = jiwer.cer(self._asr_refs, self._asr_preds)

        return ASRMetrics(
            wer=float(wer),
            cer=float(cer),
            num_samples=len(self._asr_refs),
        )

    def _compute_efficiency(self) -> EfficiencyMetrics:
        if not self._inference_times:
            return EfficiencyMetrics()

        avg_ms = float(np.mean(self._inference_times))
        total_infer_s = float(np.sum(self._inference_times)) / 1000.0
        total_audio_s = float(np.sum(self._audio_durations))
        rtf = total_infer_s / max(1e-6, total_audio_s)
        exit_rate = self._early_exits / max(1, self._total_samples)

        return EfficiencyMetrics(
            avg_inference_ms=avg_ms,
            exit_rate=float(exit_rate),
            rtf=float(rtf),
        )
