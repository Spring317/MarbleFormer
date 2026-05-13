#!/usr/bin/env python3
"""
Single-file inference demo for VADASR.

Demonstrates the gated early exit: if no speech is detected,
returns "" immediately without running the Conformer decoder.

Usage:
    python scripts/inference.py --config configs/default.yaml \
        --checkpoint checkpoints/best.pt --audio_path test.wav

    python scripts/inference.py --config configs/default.yaml \
        --checkpoint checkpoints/best.pt --audio_dir wav/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torchaudio
import yaml
import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.models.vadasr_model import VADASRModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("inference")


def load_audio(path: str | Path, sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    """Load and preprocess a single audio file."""
    waveform, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # [T]
    return waveform, waveform.size(0)


def ctc_greedy_decode(log_probs: torch.Tensor, length: int, tokenizer) -> str:
    """Greedy CTC decoding."""
    log_probs = log_probs[:length]
    token_ids = log_probs.argmax(dim=-1).cpu().tolist()
    blank_id = tokenizer.blank_id
    cleaned = []
    prev = -1
    for tid in token_ids:
        if tid != prev and tid != blank_id:
            if 0 <= tid < tokenizer.vocab_size:
                cleaned.append(tid)
        prev = tid
    return tokenizer.decode(cleaned) if cleaned else ""


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate using jiwer."""
    if not reference.strip():
        return 0.0
    return jiwer.wer(reference, hypothesis)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate using jiwer."""
    if not reference.strip():
        return 0.0
    return jiwer.cer(reference, hypothesis)


def main() -> None:
    parser = argparse.ArgumentParser(description="VADASR Inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_path", type=str, default=None, help="Single audio file")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory of audio files")
    parser.add_argument("--threshold", type=float, default=None, help="Gate threshold override")
    parser.add_argument("--reference", type=str, default=None, help="Ground truth text or JSON manifest for WER/CER calculation")
    args = parser.parse_args()

    if not args.audio_path and not args.audio_dir:
        parser.error("Specify --audio_path or --audio_dir")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load tokenizer
    tokenizer = BPETokenizer.from_config(cfg["tokenizer"])

    # Load model
    model = VADASRModel.from_config(cfg, vocab_size=tokenizer.vocab_size)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    if args.threshold is not None:
        model.vad_gate.threshold = args.threshold

    logger.info("Model loaded (epoch %d)", ckpt["epoch"])

    # Load manifest if provided
    manifest_texts = {}
    if args.reference and args.reference.endswith(".json"):
        import json
        with open(args.reference, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                basename = Path(item["audio_filepath"]).name
                manifest_texts[basename] = item["text"]
        logger.info("Loaded %d references from manifest", len(manifest_texts))

    # Collect audio files
    audio_files: list[Path] = []
    if args.audio_path:
        audio_files.append(Path(args.audio_path))
    if args.audio_dir:
        exts = {".wav", ".flac", ".ogg", ".mp3"}
        audio_files.extend(
            p for p in sorted(Path(args.audio_dir).rglob("*"))
            if p.suffix.lower() in exts
        )

    logger.info("Processing %d file(s)...", len(audio_files))
    print("=" * 70)

    sample_rate = cfg["features"]["sample_rate"]

    all_refs = []
    all_hyps = []

    for audio_path in audio_files:
        waveform, wav_len = load_audio(audio_path, sample_rate)
        waveform = waveform.unsqueeze(0).to(device)  # [1, T]
        wav_lengths = torch.tensor([wav_len], dtype=torch.long, device=device)

        t0 = time.time()
        output = model.inference(waveform, wav_lengths)
        elapsed_ms = (time.time() - t0) * 1000

        has_voice = output.has_voice[0].item()
        gate_prob = torch.sigmoid(output.gate_logits[0]).item()

        if has_voice and output.ctc_log_probs is not None:
            text = ctc_greedy_decode(
                output.ctc_log_probs[0],
                output.ctc_lengths[0].item(),
                tokenizer,
            )
            status = "SPEECH"
        else:
            text = ""
            status = "SILENCE (early exit)"

        duration = wav_len / sample_rate
        rtf = (elapsed_ms / 1000) / duration

        print(
            f"[{status}] {audio_path.name} "
            f"(gate={gate_prob:.3f}, {elapsed_ms:.1f}ms, "
            f"RTF={rtf:.3f}, dur={duration:.1f}s)"
        )
        if text:
            print(f"  → Hypothesis: {text}")
        
        # Try to determine reference text
        ref_text = None
        if manifest_texts:
            ref_text = manifest_texts.get(audio_path.name)
        elif args.reference and len(audio_files) == 1 and not args.reference.endswith(".json"):
            ref_text = args.reference
        else:
            txt_path = audio_path.with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
        
        if ref_text is not None:
            print(f"  → Reference : {ref_text}")
            wer = calculate_wer(ref_text, text)
            cer = calculate_cer(ref_text, text)
            print(f"  → WER: {wer:.2%}, CER: {cer:.2%}")
            
            all_refs.append(ref_text)
            all_hyps.append(text)
        elif text:
            print() # add a newline if no reference was printed

    print("=" * 70)

    if all_refs:
        avg_wer = jiwer.wer(all_refs, all_hyps)
        avg_cer = jiwer.cer(all_refs, all_hyps)
        print(f"Average WER: {avg_wer:.2%}")
        print(f"Average CER: {avg_cer:.2%}")
        print("=" * 70)

    logger.info("Done.")


if __name__ == "__main__":
    main()
