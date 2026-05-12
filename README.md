# VADASR — Gated Early Exit VAD-ASR Model

**MarbleNet (VAD) + Conformer (ASR) with CTC Decoder**

A unified model that combines voice activity detection with automatic speech recognition. The model uses a **gated early exit** mechanism: if the VAD gate detects no speech, the expensive Conformer decoder is skipped entirely — saving compute on silence/noise segments.

## Architecture

```
Audio → Mel Features → MarbleNet Encoder → VAD Gate ─┐
                                                       ├─ No Voice → "" (early exit)
                                                       └─ Voice → Conformer → CTC → Transcript
```

### Key Design Decisions

- **Dual-headed labels**: Each sample has `{"text": str, "has_voice": bool}` — the gate receives direct BCE supervision instead of learning from CTC blank ambiguity
- **Training**: Both branches always execute (for gradient flow). Loss = `λ_vad * BCE + λ_ctc * CTC`
- **Inference**: Gate makes a hard decision. Non-speech samples skip Conformer entirely

## Project Structure (SOLID)

```
vadasr/
├── configs/default.yaml          # All hyperparameters
├── src/
│   ├── features/mel_extractor.py # Mel spectrogram extraction
│   ├── models/
│   │   ├── marblenet_encoder.py  # 1D separable conv VAD encoder
│   │   ├── conformer_encoder.py  # Conformer ASR encoder
│   │   ├── vad_gate.py           # Binary gate (early exit)
│   │   ├── ctc_head.py           # CTC projection
│   │   └── vadasr_model.py       # Composed model (DI)
│   ├── data/
│   │   ├── dataset.py            # Unified BUD500 + noise dataset
│   │   ├── collator.py           # Batch padding
│   │   └── augmentation.py       # Pluggable augmentations
│   ├── training/
│   │   ├── loss.py               # Dual BCE + CTC loss
│   │   ├── trainer.py            # Training loop + checkpointing
│   │   └── scheduler.py          # Warmup + cosine annealing
│   ├── evaluation/
│   │   ├── metrics.py            # VAD F1, WER, RTF
│   │   └── evaluator.py          # Eval pipeline + threshold search
│   └── tokenizer/
│       └── bpe_tokenizer.py      # SentencePiece wrapper
└── scripts/
    ├── prepare_data.py           # Download BUD500 + generate manifests
    ├── train.py                  # Training entry point
    ├── evaluate.py               # Evaluation entry point
    └── inference.py              # Single-file inference demo
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
# Download BUD500 and generate manifests
python scripts/prepare_data.py --config configs/default.yaml

# For a quick test with limited samples:
python scripts/prepare_data.py --config configs/default.yaml --max_samples 1000
```

Place noise audio files in `data/noise/audio/` (WAV/FLAC/OGG format).

### 3. Train

```bash
# Full training
python scripts/train.py --config configs/default.yaml

# Debug mode (10 samples, 50 epochs)
python scripts/train.py --config configs/default.yaml --debug --max_samples 10 --max_epochs 50

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume checkpoints/best.pt
```

### 4. Evaluate

```bash
# Standard evaluation
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt

# With automatic threshold search
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt --threshold_search
```

### 5. Inference

```bash
# Single file
python scripts/inference.py --config configs/default.yaml --checkpoint checkpoints/best.pt --audio_path test.wav

# Directory of files
python scripts/inference.py --config configs/default.yaml --checkpoint checkpoints/best.pt --audio_dir wav/
```

## Dataset

- **Speech**: [BUD500](https://huggingface.co/datasets/linhtran92/viet_bud500) — ~500 hours Vietnamese ASR
- **Noise**: MUSAN + Freesound background noise (place in `data/noise/audio/`)

## Metrics

| Metric | Target |
|--------|--------|
| VAD F1 | ≥ 0.95 |
| WER | Comparable to standalone Conformer |
| Exit Rate (noise) | ≥ 90% |
| RTF Improvement | Measured vs always-decode baseline |

## License

Apache 2.0
