# VADASR — Gated Early Exit VAD-ASR Model

**MarbleNet (VAD) + Conformer (ASR) with CTC Decoder**

A unified model that combines voice activity detection with automatic speech recognition. The model uses a **gated early exit** mechanism: if the VAD gate detects no speech, the expensive Conformer decoder is skipped entirely — saving compute on silence/noise segments.

## Architecture

```
Audio (16kHz) → 80-band Log-Mel → MarbleNet Encoder (300K params)
                                        │
                                   VAD Gate (17K params)
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
                  p < threshold                   p ≥ threshold
                  (No Voice)                      (Voice Detected)
                        │                               │
                   Return ""                    Conformer Encoder (5.6M)
                 early exit                         │
                                                 CTC Head (1M)
                                                        │
                                                   Transcript
```

**Total: ~7M parameters**

### Key Design Decisions

- **Dual-headed labels**: Each sample has `{"text": str, "has_voice": bool}` — the gate receives direct BCE supervision instead of learning from CTC blank ambiguity
- **Training**: Both branches always execute (for gradient flow). Loss = `λ_vad * BCE + λ_ctc * CTC` (CTC is masked to 0 for noise samples)
- **Inference**: Gate makes a hard decision. Non-speech samples skip Conformer entirely
- **Balanced data**: Noise dataset is automatically generated/augmented to match the speech dataset in both sample count and total hours

## Project Structure (SOLID)

```
vadasr/
├── configs/
│   └── default.yaml              # All hyperparameters
├── src/
│   ├── features/
│   │   └── mel_extractor.py      # Log-mel spectrogram with CMVN
│   ├── models/
│   │   ├── marblenet_encoder.py  # 1D depthwise separable conv blocks (3×2×64)
│   │   ├── conformer_encoder.py  # Conformer + 4× conv subsampling
│   │   ├── vad_gate.py           # Binary gate (pooling → MLP → sigmoid)
│   │   ├── ctc_head.py           # Linear → LogSoftmax projection
│   │   └── vadasr_model.py       # Composed model with dependency injection
│   ├── data/
│   │   ├── dataset.py            # Unified speech + noise dataset
│   │   ├── collator.py           # Variable-length batch padding
│   │   └── augmentation.py       # Pluggable speed perturb / noise / masking
│   ├── training/
│   │   ├── loss.py               # Dual BCE (gate) + masked CTC (decoder)
│   │   ├── trainer.py            # AMP, grad accum, checkpointing, early stopping
│   │   └── scheduler.py          # Warmup + cosine annealing
│   ├── evaluation/
│   │   ├── metrics.py            # VAD F1, WER/CER, RTF, exit rate
│   │   └── evaluator.py          # Eval pipeline + automatic threshold search
│   └── tokenizer/
│       └── bpe_tokenizer.py      # SentencePiece BPE wrapper
├── scripts/
│   ├── prepare_data.py           # Data preparation + noise balancing
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation entry point
│   ├── inference.py              # Single-file inference demo
│   └── sanity_check.py           # Verify model builds & gradients flow
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

The data pipeline expects a **locally available** BUD500 dataset (no automatic download). Background noise is sourced from Freesound and/or synthetically generated to match the speech volume.

#### 2a. Set up directory structure

Place your downloaded BUD500 dataset in the path specified by `data.bud500_local_dir` in the config (default: `data/bud500/`):

```
data/
├── bud500/
│   ├── train/          # *.wav, *.flac, *.parquet, or *.jsonl
│   └── test/
└── bpe_4000/
    └── bpe.model       # SentencePiece tokenizer
```

#### 2b. Generate manifests (local BUD500 only)

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

This will:
1. **Scan** your local BUD500 directory and create speech manifests (`speech_train.jsonl`, `speech_test.jsonl`)
2. **Count** the speech samples and total hours
3. **Generate synthetic noise** (white, pink, brown, babble, hum, fan) to match the speech volume in both sample count and total hours
4. **Create** a balanced noise manifest (`noise.jsonl`)

#### 2c. (Optional) Download Freesound background noise

If you want real-world noise instead of (or in addition to) synthetic noise:

```bash
python scripts/prepare_data.py --config configs/default.yaml \
    --download_freesound --freesound_api_key YOUR_API_KEY
```

Get an API key at [freesound.org/apiv2/apply](https://freesound.org/apiv2/apply/). This downloads noise across 30 categories (air-conditioner, rain, traffic, crowd, etc.) and resamples to 16kHz mono WAV.

You can customize the download:

```bash
# Limit to 50 sounds per category
python scripts/prepare_data.py --config configs/default.yaml \
    --download_freesound --freesound_api_key YOUR_KEY \
    --max_per_category 50

# Use specific categories only
python scripts/prepare_data.py --config configs/default.yaml \
    --download_freesound --freesound_api_key YOUR_KEY \
    --freesound_categories rain traffic wind crowd-noise

# Skip resampling (if data is already 16kHz WAV)
python scripts/prepare_data.py --config configs/default.yaml \
    --download_freesound --freesound_api_key YOUR_KEY \
    --skip_resample
```

Any remaining deficit (speech samples/hours not covered by Freesound downloads) is filled with synthetic noise automatically.

#### 2d. Quick test with limited samples

```bash
python scripts/prepare_data.py --config configs/default.yaml --max_samples 1000
```

#### 2e. Force re-generation of manifests

```bash
python scripts/prepare_data.py --config configs/default.yaml --force
```

### 3. Train

#### Standard Training

```bash
# Full training
python scripts/train.py --config configs/default.yaml

# Debug mode (small dataset, few epochs)
python scripts/train.py --config configs/default.yaml \
    --debug --max_samples 10 --max_epochs 50

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml \
    --resume checkpoints/best.pt
```

#### Transfer Learning with NeMo Weights

You can load pretrained weights from a NeMo `.nemo`, `.ckpt`, or `.pth` file. The loader uses shape-based matching to transfer weights into the VADASR architecture. 

**CRITICAL:** The architecture specified in `configs/default.yaml` MUST exactly match the NeMo model (e.g. `encoder_dim`, `num_layers`, `ffn_dim`).

```bash
# Phase 1: Load NeMo weights, freeze Conformer, train only MarbleNet (VAD) + CTC
python scripts/train.py --config configs/default.yaml \
    --nemo_weights /path/to/nemo_weights.pth \
    --freeze_conformer

# Phase 2: Unfreeze Conformer, train end-to-end (resume from Phase 1)
python scripts/train.py --config configs/default.yaml \
    --resume checkpoints/best.pt
```

### 4. Evaluate

```bash
# Standard evaluation
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt

# With automatic gate threshold search
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --threshold_search

# Override threshold manually
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --threshold 0.45
```

### 5. Inference

```bash
# Single file
python scripts/inference.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --audio_path test.wav

# Directory of files
python scripts/inference.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --audio_dir wav/

# With custom gate threshold
python scripts/inference.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --audio_dir wav/ --threshold 0.4
```

Example output:
```
[SPEECH]  recording_001.wav (gate=0.987, 42.3ms, RTF=0.021, dur=2.0s)
  → ai cho phép em uống nhiều rượu như vậy

[SILENCE (early exit)]  noise_clip.wav (gate=0.023, 1.2ms, RTF=0.001, dur=1.5s)
```

### 6. Sanity check

Verify the model builds correctly, forward pass runs, gradients flow, and early exit works:

```bash
python scripts/sanity_check.py
```

## Dataset

| Source | Type | Purpose |
|--------|------|---------|
| [BUD500](https://huggingface.co/datasets/linhtran92/viet_bud500) | ~500h Vietnamese speech | ASR training (speech class) |
| [Freesound](https://freesound.org/) | 30 noise categories | Background noise (download optional) |
| Synthetic | White/pink/brown/babble/hum/fan | Auto-generated to balance speech volume |

The data pipeline automatically balances noise to match speech:
- Counts BUD500 train samples and total hours
- If existing noise is insufficient, generates synthetic noise to fill the gap
- Final noise manifest has ≥ same count and ≥ same hours as speech

## Configuration

All hyperparameters are in `configs/default.yaml`. Key sections:

| Section | What it controls |
|---------|-----------------|
| `features` | Mel spectrogram (n_mels, n_fft, hop, sample rate) |
| `marblenet` | VAD encoder (blocks, channels, kernels, dropout) |
| `gate` | Early exit threshold, temperature, hidden dim |
| `conformer` | ASR encoder (layers, heads, dims, conv kernel) |
| `tokenizer` | SentencePiece model path and vocab size |
| `data` | Dataset paths, speech/noise ratio, workers |
| `augmentation` | Speed perturb, SpecAugment, noise mixing |
| `training` | Batch size, LR, epochs, loss weights, checkpointing |
| `evaluation` | Batch size, threshold search range |

## Metrics

| Metric | Target |
|--------|--------|
| VAD F1 | ≥ 0.95 |
| WER | Comparable to standalone Conformer |
| Exit Rate (noise) | ≥ 90% |
| RTF Improvement | Measured vs always-decode baseline |

## SOLID Principles

| Principle | Where |
|-----------|-------|
| **S**ingle Responsibility | Each module has exactly one job |
| **O**pen/Closed | `AugmentationPipeline` accepts new transforms without modification |
| **L**iskov Substitution | `VADASRDataset` implements standard `torch.utils.data.Dataset` |
| **I**nterface Segregation | `TokenizerProtocol` exposes only `encode`/`decode`/`vocab_size`/`blank_id` |
| **D**ependency Inversion | `VADASRModel` accepts all sub-modules via constructor injection |

## License

Apache 2.0
