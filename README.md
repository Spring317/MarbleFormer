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
                                                 ctc head (1m)
                                                        │
                                                   transcript
```

**total: ~7m parameters**

### key design decisions

- **dual-headed labels**: each sample has `{"text": str, "has_voice": bool}` — the gate receives direct bce supervision instead of learning from ctc blank ambiguity
- **training**: both branches always execute (for gradient flow). loss = `λ_vad * bce + λ_ctc * ctc` (ctc is masked to 0 for noise samples)
- **inference**: gate makes a hard decision. non-speech samples skip conformer entirely
- **balanced data**: noise dataset is automatically generated/augmented to match the speech dataset in both sample count and total hours

## project structure (solid)

```
vadasr/
├── configs/
│   └── default.yaml              # all hyperparameters
├── src/
│   ├── features/
│   │   └── mel_extractor.py      # log-mel spectrogram with cmvn
│   ├── models/
│   │   ├── marblenet_encoder.py  # 1d depthwise separable conv blocks (3×2×64)
│   │   ├── conformer_encoder.py  # conformer + 4× conv subsampling
│   │   ├── vad_gate.py           # binary gate (pooling → mlp → sigmoid)
│   │   ├── ctc_head.py           # linear → logsoftmax projection
│   │   └── vadasr_model.py       # composed model with dependency injection
│   ├── data/
│   │   ├── dataset.py            # unified speech + noise dataset
│   │   ├── collator.py           # variable-length batch padding
│   │   └── augmentation.py       # pluggable speed perturb / noise / masking
│   ├── training/
│   │   ├── loss.py               # dual bce (gate) + masked ctc (decoder)
│   │   ├── trainer.py            # amp, grad accum, checkpointing, early stopping
│   │   └── scheduler.py          # warmup + cosine annealing
│   ├── evaluation/
│   │   ├── metrics.py            # vad f1, wer/cer, rtf, exit rate
│   │   └── evaluator.py          # eval pipeline + automatic threshold search
│   └── tokenizer/
│       └── bpe_tokenizer.py      # sentencepiece bpe wrapper
├── scripts/
│   ├── prepare_data.py           # data preparation + noise balancing
│   ├── train.py                  # training entry point
│   ├── evaluate.py               # evaluation entry point
│   ├── inference.py              # single-file inference demo
│   └── sanity_check.py           # verify model builds & gradients flow
├── requirements.txt
└── readme.md
```

## quick start

### 1. install dependencies

```bash
pip install -r requirements.txt
```

### 2. prepare data

the data pipeline expects a **locally available** bud500 dataset (no automatic download). background noise is sourced from freesound, curated public sound datasets (esc-50, fsd50k, arca23k, fsdnoisy18k), and/or synthetically generated to match the speech volume.

#### 2a. set up directory structure

place your downloaded bud500 dataset in the path specified by `data.bud500_local_dir` in the config (default: `data/bud500/`):

```
data/
├── bud500/
│   ├── train/          # *.wav, *.flac, *.parquet, or *.jsonl
│   └── test/
└── bpe_4000/
    └── bpe.model       # sentencepiece tokenizer
```

#### 2b. generate manifests (local bud500 only)

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

this will:
1. **scan** your local bud500 directory and create speech manifests (`speech_train.jsonl`, `speech_test.jsonl`)
2. **count** the speech samples and total hours
3. **generate synthetic noise** (white, pink, brown, babble, hum, fan) to match the speech volume in both sample count and total hours
4. **create** a balanced noise manifest (`noise.jsonl`)

#### 2c. (optional) download freesound background noise

if you want real-world noise instead of (or in addition to) synthetic noise:

```bash
python scripts/prepare_data.py --config configs/default.yaml \
    --download_freesound --freesound_api_key your_api_key
```

get an api key at [freesound.org/apiv2/apply](https://freesound.org/apiv2/apply/). this downloads noise across 30 categories (air-conditioner, rain, traffic, crowd, etc.) and resamples to 16khz mono wav.

you can customize the download:

```bash
# limit to 50 sounds per category
python scripts/prepare_data.py --config configs/default.yaml \
    --download_freesound --freesound_api_key your_key \
    --max_per_category 50

# use specific categories only
python scripts/prepare_data.py --config configs/default.yaml \
    --download_freesound --freesound_api_key your_key \
    --freesound_categories rain traffic wind crowd-noise

# skip resampling (if data is already 16khz wav)
python scripts/prepare_data.py --config configs/default.yaml \
    --download_freesound --freesound_api_key your_key \
    --skip_resample
```

#### 2d. (optional) download public sound datasets

you can also programmatically download public audio datasets to use as non-voice background noise. the pipeline will automatically fetch the data, filter out all human-activity classes (like speech, singing, crying, etc.), and resample the remaining clips to 16khz.

```bash
# download all 4 supported datasets (fsd50k, arca23k, fsdnoisy18k, esc-50)
python scripts/prepare_data.py --config configs/default.yaml \
    --download_datasets --cleanup_raw

# download only specific datasets
python scripts/prepare_data.py --config configs/default.yaml \
    --download_datasets --datasets esc50 fsdnoisy18k --cleanup_raw
```

*note: the `--cleanup_raw` flag deletes the heavy original `.zip` archives and temporary uncompressed files immediately after processing to save disk space. you can change where these datasets are downloaded and saved by updating `public_datasets_download_dir` and `public_datasets_noise_dir` under `data:` in `configs/default.yaml`.*

any remaining deficit (speech samples/hours not covered by freesound or public downloads) is filled with synthetic noise automatically.

#### 2e. quick test with limited samples

```bash
python scripts/prepare_data.py --config configs/default.yaml --max_samples 1000
```

#### 2f. force re-generation of manifests

```bash
python scripts/prepare_data.py --config configs/default.yaml --force
```

### 3. train

#### standard training

```bash
# full training
python scripts/train.py --config configs/default.yaml

# debug mode (small dataset, few epochs)
python scripts/train.py --config configs/default.yaml \
    --debug --max_samples 10 --max_epochs 50

# resume from checkpoint
python scripts/train.py --config configs/default.yaml \
    --resume checkpoints/best.pt
```

#### transfer learning with nemo weights

you can load pretrained weights from a nemo `.nemo`, `.ckpt`, or `.pth` file. the loader uses shape-based matching to transfer weights into the vadasr architecture. 

**critical:** the architecture specified in `configs/default.yaml` must exactly match the nemo model (e.g. `encoder_dim`, `num_layers`, `ffn_dim`).

```bash
# phase 1: load nemo weights, freeze conformer, train only marblenet (vad) + ctc
python scripts/train.py --config configs/default.yaml \
    --nemo_weights /path/to/nemo_weights.pth \
    --freeze_conformer

# phase 2: unfreeze conformer, train end-to-end (resume from phase 1)
python scripts/train.py --config configs/default.yaml \
    --resume checkpoints/best.pt
```

### 4. evaluate

```bash
# standard evaluation
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt

# with automatic gate threshold search
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --threshold_search

# override threshold manually
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --threshold 0.45
```

### 5. inference

```bash
# single file
python scripts/inference.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --audio_path test.wav

# directory of files
python scripts/inference.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --audio_dir wav/

# with custom gate threshold
python scripts/inference.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --audio_dir wav/ --threshold 0.4
```

example output:
```
[speech]  recording_001.wav (gate=0.987, 42.3ms, rtf=0.021, dur=2.0s)
  → ai cho phép em uống nhiều rượu như vậy

[silence (early exit)]  noise_clip.wav (gate=0.023, 1.2ms, rtf=0.001, dur=1.5s)
```

### 6. sanity check

verify the model builds correctly, forward pass runs, gradients flow, and early exit works:

```bash
python scripts/sanity_check.py
```

## dataset

| source | type | purpose |
|--------|------|---------|
| [bud500](https://huggingface.co/datasets/linhtran92/viet_bud500) | ~500h vietnamese speech | asr training (speech class) |
| [freesound](https://freesound.org/) | 30 noise categories | background noise (download optional) |
| [esc-50](https://github.com/karolpiczak/esc-50) | env. sound classification | background noise (filtered) |
| [fsd50k](https://zenodo.org/records/4060432) | freesound dataset 50k | background noise (filtered) |
| [arca23k](https://zenodo.org/records/5117901) | audio related clips | background noise (filtered) |
| [fsdnoisy18k](https://zenodo.org/records/2529934) | noisy audio dataset | background noise (filtered) |
| synthetic | white/pink/brown/babble/hum/fan | auto-generated to balance speech volume |

the data pipeline automatically balances noise to match speech:
- counts bud500 train samples and total hours
- if existing noise is insufficient, generates synthetic noise to fill the gap
- final noise manifest has ≥ same count and ≥ same hours as speech

## configuration

all hyperparameters are in `configs/default.yaml`. key sections:

| section | what it controls |
|---------|-----------------|
| `features` | mel spectrogram (n_mels, n_fft, hop, sample rate) |
| `marblenet` | vad encoder (blocks, channels, kernels, dropout) |
| `gate` | early exit threshold, temperature, hidden dim |
| `conformer` | asr encoder (layers, heads, dims, conv kernel) |
| `tokenizer` | sentencepiece model path and vocab size |
| `data` | dataset paths, speech/noise ratio, workers |
| `augmentation` | speed perturb, specaugment, noise mixing |
| `training` | batch size, lr, epochs, loss weights, checkpointing |
| `evaluation` | batch size, threshold search range |

## metrics

| metric | target |
|--------|--------|
| vad f1 | ≥ 0.95 |
| wer | comparable to standalone conformer |
| exit rate (noise) | ≥ 90% |
| rtf improvement | measured vs always-decode baseline |

## solid principles

| principle | where |
|-----------|-------|
| **s**ingle responsibility | each module has exactly one job |
| **o**pen/closed | `augmentationpipeline` accepts new transforms without modification |
| **l**iskov substitution | `vadasrdataset` implements standard `torch.utils.data.dataset` |
| **i**nterface segregation | `tokenizerprotocol` exposes only `encode`/`decode`/`vocab_size`/`blank_id` |
| **d**ependency inversion | `vadasrmodel` accepts all sub-modules via constructor injection |

## license

apache 2.0

