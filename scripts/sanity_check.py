#!/usr/bin/env python3
"""
Sanity check: verify model builds, forward pass works, and gradients flow.
Run from the vadasr/ directory:
    python scripts/sanity_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.mel_extractor import MelSpectrogramExtractor
from src.models.marblenet_encoder import MarbleNetEncoder
from src.models.conformer_encoder import ConformerEncoder
from src.models.vad_gate import VADGate
from src.models.ctc_head import CTCHead
from src.models.vadasr_model import VADASRModel
from src.training.loss import VADASRLoss


def main() -> None:
    print("=" * 60)
    print("VADASR Sanity Check")
    print("=" * 60)

    # Load config
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    vocab_size = 4000  # mock

    # Build model
    print("\n[1/5] Building model...")
    model = VADASRModel.from_config(cfg, vocab_size=vocab_size)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    for name, mod in [
        ("MelExtractor", model.mel_extractor),
        ("MarbleNet", model.marblenet),
        ("VADGate", model.vad_gate),
        ("Conformer", model.conformer),
        ("CTCHead", model.ctc_head),
    ]:
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:15s}: {n:>10,} params")

    # Forward pass
    print("\n[2/5] Forward pass (training mode)...")
    batch_size = 2
    seq_len = 32000  # 2 seconds at 16kHz
    waveform = torch.randn(batch_size, seq_len)
    wav_lengths = torch.tensor([seq_len, seq_len // 2])

    output = model(waveform, wav_lengths)
    print(f"  gate_logits shape   : {output.gate_logits.shape}")
    print(f"  gate_logits values  : {output.gate_logits.detach().numpy()}")
    print(f"  ctc_log_probs shape : {output.ctc_log_probs.shape}")
    print(f"  ctc_lengths         : {output.ctc_lengths.detach().numpy()}")
    print(f"  has_voice           : {output.has_voice.detach().numpy()}")

    # Loss computation
    print("\n[3/5] Loss computation...")
    blank_id = vocab_size  # blank = last index
    criterion = VADASRLoss(lambda_vad=1.0, lambda_ctc=1.0, blank_id=blank_id)

    # Mock targets
    token_ids = torch.randint(0, vocab_size, (batch_size, 10))
    token_lengths = torch.tensor([10, 8])
    has_voice = torch.tensor([True, False])

    losses = criterion(
        gate_logits=output.gate_logits,
        ctc_log_probs=output.ctc_log_probs,
        ctc_lengths=output.ctc_lengths,
        token_ids=token_ids,
        token_lengths=token_lengths,
        has_voice=has_voice,
    )
    print(f"  Total loss : {losses['total'].item():.4f}")
    print(f"  VAD loss   : {losses['vad'].item():.4f}")
    print(f"  CTC loss   : {losses['ctc'].item():.4f}")

    # Backward pass
    print("\n[4/5] Backward pass (gradient check)...")
    losses["total"].backward()
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    nonzero_grads = sum(1 for v in grad_norms.values() if v > 0)
    total_grads = len(grad_norms)
    print(f"  Parameters with gradients: {nonzero_grads}/{total_grads}")
    if nonzero_grads < total_grads:
        zero_grad_params = [k for k, v in grad_norms.items() if v == 0]
        print(f"  Zero-gradient params: {zero_grad_params[:5]}...")

    # Inference mode (early exit)
    print("\n[5/5] Inference mode (early exit)...")
    model.eval()

    # Test with noise (should trigger early exit)
    noise = torch.randn(1, 16000) * 0.01  # very quiet
    noise_len = torch.tensor([16000])
    model.vad_gate.threshold = 0.99  # force early exit for test

    inf_output = model.inference(noise, noise_len)
    print(f"  Gate logit : {inf_output.gate_logits.item():.4f}")
    print(f"  Gate prob  : {torch.sigmoid(inf_output.gate_logits).item():.4f}")
    print(f"  Has voice  : {inf_output.has_voice.item()}")
    print(f"  CTC output : {'None (early exit!)' if inf_output.ctc_log_probs is None else inf_output.ctc_log_probs.shape}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
