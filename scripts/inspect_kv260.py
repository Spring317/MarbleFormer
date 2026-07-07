#!/usr/bin/env python3
"""
Inspect the VADASRModel sub-graphs for Vitis AI Xilinx KV260.
This checks if the operations in your models are supported by the KV260 DPU.

Usage within Vitis AI 3.0 PyTorch Docker:
    python scripts/inspect_kv260.py --checkpoint best.pt
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.models.vadasr_model import VADASRModel
from scripts.quantize_kv260 import VADWrapper, ASRWrapper

try:
    from pytorch_nndct.apis import Inspector
except ImportError:
    print("Error: pytorch_nndct not found. Please run this inside Vitis AI Docker.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best.pt')
    parser.add_argument('--target', type=str, default='DPUCZDX8G_ISA1_B4096', help='DPU Target')
    args = parser.parse_args()

    device = torch.device("cpu")
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load Model
    tokenizer = BPETokenizer.from_config(cfg["tokenizer"])
    model = VADASRModel.from_config(cfg, vocab_size=tokenizer.vocab_size)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    # Create the two wrappers
    vad_model = VADWrapper(model).to(device)
    asr_model = ASRWrapper(model).to(device)
    vad_model.eval()
    asr_model.eval()

    # Dummy inputs
    sample_rate = cfg["features"]["sample_rate"]
    dummy_waveform = torch.randn([1, sample_rate]).to(device) 
    with torch.no_grad():
        _, dummy_marble_out = vad_model(dummy_waveform)

    # Initialize Inspector
    inspector = Inspector(args.target)

    # Inspect VAD
    print("Inspecting VAD Sub-graph (MarbleNet)...")
    inspector.inspect(vad_model, (dummy_waveform,), device=device, output_dir="inspect_vad")
    print("VAD inspection complete. Results saved to inspect_vad/")

    # Inspect ASR
    print("\nInspecting ASR Sub-graph (QuartzNet)...")
    inspector.inspect(asr_model, (dummy_marble_out,), device=device, output_dir="inspect_asr")
    print("ASR inspection complete. Results saved to inspect_asr/")

if __name__ == '__main__':
    main()
