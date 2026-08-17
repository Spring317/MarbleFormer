#!/usr/bin/env python3
"""
Quantize VADASRModel for Vitis AI Xilinx KV260.

Usage within Vitis AI 3.0 PyTorch Docker:
    1. Calibrate (generate quantization config):
       python scripts/quantize_kv260.py --checkpoint best.pt --quant_mode calib
       
    2. Test & Export (generate .xmodel):
       python scripts/quantize_kv260.py --checkpoint best.pt --quant_mode test --deploy
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.dataset import VADASRDataset
from src.data.collator import VADASRCollator
from src.models.vadasr_model import VADASRModel
import torch.nn as nn

# Vitis AI Quantizer
try:
    from pytorch_nndct.apis import torch_quantizer
except ImportError:
    print("Error: pytorch_nndct not found. Please run this script inside the Vitis AI 3.0 PyTorch Docker environment.")
    sys.exit(1)


class VADWrapper(nn.Module):
    """
    Sub-graph 1: Feature Extraction + MarbleNet + VAD Gate.
    Runs on the DPU to evaluate if speech is present.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, waveform: torch.Tensor):
        # Static Mel Extraction (without dynamic length slicing loop)
        mel = self.model.mel_extractor.mel_spec(waveform)
        log_mel = self.model.mel_extractor.log_transform(mel)
        if self.model.mel_extractor.normalize:
            mean = log_mel.mean(dim=-1, keepdim=True)
            std = log_mel.std(dim=-1, keepdim=True).clamp(min=1e-6)
            log_mel = (log_mel - mean) / std
            
        marble_out, _ = self.model.marblenet(log_mel, None)
        gate_logits = self.model.vad_gate(marble_out, None)
        
        # Return both the gate decision and the intermediate feature 
        return gate_logits, marble_out


class ASRWrapper(nn.Module):
    """
    Sub-graph 2: QuartzNet + CTC.
    Runs on the DPU ONLY if the CPU reads speech from Sub-graph 1.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, marble_out: torch.Tensor):
        x, _ = self.model.quartznet(marble_out, None)
        ctc_log_probs = self.model.ctc_head(x)
        return ctc_log_probs


def evaluate_vad(model, val_loader, device, num_batches=100):
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            waveform = batch["waveform"].to(device)
            _ = model(waveform) 
            count += 1
            if count >= num_batches:
                break

def evaluate_asr(model, vad_model, val_loader, device, num_batches=100):
    model.eval()
    vad_model.eval()
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            waveform = batch["waveform"].to(device)
            _, marble_out = vad_model(waveform)
            _ = model(marble_out) 
            count += 1
            if count >= num_batches:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (e.g. best.pt)')
    parser.add_argument('--quant_mode', default='calib', choices=['calib', 'test'], 
                        help='Quantization mode: calib for calibration, test for evaluation and deploy')
    parser.add_argument('--deploy', action='store_true', help='Export xmodel for deployment')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for calibration and deploy (must be 1 for deploy)')
    args = parser.parse_args()

    # Assertions for deploy
    if args.deploy and args.batch_size != 1:
        print("Warning: Exporting xmodel needs batch size to be 1. Changing automatically.")
        args.batch_size = 1
    if args.deploy and args.quant_mode != 'test':
        print("Warning: Exporting xmodel must be done in 'test' quant_mode. Changing automatically.")
        args.quant_mode = 'test'

    device = torch.device("cpu") # Quantization in Vitis AI is typically run on CPU

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Load Tokenizer
    tokenizer = BPETokenizer.from_config(cfg["tokenizer"])

    # 2. Setup Dataset for Calibration
    data_cfg = cfg["data"]
    manifest_dir = Path(data_cfg.get("manifest_dir", "data/manifest"))
    manifest_file = manifest_dir / "combined_test.jsonl"
    if not manifest_file.exists():
        manifest_file = manifest_dir / "speech_test.jsonl"
        
    dataset_exists = manifest_file.exists()
    if dataset_exists:
        dataset = VADASRDataset.from_manifest(
            manifest=manifest_file,
            tokenizer=tokenizer,
            sample_rate=cfg["features"]["sample_rate"],
            max_audio_len_sec=data_cfg.get("max_audio_len_sec", 15.0),
            min_audio_len_sec=data_cfg.get("min_audio_len_sec", 0.5),
        )
        collator = VADASRCollator()
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    else:
        print(f"Warning: Manifest {manifest_file} not found. Using dummy data for calibration.")
        val_loader = None

    # 3. Load Model
    model = VADASRModel.from_config(cfg, vocab_size=tokenizer.vocab_size)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    # Split model into two wrappers for hardware-driven early exit
    vad_model = VADWrapper(model).to(device)
    asr_model = ASRWrapper(model).to(device)
    vad_model.eval()
    asr_model.eval()

    # 4. Dummy Input for Tracing
    sample_rate = cfg["features"]["sample_rate"]
    dummy_waveform = torch.randn([args.batch_size, sample_rate]).to(device) 
    
    with torch.no_grad():
        _, dummy_marble_out = vad_model(dummy_waveform)

    # 5. Initialize Torch Quantizers for both sub-graphs
    quantizer_vad = torch_quantizer(
        args.quant_mode, 
        vad_model, 
        (dummy_waveform,), 
        device=device
    )
    
    quantizer_asr = torch_quantizer(
        args.quant_mode, 
        asr_model, 
        (dummy_marble_out,), 
        device=device
    )

    # 6. Calibration/Forward Pass
    if dataset_exists:
        print("Calibrating VAD sub-graph...")
        evaluate_vad(quantizer_vad.quant_model, val_loader, device)
        print("Calibrating ASR sub-graph...")
        evaluate_asr(quantizer_asr.quant_model, vad_model, val_loader, device)
    else:
        # Dummy calibration passes
        for _ in range(10):
            wav = torch.randn([args.batch_size, sample_rate]).to(device)
            _ = quantizer_vad.quant_model(wav)
            _, feat = vad_model(wav)
            _ = quantizer_asr.quant_model(feat)
        print("Dummy calibration completed.")

    # 7. Export Outputs
    if args.quant_mode == 'calib':
        quantizer_vad.export_quant_config()
        quantizer_asr.export_quant_config()
        print("Calibration finished. Quantization configs exported.")
    
    if args.deploy:
        quantizer_vad.export_torch_script()
        quantizer_vad.export_xmodel()
        
        quantizer_asr.export_torch_script()
        quantizer_asr.export_xmodel()
        print("Both xmodels exported to quantize_result/. Ready for KV260 compilation!")

if __name__ == '__main__':
    main()
