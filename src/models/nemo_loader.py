"""
NeMo Conformer/FastConformer weight loader for VADASR.

Loads pretrained NeMo ASR model weights (.nemo, .ckpt, or .pth) into the
VADASR model's Conformer encoder and CTC head.

Supports two NeMo model families:
  - Conformer-CTC-BPE (d_model=176, n_layers=16, n_heads=4, conv_kernel=31)
  - FastConformer-CTC-BPE (d_model=512, n_layers=18, n_heads=8, conv_kernel=9)

The loader performs:
  1. Extraction of state_dict from .nemo tar / .ckpt / .pth files
  2. Auto-detection of the NeMo key prefix
  3. Diagnostic logging of what was loaded, skipped, and shape-mismatched
  4. Partial loading of Conformer layers (if layer count differs)
  5. Optional CTC head loading (if vocab size matches)

Usage:
    python scripts/train.py --config configs/default.yaml \\
        --nemo_weights path/to/conformer.nemo

    # Or programmatically:
    from src.models.nemo_loader import load_nemo_weights
    load_nemo_weights(model, "path/to/model.nemo", device="cuda")
"""

from __future__ import annotations

import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Step 1: Extract state_dict from various NeMo file formats
# ============================================================================

def _extract_nemo_checkpoint(nemo_path: str | Path) -> dict[str, Any]:
    """Extract state_dict from a .nemo archive, .ckpt, or .pth file.

    .nemo files are tar archives containing model_weights.ckpt.
    .ckpt files are PyTorch Lightning checkpoints.
    .pth files are raw PyTorch state dicts.

    Returns
    -------
    state_dict : dict
        The raw state dictionary from the NeMo model.
    """
    nemo_path = Path(nemo_path)

    if nemo_path.suffix == ".nemo":
        # .nemo is a tar archive
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(nemo_path, "r") as tar:
                tar.extractall(tmpdir)

            # Look for model_weights.ckpt or model_weights.pth
            ckpt_path = None
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f in ("model_weights.ckpt", "model_weights.pth"):
                        ckpt_path = os.path.join(root, f)
                        break
                if ckpt_path:
                    break

            if ckpt_path is None:
                raise FileNotFoundError(
                    f"No model_weights.ckpt/pth found inside {nemo_path}"
                )

            ckpt = torch.load(ckpt_path, map_location="cpu")

    elif nemo_path.suffix in (".ckpt", ".pt", ".pth"):
        ckpt = torch.load(nemo_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported file format: {nemo_path.suffix}")

    # PyTorch Lightning wraps state_dict under 'state_dict' key
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


# ============================================================================
# Step 2: Detect the NeMo model architecture from the state_dict
# ============================================================================

def _detect_nemo_architecture(
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Detect NeMo model architecture from state_dict keys and shapes.

    Returns
    -------
    dict with:
        - encoder_prefix: str (e.g., "encoder." or "model.encoder.")
        - decoder_prefix: str
        - n_layers: int
        - d_model: int
        - n_heads: int (estimated)
        - model_type: str ("conformer" or "fast_conformer")
    """
    info: dict[str, Any] = {
        "encoder_prefix": "",
        "decoder_prefix": "",
        "n_layers": 0,
        "d_model": 0,
        "n_heads": 0,
        "model_type": "unknown",
    }

    # Detect encoder prefix
    for prefix in ["encoder.", "model.encoder."]:
        if any(k.startswith(prefix) for k in state):
            info["encoder_prefix"] = prefix
            break

    # Detect decoder prefix
    for prefix in ["decoder.", "model.decoder."]:
        if any(k.startswith(prefix) for k in state):
            info["decoder_prefix"] = prefix
            break

    if not info["encoder_prefix"]:
        logger.warning("Could not detect encoder prefix in state_dict")
        return info

    # Count layers
    layer_indices = set()
    ep = info["encoder_prefix"]
    for key in state:
        if key.startswith(ep + "layers."):
            parts = key[len(ep):].split(".")
            try:
                layer_indices.add(int(parts[1]))
            except (IndexError, ValueError):
                pass
    info["n_layers"] = len(layer_indices)

    # Detect d_model from a layer norm weight shape
    for key, val in state.items():
        if key.startswith(ep + "layers.0.") and "norm" in key and "weight" in key:
            if val.dim() == 1:
                info["d_model"] = val.shape[0]
                break

    # Infer model type from d_model and layer count
    d = info["d_model"]
    n = info["n_layers"]
    if d == 176 and n == 16:
        info["model_type"] = "conformer_small"
        info["n_heads"] = 4
    elif d == 256 and n == 16:
        info["model_type"] = "conformer_medium"
        info["n_heads"] = 4
    elif d == 512 and n in (17, 18):
        info["model_type"] = "fast_conformer"
        info["n_heads"] = 8
    elif d == 1024 and n == 24:
        info["model_type"] = "fast_conformer_xlarge"
        info["n_heads"] = 8
    else:
        info["model_type"] = f"unknown_d{d}_n{n}"

    logger.info(
        "Detected NeMo architecture: %s (d_model=%d, n_layers=%d, n_heads=%d)",
        info["model_type"], info["d_model"], info["n_layers"], info["n_heads"],
    )

    return info


# ============================================================================
# Step 3: Load Conformer layers by key matching within each layer
# ============================================================================

def _load_conformer_layers(
    nemo_state: dict[str, torch.Tensor],
    target_conformer: nn.Module,
    nemo_prefix: str,
    num_target_layers: int,
) -> tuple[int, int, list[str]]:
    """Load NeMo encoder layer weights into NeMo-compatible Conformer layers.

    Since the target Conformer mirrors NeMo's architecture and naming,
    this performs direct key mapping:

        NeMo:   {encoder_prefix}layers.{i}.{sub_key}
        Target: conformer_layers.{i}.{sub_key}

    No shape-based guessing — keys are mapped by name, with shape
    validation as a safety check.

    Parameters
    ----------
    nemo_state : dict
        Full NeMo state_dict.
    target_conformer : nn.Module
        The NeMoConformer module (model.conformer.conformer).
    nemo_prefix : str
        Key prefix for encoder params (e.g., "encoder.").
    num_target_layers : int
        Number of layers in the target model.

    Returns
    -------
    (loaded_count, total_target_params, skipped_keys)
    """
    target_state = target_conformer.state_dict()

    # Group NeMo params by layer index
    nemo_by_layer: dict[int, dict[str, torch.Tensor]] = {}
    layers_prefix = nemo_prefix + "layers."

    for key, val in nemo_state.items():
        if not key.startswith(layers_prefix):
            continue
        relative = key[len(layers_prefix):]
        parts = relative.split(".", 1)
        try:
            layer_idx = int(parts[0])
            sub_key = parts[1]
            nemo_by_layer.setdefault(layer_idx, {})[sub_key] = val
        except (IndexError, ValueError):
            pass

    n_nemo = len(nemo_by_layer)
    n_load = min(n_nemo, num_target_layers)

    if n_nemo != num_target_layers:
        logger.warning(
            "Layer count mismatch: NeMo has %d layers, target has %d. "
            "Loading first %d layers.", n_nemo, num_target_layers, n_load,
        )

    loaded_count = 0
    skipped_keys: list[str] = []
    updates: dict[str, torch.Tensor] = {}

    nemo_layer_indices = sorted(nemo_by_layer.keys())[:n_load]

    for i, src_idx in enumerate(nemo_layer_indices):
        src_params = nemo_by_layer[src_idx]

        for sub_key, nemo_val in sorted(src_params.items()):
            # NeMo sometimes uses 'conv.' instead of 'conv_module.'
            mapped_sub_key = sub_key
            if mapped_sub_key.startswith("conv."):
                mapped_sub_key = "conv_module." + mapped_sub_key[5:]

            target_key = f"conformer_layers.{i}.{mapped_sub_key}"

            if target_key not in target_state:
                skipped_keys.append(
                    f"nemo layer {src_idx}.{sub_key} (no matching target key)"
                )
                continue

            target_val = target_state[target_key]
            if nemo_val.shape != target_val.shape:
                skipped_keys.append(
                    f"nemo layer {src_idx}.{sub_key} "
                    f"(shape mismatch: nemo {list(nemo_val.shape)} "
                    f"vs target {list(target_val.shape)})"
                )
                continue

            updates[target_key] = nemo_val
            loaded_count += 1

    # Apply all matched weights at once
    if updates:
        target_state.update(updates)
        target_conformer.load_state_dict(target_state, strict=False)

    total_target = sum(
        1 for _ in target_conformer.parameters()
    ) + sum(1 for _ in target_conformer.buffers())

    return loaded_count, total_target, skipped_keys


# ============================================================================
# Step 4: Load CTC head weights
# ============================================================================

def _load_ctc_head(
    nemo_state: dict[str, torch.Tensor],
    ctc_head: nn.Module,
    nemo_prefix: str,
) -> tuple[int, list[str]]:
    """Load NeMo decoder (CTC linear head) weights.

    NeMo CTC decoder keys:
        decoder.decoder_layers.0.weight  [vocab_size, d_model]
        decoder.decoder_layers.0.bias    [vocab_size]

    Our CTCHead keys:
        projection.weight  [vocab_size+1, encoder_dim]
        projection.bias    [vocab_size+1]

    Loading only works if vocab sizes AND encoder dims match.
    """
    decoder_params = {
        k: v for k, v in nemo_state.items()
        if k.startswith(nemo_prefix)
    }

    if not decoder_params:
        logger.warning("No decoder keys found with prefix '%s'", nemo_prefix)
        return 0, [f"no keys with prefix {nemo_prefix}"]

    ctc_state = ctc_head.state_dict()
    loaded = 0
    skipped: list[str] = []

    # Try shape-based matching
    used_nemo: set[str] = set()
    used_target: set[str] = set()

    for nemo_key, nemo_val in decoder_params.items():
        for target_key, target_val in ctc_state.items():
            if target_key in used_target:
                continue
            if nemo_val.shape == target_val.shape:
                ctc_state[target_key] = nemo_val
                loaded += 1
                used_nemo.add(nemo_key)
                used_target.add(target_key)
                logger.info(
                    "  CTC: %s → %s %s", nemo_key, target_key,
                    list(nemo_val.shape),
                )
                break

    for nk in decoder_params:
        if nk not in used_nemo:
            skipped.append(f"{nk} {list(decoder_params[nk].shape)}")

    if loaded > 0:
        ctc_head.load_state_dict(ctc_state, strict=False)

    if skipped:
        logger.info(
            "  CTC head: %d params loaded, %d skipped (likely vocab size "
            "mismatch — NeMo vs VADASR)",
            loaded, len(skipped),
        )

    return loaded, skipped


# ============================================================================
# Public API
# ============================================================================

def load_nemo_weights(
    model: nn.Module,
    nemo_path: str | Path,
    load_conformer: bool = True,
    load_ctc_head: bool = True,
    freeze_loaded: bool = False,
    strict: bool = False,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load NeMo pretrained weights into the VADASR model.

    Supports .nemo archives, .ckpt (PyTorch Lightning), and .pth files
    from NeMo Conformer-CTC-BPE and FastConformer-CTC-BPE models.

    This replicates the loading behavior from the original NeMo training
    scripts (low-precision-train-conformers.py / low-precision-train-fast-
    conformer.py) but without requiring NeMo as a dependency.

    Parameters
    ----------
    model : nn.Module
        The VADASR model instance.
    nemo_path : str | Path
        Path to the .nemo, .ckpt, or .pth file.
    load_conformer : bool
        Whether to load Conformer encoder weights (default True).
    load_ctc_head : bool
        Whether to load CTC decoder/head weights (default True).
    freeze_loaded : bool
        If True, freeze the loaded Conformer weights (requires_grad=False).
        Useful for phase-1 training where only the decoder is trained.
    strict : bool
        If True, raise error on any unmatched keys.
    device : str | torch.device
        Device for loaded tensors.

    Returns
    -------
    dict with loading diagnostics:
        - 'nemo_arch': detected NeMo architecture info
        - 'conformer_loaded': number of Conformer params loaded
        - 'ctc_loaded': number of CTC params loaded
        - 'skipped': list of keys that couldn't be mapped
    """
    nemo_path = Path(nemo_path)
    logger.info("=" * 60)
    logger.info("Loading NeMo weights from: %s", nemo_path)
    logger.info("=" * 60)

    # Step 1: Extract state dict
    nemo_state = _extract_nemo_checkpoint(nemo_path)
    logger.info("NeMo checkpoint contains %d parameter tensors", len(nemo_state))

    # Step 2: Detect architecture
    arch_info = _detect_nemo_architecture(nemo_state)

    diagnostics: dict[str, Any] = {
        "nemo_arch": arch_info,
        "conformer_loaded": 0,
        "ctc_loaded": 0,
        "skipped": [],
    }

    # Step 3: Load Conformer encoder weights
    if load_conformer and arch_info["encoder_prefix"]:
        logger.info("--- Loading Conformer encoder weights ---")
        logger.info(
            "  NeMo: %d layers, d_model=%d | Target: %d layers, d_model=%d",
            arch_info["n_layers"], arch_info["d_model"],
            model.conformer.conformer.num_layers
            if hasattr(model.conformer.conformer, "num_layers")
            else "?",
            model.conformer.encoder_dim,
        )

        conformer_loaded, conformer_total, conformer_skipped = \
            _load_conformer_layers(
                nemo_state=nemo_state,
                target_conformer=model.conformer.conformer,
                nemo_prefix=arch_info["encoder_prefix"],
                num_target_layers=len(set([
                    k.split(".")[1] for k in model.conformer.conformer.state_dict()
                    if k.startswith("conformer_layers.")
                    and k.split(".")[1].isdigit()
                ])),
            )

        diagnostics["conformer_loaded"] = conformer_loaded
        diagnostics["skipped"].extend(conformer_skipped)
        logger.info(
            "  Conformer: loaded %d params (target has %d total)",
            conformer_loaded, conformer_total,
        )

        # Optionally freeze loaded weights (phase-1 style)
        if freeze_loaded:
            frozen_count = 0
            for param in model.conformer.conformer.parameters():
                param.requires_grad = False
                frozen_count += 1
            logger.info(
                "  Froze %d Conformer parameters (phase-1 training mode)",
                frozen_count,
            )

    elif load_conformer:
        logger.warning("No encoder keys found in NeMo checkpoint!")

    # Step 4: Load CTC head weights
    if load_ctc_head and arch_info["decoder_prefix"]:
        logger.info("--- Loading CTC head weights ---")
        ctc_loaded, ctc_skipped = _load_ctc_head(
            nemo_state=nemo_state,
            ctc_head=model.ctc_head,
            nemo_prefix=arch_info["decoder_prefix"],
        )
        diagnostics["ctc_loaded"] = ctc_loaded
        diagnostics["skipped"].extend(ctc_skipped)
    elif load_ctc_head:
        logger.warning("No decoder keys found in NeMo checkpoint!")

    # Step 5: Summary
    total = diagnostics["conformer_loaded"] + diagnostics["ctc_loaded"]
    n_skipped = len(diagnostics["skipped"])
    logger.info("=" * 60)
    logger.info(
        "NeMo weight loading complete: %d loaded (%d conformer + %d ctc), "
        "%d skipped",
        total, diagnostics["conformer_loaded"],
        diagnostics["ctc_loaded"], n_skipped,
    )
    if n_skipped > 0:
        logger.info("  Skipped keys (first 10): %s", diagnostics["skipped"][:10])
    logger.info("=" * 60)

    if strict and n_skipped > 0:
        raise RuntimeError(
            f"Strict loading failed: {n_skipped} keys could not be mapped. "
            f"First 5: {diagnostics['skipped'][:5]}"
        )

    return diagnostics
