"""
NeMo-compatible Conformer blocks for VADASR.

Implements the exact same architecture and parameter naming as
NVIDIA NeMo's ConformerEncoder, enabling direct weight loading
from .nemo pretrained models without any key remapping.

Key architecture choices matching NeMo:
  - Relative positional multi-head attention (separate Q/K/V/Out projections)
  - Macaron-style feed-forward with half-step residuals (fc_factor=0.5)
  - Depthwise separable convolution with GLU gating
  - Pre-layer-norm for each sub-module, final layer-norm at block output

NeMo state_dict key alignment (per layer):
  layers.{i}.norm_feed_forward1.weight/bias
  layers.{i}.feed_forward1.linear1.weight/bias
  layers.{i}.feed_forward1.linear2.weight/bias
  layers.{i}.norm_self_att.weight/bias
  layers.{i}.self_attn.linear_q/k/v/out.weight/bias
  layers.{i}.self_attn.linear_pos.weight
  layers.{i}.self_attn.pos_bias_u/v
  layers.{i}.norm_conv.weight/bias
  layers.{i}.conv_module.pointwise_conv1/2.weight/bias
  layers.{i}.conv_module.depthwise_conv.weight/bias
  layers.{i}.conv_module.batch_norm.*
  layers.{i}.norm_feed_forward2.weight/bias
  layers.{i}.feed_forward2.linear1.weight/bias
  layers.{i}.feed_forward2.linear2.weight/bias
  layers.{i}.norm_out.weight/bias

Our module names (under NeMoConformer) use `conformer_layers.{i}.*` to
match the torchaudio convention that the existing nemo_loader expects.
The sub-keys within each layer are identical to NeMo.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Relative Positional Multi-Head Attention
# ============================================================================

class RelPositionMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding.

    Mirrors NeMo's ``RelPositionMultiHeadAttention`` parameter names:
      - linear_q, linear_k, linear_v, linear_out  (nn.Linear, with bias)
      - linear_pos  (nn.Linear, no bias)
      - pos_bias_u, pos_bias_v  (nn.Parameter, [n_heads, head_dim])

    Parameters
    ----------
    n_feat : int
        Model dimension (d_model).
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout rate.
    """

    def __init__(self, n_feat: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert n_feat % n_heads == 0, (
            f"n_feat ({n_feat}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.head_dim = n_feat // n_heads
        self.n_feat = n_feat
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Separate Q/K/V/Out projections (NeMo naming)
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

        # Relative positional encoding projection (no bias, NeMo convention)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        # Learnable relative position biases (NeMo naming)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        """Relative shift (skew trick) — matches NeMo's implementation.

        Converts positional scores from shape [B, H, T, 2T-1] to [B, H, T, T]
        by extracting the correct relative position for each (query, key) pair.
        """
        b, h, qlen, pos_len = x.size()
        # Pad on the left with one zero column
        zero_pad = torch.zeros(
            (b, h, qlen, 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)  # [B, H, T, 2T]
        # Reshape and remove the first row to perform the diagonal shift
        x_padded = x_padded.reshape(b, h, pos_len + 1, qlen)
        x = x_padded[:, :, 1:].reshape(b, h, qlen, pos_len)
        # Keep only the first T columns (valid relative positions)
        x = x[:, :, :, :qlen]
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query, key, value : Tensor [B, T, D]
        pos_emb : Tensor [1, 2T-1, D]
            Relative sinusoidal positional encoding.
        mask : Optional[Tensor] [B, 1, 1, T]
            Attention mask (True = attend, False = ignore).

        Returns
        -------
        output : Tensor [B, T, D]
        """
        B, T, _ = query.size()

        # Project Q, K, V → [B, H, T, D_h]
        q = self.linear_q(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(key).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(value).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Project positional encoding → [1, H, 2T-1, D_h]
        p = self.linear_pos(pos_emb).view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Content-based attention: (q + pos_bias_u) @ k^T
        q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)  # [B, H, T, D_h]
        content_score = torch.matmul(q_with_bias_u, k.transpose(-2, -1))  # [B, H, T, T]

        # Position-based attention: (q + pos_bias_v) @ p^T → rel_shift
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)  # [B, H, T, D_h]
        pos_score = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # [B, H, T, 2T-1]
        pos_score = self._rel_shift(pos_score)  # [B, H, T, T]

        # Combine scores
        scores = (content_score + pos_score) * self.scale

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, T, D_h]
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_feat)
        return self.linear_out(out)


# ============================================================================
# Feed-Forward Module (Macaron-style)
# ============================================================================

class ConformerFeedForward(nn.Module):
    """Conformer feed-forward module matching NeMo's ``ConformerFeedForward``.

    Attribute names match NeMo:
      - linear1 (d_model → d_ff)
      - linear2 (d_ff → d_model)

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()  # Swish, matching NeMo default
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, T, D]."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# ============================================================================
# Convolution Module
# ============================================================================

class ConformerConvModule(nn.Module):
    """Conformer convolution module matching NeMo's ``ConformerConvolution``.

    Structure:
      pointwise_conv1 (×2 channels, GLU gating) → depthwise_conv
      → batch_norm → SiLU → pointwise_conv2

    Attribute names match NeMo:
      - pointwise_conv1  (Conv1d, d → 2d)
      - depthwise_conv   (Conv1d, d → d, groups=d)
      - batch_norm        (BatchNorm1d)
      - pointwise_conv2  (Conv1d, d → d)

    Parameters
    ----------
    d_model : int
        Model dimension / number of channels.
    kernel_size : int
        Depthwise convolution kernel size.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self, d_model: int, kernel_size: int = 31, dropout: float = 0.0
    ) -> None:
        super().__init__()
        # Pointwise expansion with GLU (2× channels for gating)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, 2 * d_model, kernel_size=1, bias=True
        )
        # Depthwise separable conv
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            groups=d_model, padding=padding, bias=True
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=1, bias=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, T, D]."""
        x = x.transpose(1, 2)                # [B, D, T]
        x = self.pointwise_conv1(x)           # [B, 2D, T]
        x = F.glu(x, dim=1)                  # [B, D, T]
        x = self.depthwise_conv(x)            # [B, D, T]
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)           # [B, D, T]
        x = self.dropout(x)
        return x.transpose(1, 2)              # [B, T, D]


# ============================================================================
# Conformer Block
# ============================================================================

class ConformerBlock(nn.Module):
    """Single Conformer block matching NeMo's ``ConformerLayer``.

    Macaron-Net architecture:
      FFN₁ (half-step) → Self-Attn → Conv → FFN₂ (half-step) → LayerNorm

    Attribute names match NeMo:
      - norm_feed_forward1, feed_forward1
      - norm_self_att, self_attn
      - norm_conv, conv_module
      - norm_feed_forward2, feed_forward2
      - norm_out

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_ff : int
        Feed-forward hidden dimension.
    n_heads : int
        Number of attention heads.
    conv_kernel_size : int
        Depthwise convolution kernel size.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc_factor = 0.5  # Macaron half-step factor

        # FFN1 (Macaron first half-step)
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff, dropout)

        # Self-attention with relative positional encoding
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(d_model, n_heads, dropout)

        # Convolution module
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)

        # FFN2 (Macaron second half-step)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff, dropout)

        # Final layer norm
        self.norm_out = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, T, D]
        pos_emb : Tensor [1, 2T-1, D]
            Relative sinusoidal positional encoding.
        mask : Optional[Tensor] [B, 1, 1, T]
            Attention mask.

        Returns
        -------
        Tensor [B, T, D]
        """
        # FFN1 with half-step residual
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        x = residual + self.fc_factor * x

        # Self-attention with residual
        residual = x
        x_norm = self.norm_self_att(x)
        x_attn = self.self_attn(x_norm, x_norm, x_norm, pos_emb, mask)
        x = residual + self.dropout(x_attn)

        # Convolution with residual
        residual = x
        x = self.norm_conv(x)
        x = self.conv_module(x)
        x = residual + x

        # FFN2 with half-step residual
        residual = x
        x = self.norm_feed_forward2(x)
        x = self.feed_forward2(x)
        x = residual + self.fc_factor * x

        # Final layer norm
        x = self.norm_out(x)

        return x


# ============================================================================
# NeMo-compatible Conformer (stack of blocks)
# ============================================================================

class NeMoConformer(nn.Module):
    """Stack of NeMo-compatible Conformer blocks.

    Drop-in replacement for torchaudio's ``Conformer``, matching:
      - NeMo's parameter naming for direct weight loading
      - Relative sinusoidal positional encoding (computed on-the-fly)
      - Input scaling by ``sqrt(d_model)`` (NeMo convention)

    The blocks are stored as ``self.conformer_layers`` (ModuleList) to
    match the key prefix convention expected by the nemo_loader.

    Parameters
    ----------
    input_dim : int
        Model dimension (d_model).
    num_heads : int
        Number of attention heads.
    ffn_dim : int
        Feed-forward hidden dimension.
    num_layers : int
        Number of Conformer blocks.
    depthwise_conv_kernel_size : int
        Kernel size for depthwise convolution in each block.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.xscale = math.sqrt(input_dim)

        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                d_model=input_dim,
                d_ff=ffn_dim,
                n_heads=num_heads,
                conv_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    @staticmethod
    def _create_pos_embedding(
        length: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create sinusoidal relative positional encoding (NeMo convention).

        Generates positions [+(T-1), ..., +1, 0, -1, ..., -(T-1)]
        as a tensor of shape [1, 2T-1, D].

        This matches NeMo's ``RelPositionalEncoding.extend_pe()``.
        """
        # Positive positions: [0, 1, ..., T-1]
        position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype)
            * -(math.log(10000.0) / d_model)
        )

        pe_positive = torch.zeros(length, d_model, device=device, dtype=dtype)
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)

        # Negative positions: [-1, -2, ..., -(T-1)]
        pe_negative = torch.zeros(length, d_model, device=device, dtype=dtype)
        pe_negative[:, 0::2] = torch.sin(-position * div_term)
        pe_negative[:, 1::2] = torch.cos(-position * div_term)

        # NeMo convention: reverse positive, drop position 0 from negative
        pe_positive = torch.flip(pe_positive, [0])   # [T-1, T-2, ..., 0]
        pe_negative = pe_negative[1:]                  # [-1, -2, ..., -(T-1)]

        pe = torch.cat([pe_positive, pe_negative], dim=0)  # [2T-1, D]
        return pe.unsqueeze(0)  # [1, 2T-1, D]

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor [B, T, D]
            Subsampled encoder features.
        lengths : Tensor [B]
            Valid frame counts after subsampling.

        Returns
        -------
        (output, lengths) — output is [B, T, D], lengths unchanged.
        """
        # Scale input (NeMo convention)
        x = x * self.xscale

        T = x.size(1)
        pos_emb = self._create_pos_embedding(T, self.input_dim, x.device, x.dtype)

        # Create attention mask from lengths [B, 1, 1, T]
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            arange = torch.arange(max_len, device=x.device)
            mask = arange.unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]
            mask = mask.unsqueeze(1).unsqueeze(1)                # [B, 1, 1, T]

        for layer in self.conformer_layers:
            x = layer(x, pos_emb, mask)

        return x, lengths
