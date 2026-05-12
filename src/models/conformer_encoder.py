"""
Conformer Encoder for ASR.

Single Responsibility: Transform MarbleNet-encoded features into
high-level representations suitable for CTC decoding.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torchaudio.models import Conformer


class _ConvSubsampling(nn.Module):
    """Convolutional subsampling front-end (factor 4)."""

    def __init__(self, input_channels: int, encoder_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_dim, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        freq_out = math.ceil(math.ceil(input_channels / 2) / 2)
        self.linear = nn.Linear(encoder_dim * freq_out, encoder_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.unsqueeze(1)  # [B,C,T] -> [B,1,C,T]
        x = self.conv(x)    # [B,D,C//4,T//4]
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.linear(x)
        out_lengths = (lengths.float() / 2).ceil().long()
        out_lengths = (out_lengths.float() / 2).ceil().long()
        out_lengths = out_lengths.clamp(max=t)
        return x, out_lengths


class ConformerEncoder(nn.Module):
    """Conformer encoder with convolutional subsampling.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (MarbleNet output channels).
    encoder_dim : int
        Conformer model dimension.
    num_heads : int
        Number of attention heads.
    ffn_dim : int
        Feed-forward hidden dimension.
    num_layers : int
        Number of Conformer blocks.
    depthwise_conv_kernel_size : int
        Kernel size for depthwise conv.
    dropout : float
        Dropout rate.
    subsampling_factor : int
        Temporal subsampling factor (default 4).
    """

    def __init__(
        self,
        input_dim: int = 128,
        encoder_dim: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 256,
        num_layers: int = 4,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        subsampling_factor: int = 4,
    ) -> None:
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.encoder_dim = encoder_dim
        self.subsampling = _ConvSubsampling(input_dim, encoder_dim)
        self.conformer = Conformer(
            input_dim=encoder_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "ConformerEncoder":
        return cls(
            input_dim=cfg.get("input_dim", 128),
            encoder_dim=cfg.get("encoder_dim", 256),
            num_heads=cfg.get("num_heads", 4),
            ffn_dim=cfg.get("ffn_dim", 256),
            num_layers=cfg.get("num_layers", 4),
            depthwise_conv_kernel_size=cfg.get("depthwise_conv_kernel_size", 31),
            dropout=cfg.get("dropout", 0.1),
            subsampling_factor=cfg.get("subsampling_factor", 4),
        )

    def forward(self, features: torch.Tensor, feat_lengths: torch.Tensor):
        """
        Parameters
        ----------
        features : Tensor [B, C, T]  — MarbleNet encoder output.
        feat_lengths : Tensor [B]

        Returns
        -------
        encoded : Tensor [B, T_sub, encoder_dim]
        out_lengths : Tensor [B]
        """
        x, out_lengths = self.subsampling(features, feat_lengths)
        x, out_lengths = self.conformer(x, out_lengths)
        return x, out_lengths
