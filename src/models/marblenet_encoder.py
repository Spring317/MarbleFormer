"""
MarbleNet Encoder — 1D time-channel separable convolution blocks for VAD.

Single Responsibility: Encode mel features into a compact representation
suitable for voice activity detection.

Architecture: MarbleNet-BxRxC (default 3x2x64)
  - Prologue: 1D conv → BN → ReLU
  - B residual blocks, each with R sub-blocks of:
      depthwise 1D conv → pointwise 1D conv → BN → ReLU → Dropout
  - Epilogue: 3 additional conv layers
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution (depthwise + pointwise).

    This is the core building block of MarbleNet — dramatically reduces
    parameter count compared to standard convolution while preserving
    representational capacity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int | str = "same",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _SubBlock(nn.Module):
    """Single sub-block: DepthwiseSeparableConv1d → BN → ReLU → Dropout."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = _DepthwiseSeparableConv1d(
            channels, channels, kernel_size
        )
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.conv(x))))


class _ResidualBlock(nn.Module):
    """Residual block containing R sub-blocks with a skip connection.

    The residual path uses a pointwise conv to match channel dimensions
    when input and output channels differ.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_sub_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # First sub-block may change channels
        layers: list[nn.Module] = []
        if in_channels != out_channels:
            layers.append(
                nn.Sequential(
                    _DepthwiseSeparableConv1d(
                        in_channels, out_channels, kernel_size
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
            )
            start = 1
        else:
            start = 0

        for _ in range(start, num_sub_blocks):
            layers.append(_SubBlock(out_channels, kernel_size, dropout))

        self.sub_blocks = nn.Sequential(*layers)

        # Residual projection (1×1 conv) when dimensions change
        self.residual = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.sub_blocks(x) + self.residual(x))


class MarbleNetEncoder(nn.Module):
    """MarbleNet encoder for voice activity detection.

    Parameters
    ----------
    input_channels : int
        Number of input mel channels (default 80).
    num_blocks : int
        Number of residual blocks B (default 3).
    num_sub_blocks : int
        Number of sub-blocks per residual block R (default 2).
    num_channels : int
        Channel count C for main blocks (default 64).
    kernel_sizes : list[int]
        Kernel size per block (default [11, 13, 15]).
    prologue_channels : int
        Output channels for the prologue conv (default 128).
    prologue_kernel : int
        Kernel size for prologue conv (default 11).
    epilogue_channels : list[int]
        Channels for the 3 epilogue convs (default [64, 128, 128]).
    epilogue_kernels : list[int]
        Kernel sizes for the 3 epilogue convs (default [29, 1, 1]).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        input_channels: int = 80,
        num_blocks: int = 3,
        num_sub_blocks: int = 2,
        num_channels: int = 64,
        kernel_sizes: list[int] | None = None,
        prologue_channels: int = 128,
        prologue_kernel: int = 11,
        epilogue_channels: list[int] | None = None,
        epilogue_kernels: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [11, 13, 15]
        if epilogue_channels is None:
            epilogue_channels = [64, 128, 128]
        if epilogue_kernels is None:
            epilogue_kernels = [29, 1, 1]

        assert len(kernel_sizes) == num_blocks
        assert len(epilogue_channels) == 3
        assert len(epilogue_kernels) == 3

        # --- Prologue ---
        self.prologue = nn.Sequential(
            nn.Conv1d(
                input_channels, prologue_channels,
                kernel_size=prologue_kernel, padding="same", bias=False
            ),
            nn.BatchNorm1d(prologue_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # --- Main residual blocks ---
        blocks: list[nn.Module] = []
        in_ch = prologue_channels
        for i in range(num_blocks):
            blocks.append(
                _ResidualBlock(
                    in_channels=in_ch,
                    out_channels=num_channels,
                    kernel_size=kernel_sizes[i],
                    num_sub_blocks=num_sub_blocks,
                    dropout=dropout,
                )
            )
            in_ch = num_channels
        self.blocks = nn.Sequential(*blocks)

        # --- Epilogue (3 conv layers) ---
        epilogue_layers: list[nn.Module] = []
        in_ch = num_channels
        for i in range(3):
            epilogue_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch, epilogue_channels[i],
                        kernel_size=epilogue_kernels[i],
                        padding="same", bias=False,
                    ),
                    nn.BatchNorm1d(epilogue_channels[i]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
            )
            in_ch = epilogue_channels[i]
        self.epilogue = nn.Sequential(*epilogue_layers)

        self.output_channels = epilogue_channels[-1]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "MarbleNetEncoder":
        """Construct from a ``marblenet`` config dict."""
        return cls(
            input_channels=cfg.get("input_channels", 80),
            num_blocks=cfg.get("num_blocks", 3),
            num_sub_blocks=cfg.get("num_sub_blocks", 2),
            num_channels=cfg.get("num_channels", 64),
            kernel_sizes=cfg.get("kernel_sizes"),
            prologue_channels=cfg.get("prologue_channels", 128),
            prologue_kernel=cfg.get("prologue_kernel", 11),
            epilogue_channels=cfg.get("epilogue_channels"),
            epilogue_kernels=cfg.get("epilogue_kernels"),
            dropout=cfg.get("dropout", 0.1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, features: torch.Tensor, feat_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode mel features.

        Parameters
        ----------
        features : Tensor [B, n_mels, T]
            Log-mel spectrogram.
        feat_lengths : Tensor [B]
            Valid frame counts.

        Returns
        -------
        encoded : Tensor [B, output_channels, T]
            Encoded features (same temporal resolution as input).
        feat_lengths : Tensor [B]
            Unchanged lengths (MarbleNet preserves temporal resolution).
        """
        x = self.prologue(features)
        x = self.blocks(x)
        x = self.epilogue(x)
        return x, feat_lengths
