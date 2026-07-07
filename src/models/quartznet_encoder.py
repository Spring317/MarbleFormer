"""
QuartzNet Encoder — 1D time-channel separable convolution blocks for ASR.
"""

from __future__ import annotations
import torch
import torch.nn as nn

class _DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int | str = "same", bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class _SubBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.conv = _DepthwiseSeparableConv1d(channels, channels, kernel_size)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.conv(x))))

class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_sub_blocks: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        if in_channels != out_channels:
            layers.append(nn.Sequential(
                _DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ))
            start = 1
        else:
            start = 0

        for _ in range(start, num_sub_blocks):
            layers.append(_SubBlock(out_channels, kernel_size, dropout))

        self.sub_blocks = nn.Sequential(*layers)
        self.residual = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels)) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.sub_blocks(x) + self.residual(x))

class QuartzNetEncoder(nn.Module):
    def __init__(
        self, 
        input_channels: int = 128, 
        num_sub_blocks: int = 5, 
        channels: list[int] = [256, 256, 512, 512, 512], 
        kernel_sizes: list[int] = [33, 39, 51, 63, 75], 
        epilogue_channels: list[int] = [512, 1024], 
        epilogue_kernels: list[int] = [87, 1], 
        dropout: float = 0.1
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        
        self.encoder_dim = epilogue_channels[-1]
        
        # Subsampling layer (QuartzNet C1 usually does stride 2)
        # Using padding = kernel_size // 2 to maintain roughly the same length before stride
        self.prologue = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=33, padding=16, stride=2, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        
        blocks = []
        in_ch = channels[0]
        for i in range(len(channels)):
            blocks.append(_ResidualBlock(in_ch, channels[i], kernel_sizes[i], num_sub_blocks, dropout))
            in_ch = channels[i]
        self.blocks = nn.Sequential(*blocks)

        epilogue_layers = []
        for i in range(len(epilogue_channels)):
            epilogue_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, epilogue_channels[i], kernel_size=epilogue_kernels[i], padding="same", bias=False),
                nn.BatchNorm1d(epilogue_channels[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ))
            in_ch = epilogue_channels[i]
        self.epilogue = nn.Sequential(*epilogue_layers)

    @classmethod
    def from_config(cls, cfg: dict) -> "QuartzNetEncoder":
        return cls(
            input_channels=cfg.get("input_channels", 128),
            num_sub_blocks=cfg.get("num_sub_blocks", 5),
            channels=cfg.get("channels", [256, 256, 512, 512, 512]),
            kernel_sizes=cfg.get("kernel_sizes", [33, 39, 51, 63, 75]),
            epilogue_channels=cfg.get("epilogue_channels", [512, 1024]),
            epilogue_kernels=cfg.get("epilogue_kernels", [87, 1]),
            dropout=cfg.get("dropout", 0.1),
        )

    def forward(self, features: torch.Tensor, feat_lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.prologue(features)
        out_lengths = (feat_lengths.float() / 2).ceil().long() if feat_lengths is not None else None
        x = self.blocks(x)
        x = self.epilogue(x)
        x = x.transpose(1, 2)
        return x, out_lengths
