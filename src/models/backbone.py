"""Single-modality ECG backbone: multi-scale stem -> residual DS blocks -> downsample."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import ConfigNode
from src.models.layers import MultiScaleConvStem, ResidualDSBlock


class ECGBackbone(nn.Module):
    def __init__(self, cfg: ConfigNode):
        super().__init__()
        m = cfg.model
        self.stem = MultiScaleConvStem(m.in_channels, m.stem_out_channels, list(m.stem_kernel_sizes))
        self.proj = nn.Conv1d(m.stem_out_channels, m.residual_channels, kernel_size=1, bias=False)
        use_se = getattr(m, "use_se", True)
        self.blocks = nn.ModuleList(
            [
                ResidualDSBlock(m.residual_channels, se_reduction=m.se_reduction, use_se=use_se)
                for _ in range(m.residual_blocks)
            ]
        )
        # Progressive 2x downsample per block (2^n_blocks total) rather than one
        # aggressive pool, so temporal resolution degrades gradually and the
        # self-attention stage still has enough tokens to attend over.
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
            x = self.downsample(x)
        return x  # (B, residual_channels, T_down)
