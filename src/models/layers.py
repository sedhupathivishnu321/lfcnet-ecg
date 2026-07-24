"""
Reusable lightweight 1D building blocks for LAFNet.

Everything here is depthwise-separable by default to keep the embedded
(STM32 / CMSIS-NN / TFLite Micro) parameter and FLOP budget small - this is
the single-modality descendant of LMFCNet's ECG branch, so the same building
blocks apply, just without a second (PCG) branch to fuse.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MultiScaleConvStem(nn.Module):
    """Parallel depthwise-separable convs at several kernel sizes, fused by 1x1 conv.

    Captures fine (QRS-complex), medium (P/T-wave), and coarse (rhythm-level,
    i.e. RR-interval-scale) temporal structure in a single layer.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_sizes: list[int]):
        super().__init__()
        branch_ch = max(out_ch // len(kernel_sizes), 1)
        self.branches = nn.ModuleList(
            [DepthwiseSeparableConv1d(in_ch, branch_ch, k) for k in kernel_sizes]
        )
        self.fuse = nn.Conv1d(branch_ch * len(kernel_sizes), out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        x = self.bn(x)
        return self.act(x)


class SEBlock1d(nn.Module):
    """Squeeze-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        s = x.mean(dim=-1)  # global average pool -> (B, C)
        s = self.act(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class ResidualDSBlock(nn.Module):
    """Depthwise-separable conv block with optional SE + residual connection.

    `use_se=False` is used by the SE-ablation arm (`ablation.se: [true, false]`
    in config.yaml) to measure how much the Squeeze-Excitation stage
    contributes to accuracy vs. its (small) parameter/FLOP cost.
    """

    def __init__(self, channels: int, kernel_size: int = 7, se_reduction: int = 4, use_se: bool = True):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(channels, channels, kernel_size)
        self.use_se = use_se
        self.se = SEBlock1d(channels, se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(x)
        x = self.se(x)
        return x + residual


class TemporalSelfAttention(nn.Module):
    """
    Single-head scaled dot-product self-attention over pooled temporal tokens.

    Replaces LMFCNet's ECG<->PCG cross-attention: with only one modality left,
    the analogous long-range structure to capture is *within-signal* temporal
    context (e.g. relating an early ectopic complex to a later irregular run),
    so attention runs ECG-tokens-to-ECG-tokens instead of ECG-to-PCG.
    """

    def __init__(self, channels: int, attn_dim: int, heads: int = 1):
        super().__init__()
        self.heads = heads
        self.attn_dim = attn_dim
        self.q = nn.Linear(channels, attn_dim)
        self.k = nn.Linear(channels, attn_dim)
        self.v = nn.Linear(channels, attn_dim)
        self.out = nn.Linear(attn_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> tokens (B, T, C)
        tokens = x.transpose(1, 2)
        q, k, v = self.q(tokens), self.k(tokens), self.v(tokens)
        scale = self.attn_dim ** 0.5
        attn = torch.softmax(q @ k.transpose(-1, -2) / scale, dim=-1)
        ctx = attn @ v
        out = self.out(ctx)
        return (tokens + out).transpose(1, 2)
