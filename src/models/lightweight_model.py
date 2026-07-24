"""
LAFNet: Lightweight Atrial-Fibrillation Network.

Single-modality (ECG-only) successor to LMFCNet. Dropping the PCG branch and
the ECG<->PCG cross-attention/fusion stage - which accounted for a large share
of LMFCNet's 27,970 parameters - and replacing them with one temporal
self-attention stage over the ECG features themselves cuts parameter count
substantially while keeping the same multi-scale-stem + residual-SE backbone
design that made LMFCNet accurate on physiological waveforms.

    ECG ─▶ Multi-scale Depthwise CNN (k=3,7,15) ─▶ Residual×N (+SE) ─▶
           Temporal Self-Attention ─▶ Global-Avg-Pool ─▶ Classifier
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import ConfigNode, load_config
from src.models.backbone import ECGBackbone
from src.models.layers import TemporalSelfAttention


class LAFNet(nn.Module):
    def __init__(self, cfg: ConfigNode):
        super().__init__()
        m = cfg.model
        self.backbone = ECGBackbone(cfg)
        self.use_attention = getattr(m, "use_attention", True)
        self.attention = (
            TemporalSelfAttention(channels=m.residual_channels, attn_dim=m.attention_dim, heads=m.attention_heads)
            if self.use_attention
            else nn.Identity()
        )
        self.classifier = nn.Sequential(
            nn.Linear(m.residual_channels, m.classifier_hidden),
            nn.SiLU(),
            nn.Dropout(m.dropout),
            nn.Linear(m.classifier_hidden, m.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        feats = self.backbone(x)          # (B, C, T_down)
        feats = self.attention(feats)     # (B, C, T_down)
        pooled = feats.mean(dim=-1)       # global average pool -> (B, C)
        return self.classifier(pooled)    # (B, num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    cfg = load_config()
    model = LAFNet(cfg)
    n_params = count_parameters(model)
    print(f"LAFNet parameters: {n_params:,} ({n_params / 1000:.1f}K)")

    # sanity forward pass at the configured window length
    fs = cfg.preprocessing.target_fs
    win_len = int(cfg.preprocessing.windows.length_seconds * fs)
    dummy = torch.randn(2, cfg.model.in_channels, win_len)
    out = model(dummy)
    print(f"Forward pass OK, output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
