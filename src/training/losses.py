"""Config-selectable loss functions for binary AF classification."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ConfigNode


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_loss(cfg: ConfigNode, class_weight: torch.Tensor | None = None) -> nn.Module:
    t = cfg.training
    if t.loss == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weight)
    if t.loss == "label_smoothing":
        return nn.CrossEntropyLoss(weight=class_weight, label_smoothing=t.label_smoothing)
    if t.loss == "focal":
        return FocalLoss(gamma=t.focal_gamma, weight=class_weight)
    raise ValueError(f"Unknown loss: {t.loss}")
