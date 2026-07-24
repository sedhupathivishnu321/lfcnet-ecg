"""
Class-balanced sampler and optional feature-space SMOTE.

AF windows are typically a minority class overall (though LTAFDB skews the
other way), so this mirrors the original project's imbalance handling exactly,
just applied to a single ECG feature stream instead of ECG+PCG.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from src.config import ConfigNode


def build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def smote_oversample(X: np.ndarray, y: np.ndarray, cfg: ConfigNode):
    """Feature-space SMOTE on flattened windows (used only if cfg.sampler.smote is true)."""
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(k_neighbors=cfg.sampler.smote_k_neighbors, random_state=cfg.project.seed)
    X_flat = X.reshape(len(X), -1)
    X_res, y_res = smote.fit_resample(X_flat, y)
    return X_res.reshape(-1, X.shape[1]), y_res
