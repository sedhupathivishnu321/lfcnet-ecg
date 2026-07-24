"""Config-driven normalization: zscore / minmax / robust / decimal."""
from __future__ import annotations

import numpy as np

from src.config import ConfigNode


def normalize(x: np.ndarray, cfg: ConfigNode) -> np.ndarray:
    method = cfg.preprocessing.normalization.method
    if method == "zscore":
        mu, sigma = np.mean(x), np.std(x)
        return (x - mu) / (sigma + 1e-8)
    if method == "minmax":
        lo, hi = np.min(x), np.max(x)
        return (x - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
    if method == "robust":
        median = np.median(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        return (x - median) / (iqr + 1e-8)
    if method == "decimal":
        max_abs = np.max(np.abs(x))
        scale = 10 ** np.ceil(np.log10(max_abs + 1e-8))
        return x / scale
    raise ValueError(f"Unknown normalization method: {method}")
