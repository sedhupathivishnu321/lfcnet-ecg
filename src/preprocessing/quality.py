"""NaN/Inf/flat-line/clipping checks + a lightweight Signal Quality Index for ECG windows."""
from __future__ import annotations

import numpy as np

from src.config import ConfigNode


def has_nan_or_inf(x: np.ndarray) -> bool:
    return bool(np.isnan(x).any() or np.isinf(x).any())


def is_flat(x: np.ndarray, fs: float, cfg: ConfigNode) -> bool:
    max_flat_samples = int(cfg.preprocessing.quality.max_flat_seconds * fs)
    if len(x) < max_flat_samples:
        return False
    diffs = np.abs(np.diff(x))
    zero_runs = diffs < 1e-8
    if not zero_runs.any():
        return False
    # longest run of near-zero derivative
    run = 0
    longest = 0
    for v in zero_runs:
        run = run + 1 if v else 0
        longest = max(longest, run)
    return longest >= max_flat_samples


def clip_fraction(x: np.ndarray) -> float:
    lo, hi = np.percentile(x, [0.1, 99.9])
    return float(np.mean((x <= lo) | (x >= hi)))


def signal_quality_index(x: np.ndarray) -> float:
    """Simple SQI in [0, 1]: kurtosis-based + variance-based heuristic (higher = cleaner)."""
    if np.std(x) < 1e-8:
        return 0.0
    z = (x - np.mean(x)) / (np.std(x) + 1e-8)
    kurt = float(np.mean(z ** 4) - 3.0)
    kurt_score = 1.0 / (1.0 + abs(kurt - 3.0) / 10.0)  # clean ECG kurtosis ~ around a few units above Gaussian
    var_score = min(1.0, np.std(x) / (np.mean(np.abs(x)) + 1e-8) / 2.0)
    return float(np.clip(0.5 * kurt_score + 0.5 * var_score, 0.0, 1.0))


def passes_quality(x: np.ndarray, fs: float, cfg: ConfigNode) -> tuple[bool, str]:
    q = cfg.preprocessing.quality
    if has_nan_or_inf(x):
        return False, "nan_or_inf"
    if is_flat(x, fs, cfg):
        return False, "flat"
    if clip_fraction(x) > q.clip_fraction_threshold:
        return False, "clipped"
    if signal_quality_index(x) < q.min_sqi:
        return False, "low_sqi"
    return True, "ok"
