"""Waveform augmentations for single-lead ECG windows. All ranges come from config.yaml."""
from __future__ import annotations

import numpy as np
from scipy.signal import resample

from src.config import ConfigNode


def add_noise(x: np.ndarray, std: float) -> np.ndarray:
    return x + np.random.normal(0, std, size=x.shape).astype(x.dtype)


def random_shift(x: np.ndarray, max_frac: float) -> np.ndarray:
    shift = int(np.random.uniform(-max_frac, max_frac) * len(x))
    return np.roll(x, shift)


def random_scale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return x * np.random.uniform(lo, hi)


def random_stretch(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    factor = np.random.uniform(lo, hi)
    n_new = max(int(len(x) * factor), 8)
    stretched = resample(x, n_new)
    if n_new >= len(x):
        return stretched[: len(x)].astype(x.dtype)
    out = np.zeros_like(x)
    out[:n_new] = stretched
    return out.astype(x.dtype)


def random_crop(x: np.ndarray, frac: float) -> np.ndarray:
    n = len(x)
    crop_len = int(n * frac)
    start = np.random.randint(0, max(n - crop_len, 1))
    cropped = x[start: start + crop_len]
    out = np.zeros_like(x)
    out[: len(cropped)] = cropped
    return out


def augment_ecg(x: np.ndarray, cfg: ConfigNode) -> np.ndarray:
    a = cfg.augment
    if np.random.rand() < 0.5:
        x = add_noise(x, a.noise_std)
    if np.random.rand() < 0.5:
        x = random_shift(x, a.shift_max_frac)
    if np.random.rand() < 0.5:
        x = random_scale(x, *a.scale_range)
    if np.random.rand() < 0.3:
        x = random_stretch(x, *a.stretch_range)
    if np.random.rand() < 0.3:
        x = random_crop(x, a.crop_frac)
    return x.astype(np.float32)


def mixup(x1: np.ndarray, y1: int, x2: np.ndarray, y2: int, alpha: float):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    return x.astype(np.float32), y1, y2, lam
