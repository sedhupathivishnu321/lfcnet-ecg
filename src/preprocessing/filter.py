"""Config-driven bandpass + notch filtering for single-lead ECG."""
from __future__ import annotations

import numpy as np
from scipy import signal

from src.config import ConfigNode


def _nyquist_safe(cutoff_hz: float, fs: float) -> float:
    nyq = fs / 2.0
    return min(cutoff_hz, nyq * 0.98)


def design_bandpass(cfg: ConfigNode, fs: float):
    f = cfg.preprocessing.filter
    nyq = fs / 2.0
    low = _nyquist_safe(f.lowcut_hz, fs) / nyq
    high = _nyquist_safe(f.highcut_hz, fs) / nyq
    low, high = max(low, 1e-4), min(high, 0.999)

    if f.type == "butterworth":
        b, a = signal.butter(f.order, [low, high], btype="band")
    elif f.type == "bessel":
        b, a = signal.bessel(f.order, [low, high], btype="band")
    elif f.type == "cheby1":
        b, a = signal.cheby1(f.order, 0.5, [low, high], btype="band")
    elif f.type == "elliptic":
        b, a = signal.ellip(f.order, 0.5, 40, [low, high], btype="band")
    elif f.type == "fir":
        numtaps = f.order * 2 + 1
        b = signal.firwin(numtaps, [low, high], pass_zero=False)
        a = np.array([1.0])
    else:
        raise ValueError(f"Unknown filter type: {f.type}")
    return b, a


def apply_bandpass(x: np.ndarray, cfg: ConfigNode, fs: float) -> np.ndarray:
    b, a = design_bandpass(cfg, fs)
    return signal.filtfilt(b, a, x)


def apply_notch(x: np.ndarray, cfg: ConfigNode, fs: float) -> np.ndarray:
    f = cfg.preprocessing.filter
    nyq = fs / 2.0
    if f.notch_hz >= nyq:
        return x  # nothing to notch below Nyquist
    b, a = signal.iirnotch(f.notch_hz / nyq, f.notch_q)
    return signal.filtfilt(b, a, x)


def remove_baseline_wander(x: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> np.ndarray:
    nyq = fs / 2.0
    cutoff = min(cutoff_hz, nyq * 0.9) / nyq
    b, a = signal.butter(2, cutoff, btype="high")
    return signal.filtfilt(b, a, x)


def full_filter_chain(x: np.ndarray, cfg: ConfigNode, fs: float) -> np.ndarray:
    x = remove_baseline_wander(x, fs)
    x = apply_bandpass(x, cfg, fs)
    x = apply_notch(x, cfg, fs)
    return x
