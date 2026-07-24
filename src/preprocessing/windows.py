"""
Cuts a filtered/normalized single-lead ECG signal into fixed-length windows and
assigns each window a real AF / Non-AF label from the dense per-sample rhythm
timeline built in annotations.py (never a filename or record-level guess).

`iter_windows` is a generator: it yields one (window, label) pair at a time
instead of building up two Python lists for the whole record first. For
LTAFDB's day-plus-length recordings a single record can produce tens of
thousands of windows, so materializing the full per-record list before the
caller can do anything with it is itself a meaningful chunk of avoidable
allocation/copy overhead - src/preprocessing/hdf5.py consumes this generator
directly and flushes to the HDF5 file in small fixed-size batches, so peak
memory here is bounded by the flush buffer size, not by record length.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from src.config import ConfigNode
from src.preprocessing.quality import passes_quality


def iter_windows(
    x: np.ndarray,
    sample_labels: np.ndarray,
    fs: float,
    cfg: ConfigNode,
) -> Iterator[tuple[np.ndarray, int]]:
    w = cfg.preprocessing.windows
    win_len = int(w.length_seconds * fs)
    step = int(win_len * (1.0 - w.overlap))
    step = max(step, 1)

    for start in range(0, len(x) - win_len + 1, step):
        end = start + win_len
        seg = x[start:end]
        seg_labels = sample_labels[start:end]

        if w.reject_if_below_min_quality:
            ok, _reason = passes_quality(seg, fs, cfg)
            if not ok:
                continue

        af_fraction = float(np.mean(seg_labels))
        label = 1 if af_fraction >= w.af_positive_fraction else 0
        yield seg.astype(np.float32), label


def make_windows(
    x: np.ndarray,
    sample_labels: np.ndarray,
    fs: float,
    cfg: ConfigNode,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Back-compat wrapper around `iter_windows` that materializes the full
    per-record result into two lists. Kept for callers (tests, notebooks,
    ad-hoc scripts) that want "all windows for this record" as a simple
    return value. The streaming HDF5 writer in src/preprocessing/hdf5.py
    deliberately does NOT use this wrapper - it consumes `iter_windows`
    directly so it never holds a full record's windows in memory at once.
    """
    windows, labels = [], []
    for seg, label in iter_windows(x, sample_labels, fs, cfg):
        windows.append(seg)
        labels.append(label)
    return windows, labels
