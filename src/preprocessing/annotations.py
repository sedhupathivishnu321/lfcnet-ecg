"""
Real label derivation for AFDB / LTAFDB / SHDB-AF.

These three databases are rhythm-annotated, not metadata-table-annotated like
EPHNOGRAM: each record ships a `.atr` file where `aux_note` entries such as
"(AFIB", "(N", "(AFL", "(J", ... mark the *start* of a rhythm segment that
continues until the next rhythm aux_note. There is no per-record spreadsheet
label to join against - the ground truth lives inside the annotation stream
itself, at the sample level.

This is the ECG-only equivalent of the EPHNOGRAM "Critical fix": instead of
guessing a class from a filename, or from a top-level per-record label, we
build a genuine **per-sample** AF / Non-AF timeline for every record from its
real rhythm annotations, and only then window it. A record can (and often
does) contain both classes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import wfdb

from src.config import ConfigNode


@dataclass
class RhythmInterval:
    start_sample: int
    end_sample: int
    label: int  # 1 = AF (AFIB/AFL), 0 = Non-AF


def _is_af_token(token: str, af_tokens: list[str]) -> bool:
    return any(token.startswith(t) for t in af_tokens)


def build_rhythm_timeline(
    record_path: str, cfg: ConfigNode
) -> tuple[list[RhythmInterval], int]:
    """
    Returns (intervals, sig_len) where intervals fully tile [0, sig_len) with
    AF (1) / Non-AF (0) labels, derived only from real rhythm aux_note tokens.
    Unrecognized/non-rhythm aux_notes inherit the previously active label
    (e.g. beat-level annotations interleaved with rhythm annotations).
    """
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")
    sig_len = rec.sig_len

    af_tokens = list(cfg.labels.af_aux_tokens)
    non_af_tokens = list(cfg.labels.non_af_aux_tokens)

    rhythm_changes = []  # (sample, label)
    current_label = 0
    for sample, aux in zip(ann.sample, ann.aux_note):
        aux = (aux or "").strip()
        if not aux.startswith("("):
            continue  # not a rhythm-change annotation
        if _is_af_token(aux, af_tokens):
            current_label = 1
        elif _is_af_token(aux, non_af_tokens) or aux == "(":
            current_label = 0
        else:
            # Unknown rhythm label (e.g. "(P", "(AB") - treat conservatively as Non-AF
            current_label = 0
        rhythm_changes.append((int(sample), current_label))

    if not rhythm_changes:
        raise ValueError(f"{record_path}: no rhythm aux_note annotations found - cannot derive real labels")

    intervals: list[RhythmInterval] = []
    for i, (start, label) in enumerate(rhythm_changes):
        end = rhythm_changes[i + 1][0] if i + 1 < len(rhythm_changes) else sig_len
        if end > start:
            intervals.append(RhythmInterval(start, end, label))

    return intervals, sig_len


def sample_labels_from_intervals(intervals: list[RhythmInterval], sig_len: int) -> np.ndarray:
    """Dense per-sample 0/1 label array, for windowing / QA plots."""
    labels = np.zeros(sig_len, dtype=np.int64)
    for iv in intervals:
        labels[iv.start_sample: iv.end_sample] = iv.label
    return labels


def class_distribution(intervals: list[RhythmInterval]) -> dict[int, int]:
    dist: dict[int, int] = {0: 0, 1: 0}
    for iv in intervals:
        dist[iv.label] += iv.end_sample - iv.start_sample
    return dist
