"""
Phase 4-6: builds the compressed HDF5 training dataset for LAFNet from all
enabled ECG-only WFDB datasets (AFDB, LTAFDB, SHDB-AF).

Pipeline per record:
  raw ECG lead -> baseline removal -> bandpass (Nyquist-safe) -> notch ->
  resample to target_fs -> normalize -> window -> label each window from the
  REAL rhythm-annotation timeline (src/preprocessing/annotations.py) -> append
  to HDF5.

Streaming, memory-safe by construction: src/preprocessing/windows.py's
`iter_windows` yields one window at a time instead of returning a full-record
list, and this module pushes each yielded window straight into a small,
fixed-size flush buffer (hardware.hdf5_flush_rows windows) that's written to
the resizable HDF5 dataset and cleared as soon as it fills - regardless of how
many windows a single record produces. The older pattern of collecting every
window from a whole record (or, worse, every record across AFDB + LTAFDB +
SHDB-AF at 50% overlap - tens of GB of windows) into a Python list and only
stacking/writing at the end peaked at roughly 2x the final dataset size in RAM
(list + array); this version's peak RAM for the writer itself is bounded by
`hardware.hdf5_flush_rows` windows, independent of record length or corpus
size. The window length in samples is fully determined by config.yaml up
front (target_fs x windows.length_seconds), so the resizable dataset's column
width is known before a single record is read - only the row count grows.

The HDF5 datasets use LZF compression by default (data.hdf5_compression) with
chunk rows sized to hardware.hdf5_chunk_rows: LZF is much cheaper to inflate
than gzip on the random single-row reads a shuffled training DataLoader issues
every step, which matters far more for total wall-clock time than the extra
disk space gzip would save.

Still logs the resulting class distribution and raises a loud error (and
deletes the partial file) if fewer than 2 classes end up in the built
dataset - the ECG-only equivalent of EPHNOGRAM's degenerate-single-class
guard.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import wfdb
from scipy.signal import resample_poly
from tqdm import tqdm

from src.config import load_config, enabled_datasets
from src.preprocessing.annotations import build_rhythm_timeline, sample_labels_from_intervals
from src.preprocessing.filter import full_filter_chain
from src.preprocessing.normalization import normalize
from src.preprocessing.windows import iter_windows


def _resample(x: np.ndarray, fs_native: float, fs_target: float) -> np.ndarray:
    if abs(fs_native - fs_target) < 1e-6:
        return x
    from math import gcd
    g = gcd(int(round(fs_native)), int(round(fs_target)))
    up, down = int(round(fs_target)) // g, int(round(fs_native)) // g
    return resample_poly(x, up, down)


def _log_available_ram_gb(cfg) -> None:
    """Best-effort heads-up if free RAM looks low relative to hardware.ram_gb -
    purely informational, never blocks the build (psutil is an optional check)."""
    try:
        import psutil  # noqa: WPS433
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        budget_gb = getattr(cfg, "hardware", None) and cfg.hardware.max_hdf5_build_ram_gb
        print(f"[hdf5] available RAM: {available_gb:.1f} GB (soft build budget: {budget_gb} GB)")
        if budget_gb and available_gb < budget_gb:
            print(f"[hdf5] WARNING: available RAM ({available_gb:.1f} GB) is below the configured "
                  f"hardware.max_hdf5_build_ram_gb ({budget_gb} GB) - the streaming writer keeps "
                  "peak usage to roughly one flush buffer's worth of windows, so this should still "
                  "be safe, but close other processes if you see swapping.")
    except ImportError:
        pass  # psutil not installed - not required, this check is advisory only


def _record_windows(ds, rid: str, cfg, target_fs: float):
    """
    Yields (window: np.ndarray shape (win_len,), label: int) one at a time for
    one record, or yields nothing on failure. All the per-record signal
    processing (filter/resample/normalize) still runs once up front - only the
    windowing step itself is streamed, since that's the stage whose output
    size (one row per window) actually scales with record length.
    """
    local_dir = Path(ds.local_dir)
    record_path = str(local_dir / rid)
    try:
        rec = wfdb.rdrecord(record_path)
        intervals, sig_len = build_rhythm_timeline(record_path, cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[hdf5] skipping {ds.name}/{rid}: {e}")
        return

    ch = ds.ecg_channel if ds.ecg_channel < rec.p_signal.shape[1] else 0
    ecg = rec.p_signal[:, ch].astype(np.float64)
    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

    sample_labels = sample_labels_from_intervals(intervals, sig_len)

    fs_native = float(rec.fs)
    ecg_filt = full_filter_chain(ecg, cfg, fs_native)
    ecg_rs = _resample(ecg_filt, fs_native, target_fs)
    labels_rs = _resample(sample_labels.astype(np.float64), fs_native, target_fs)
    labels_rs = (labels_rs >= 0.5).astype(np.int64)

    ecg_norm = normalize(ecg_rs, cfg)

    yield from iter_windows(ecg_norm, labels_rs, target_fs, cfg)


def build_hdf5_dataset(cfg) -> None:
    out_path = Path(cfg.data.hdf5_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_fs = cfg.preprocessing.target_fs
    win_len = int(cfg.preprocessing.windows.length_seconds * target_fs)
    hw = getattr(cfg, "hardware", None)
    chunk_rows = max(getattr(hw, "hdf5_chunk_rows", 256) if hw else 256, 1)
    flush_rows = max(getattr(hw, "hdf5_flush_rows", chunk_rows) if hw else chunk_rows, 1)
    compression = getattr(cfg.data, "hdf5_compression", "gzip")
    compression = compression if compression not in (None, "none", "None") else None

    _log_available_ram_gb(cfg)

    if out_path.exists():
        out_path.unlink()  # start clean - a stale partial file from a previous failed run must not linger

    class_counts = {0: 0, 1: 0}
    n_written = 0

    with h5py.File(out_path, "w") as f:
        ecg_ds = f.create_dataset(
            "ecg", shape=(0, win_len), maxshape=(None, win_len),
            dtype=np.float32, chunks=(min(chunk_rows, 1024), win_len), compression=compression,
        )
        label_ds = f.create_dataset(
            "label", shape=(0,), maxshape=(None,), dtype=np.int64,
            chunks=(min(chunk_rows, 4096),), compression=compression,
        )
        subject_ds = f.create_dataset(
            "subject", shape=(0,), maxshape=(None,), dtype="S64",
            chunks=(min(chunk_rows, 4096),), compression=compression,
        )
        dataset_ds = f.create_dataset(
            "dataset", shape=(0,), maxshape=(None,), dtype="S32",
            chunks=(min(chunk_rows, 4096),), compression=compression,
        )

        # Fixed-size flush buffer: filled window-by-window straight from the
        # per-record generator and written to HDF5 (a single resize + one
        # contiguous slice-assignment per array) as soon as it's full, then
        # cleared - this is the "process and write directly" streaming path,
        # as opposed to accumulating a whole record's (or the whole corpus's)
        # windows into Python lists before ever touching the HDF5 file.
        buf_x: list[np.ndarray] = []
        buf_y: list[int] = []
        buf_subject: list[bytes] = []
        buf_dataset: list[bytes] = []

        def _flush() -> None:
            nonlocal n_written
            n = len(buf_x)
            if n == 0:
                return
            batch = np.stack(buf_x).astype(np.float32)
            new_size = n_written + n
            ecg_ds.resize(new_size, axis=0)
            label_ds.resize(new_size, axis=0)
            subject_ds.resize(new_size, axis=0)
            dataset_ds.resize(new_size, axis=0)

            ecg_ds[n_written:new_size] = batch
            label_ds[n_written:new_size] = np.array(buf_y, dtype=np.int64)
            subject_ds[n_written:new_size] = np.array(buf_subject, dtype="S64")
            dataset_ds[n_written:new_size] = np.array(buf_dataset, dtype="S32")

            for c in buf_y:
                class_counts[int(c)] = class_counts.get(int(c), 0) + 1
            n_written = new_size
            buf_x.clear()
            buf_y.clear()
            buf_subject.clear()
            buf_dataset.clear()

        for ds in enabled_datasets(cfg):
            local_dir = Path(ds.local_dir)
            if not local_dir.exists():
                print(f"[hdf5] skipping {ds.name}: {local_dir} not found")
                continue
            record_ids = sorted({p.stem for p in local_dir.glob("*.hea")})
            record_ids = [r for r in record_ids if r not in getattr(ds, "exclude_records", [])]

            for rid in tqdm(record_ids, desc=f"[hdf5] {ds.name}"):
                subject_bytes = f"{ds.name}/{rid}".encode()
                dataset_bytes = ds.name.encode()
                any_window = False
                for window, label in _record_windows(ds, rid, cfg, target_fs):
                    any_window = True
                    buf_x.append(window)
                    buf_y.append(label)
                    buf_subject.append(subject_bytes)
                    buf_dataset.append(dataset_bytes)
                    if len(buf_x) >= flush_rows:
                        _flush()
                # Each record's windows are pushed straight into the shared
                # flush buffer above and freed as soon as that buffer is
                # written out - peak RAM never holds more than
                # `hardware.hdf5_flush_rows` windows at a time, regardless of
                # how long any single record is.
                del any_window

        _flush()  # final partial buffer, if any

        f.attrs["target_fs"] = target_fs
        f.attrs["window_length_seconds"] = cfg.preprocessing.windows.length_seconds
        f.attrs["n_windows"] = n_written
        f.attrs["class_counts"] = str(class_counts)
        f.attrs["augmented"] = False  # flipped to True by src.preprocessing.augment_store

    print(f"[hdf5] Class counts: {class_counts}")

    if n_written == 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError("[hdf5] No windows were produced at all - check data/WFDB/ and config.yaml")

    n_classes = sum(1 for v in class_counts.values() if v > 0)
    if n_classes < 2:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"[hdf5] Degenerate dataset: only {n_classes} class present ({class_counts}). "
            "This means every window ended up with the same rhythm label - check "
            "labels.af_aux_tokens / non_af_aux_tokens against the actual aux_note values "
            "in your records before training on this. (The partial HDF5 file has been deleted.)"
        )

    print(f"[hdf5] Wrote {n_written} windows -> {out_path}")


def main() -> None:
    cfg = load_config()
    build_hdf5_dataset(cfg)


if __name__ == "__main__":
    main()
