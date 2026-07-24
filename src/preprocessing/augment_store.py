"""
Phase 7 (optional, run after src.preprocessing.hdf5): precomputes augmented
copies of the DEFAULT subject-wise TRAIN split's windows exactly once, and
caches the train/val/test row indices themselves, both stored in the same
HDF5 file - instead of paying either cost repeatedly at training time.

Two separate problems this removes from the training-time hot path:

1. Augmentation cost. src/dataset/augment.py's add_noise / random_shift /
   random_scale / random_stretch / random_crop are individually cheap, but
   the original design ran a random subset of them, in Python, inside
   LAFNetDataset.__getitem__, for every training window on every one of
   training.epochs (100 by default) passes. That's a lot of repeated CPU work
   producing output that (for a fixed compute budget) doesn't need to be
   regenerated from scratch every epoch. This module runs augment_ecg()
   augment.copies_per_train_window times per train-split window, ONCE, and
   stores the results as extra rows in a separate "/augmented" HDF5 group.

2. Split-computation cost. subject_wise_split() (src/dataset/loader.py) is
   cheap in isolation, but every training/ablation/experiment run that reuses
   the default split was recomputing it from scratch. This module runs it
   once and caches train_idx/val_idx/test_idx under "/split" in the HDF5
   file; src/dataset/loader.py::build_datasets reads them directly if present.

Deliberately scoped to the DEFAULT split only: the ablation / cross-validation
/ cross-database experiment scripts build their own custom index sets (via
GroupKFold, per-database filtering, etc.) that don't correspond 1:1 with this
split, and mixing these precomputed augmented rows into their group-aware
folds could bias or leak across them. Those scripts keep using the original,
unchanged on-the-fly augmentation path in LAFNetDataset(train=True).

Idempotent: safe to re-run (e.g. after changing augment.* settings) - it
deletes and rebuilds both the "/augmented" group and the "/split" group.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.config import ConfigNode, load_config
from src.dataset.augment import augment_ecg
from src.dataset.loader import subject_wise_split


def precompute_augmented_split(cfg: ConfigNode) -> None:
    hdf5_path = Path(cfg.data.hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(
            f"[augment_store] {hdf5_path} not found - run `python -m src.preprocessing.hdf5` first."
        )

    copies = max(int(getattr(cfg.augment, "copies_per_train_window", 0)), 0)
    hw = getattr(cfg, "hardware", None)
    flush_rows = max(getattr(hw, "hdf5_flush_rows", 1024) if hw else 1024, 1)
    compression = getattr(cfg.data, "hdf5_compression", "lzf")
    compression = compression if compression not in (None, "none", "None") else None

    with h5py.File(hdf5_path, "r+") as f:
        subjects = np.array([s.decode() for s in f["subject"][:]])
        win_len = f["ecg"].shape[1]

        train_idx, val_idx, test_idx = subject_wise_split(subjects, cfg)

        # Cache the split itself regardless of whether augmentation is enabled -
        # this is what removes the repeated subject_wise_split() overhead from
        # every training/loader startup.
        if "split" in f:
            del f["split"]
        split_grp = f.create_group("split")
        split_grp.create_dataset("train_idx", data=train_idx.astype(np.int64))
        split_grp.create_dataset("val_idx", data=val_idx.astype(np.int64))
        split_grp.create_dataset("test_idx", data=test_idx.astype(np.int64))
        print(f"[augment_store] Cached split -> /split "
              f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

        if "augmented" in f:
            del f["augmented"]

        if not cfg.augment.enabled or copies == 0:
            f.attrs["augmented"] = False
            print("[augment_store] augment.enabled is false or copies_per_train_window=0 - "
                  "no precomputed augmented rows written; training will fall back to "
                  "on-the-fly per-epoch augmentation.")
            return

        n_aug_total = len(train_idx) * copies
        grp = f.create_group("augmented")
        aug_ecg = grp.create_dataset(
            "ecg", shape=(0, win_len), maxshape=(None, win_len),
            dtype=np.float32, chunks=(min(flush_rows, max(n_aug_total, 1)), win_len),
            compression=compression,
        )
        aug_label = grp.create_dataset(
            "label", shape=(0,), maxshape=(None,), dtype=np.int64,
            chunks=(min(flush_rows, max(n_aug_total, 1)),), compression=compression,
        )

        buf_x: list[np.ndarray] = []
        buf_y: list[int] = []
        written = 0

        def _flush() -> None:
            nonlocal written
            n = len(buf_x)
            if n == 0:
                return
            batch = np.stack(buf_x).astype(np.float32)
            new_size = written + n
            aug_ecg.resize(new_size, axis=0)
            aug_label.resize(new_size, axis=0)
            aug_ecg[written:new_size] = batch
            aug_label[written:new_size] = np.array(buf_y, dtype=np.int64)
            written = new_size
            buf_x.clear()
            buf_y.clear()

        base_ecg = f["ecg"]
        base_label = f["label"]
        for idx in tqdm(train_idx, desc="[augment_store] precomputing augmented train windows"):
            x = base_ecg[int(idx)].astype(np.float32)
            y = int(base_label[int(idx)])
            for _ in range(copies):
                buf_x.append(augment_ecg(x, cfg))
                buf_y.append(y)
                if len(buf_x) >= flush_rows:
                    _flush()
        _flush()

        f.attrs["augmented"] = True
        f.attrs["augment_copies_per_train_window"] = copies
        print(f"[augment_store] Stored {written} precomputed augmented windows "
              f"({copies}x over {len(train_idx)} base train windows) -> /augmented.")


def main() -> None:
    cfg = load_config()
    precompute_augmented_split(cfg)


if __name__ == "__main__":
    main()
