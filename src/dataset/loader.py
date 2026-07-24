"""PyTorch Dataset over the LAFNet HDF5 file, with a subject-wise train/val/test split."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import ConfigNode
from src.dataset.augment import augment_ecg


def _hdf5_cache_kwargs(cfg: ConfigNode) -> dict:
    """
    h5py per-file-handle raw chunk cache settings, read from config.yaml's
    data.hdf5_cache_bytes / data.hdf5_cache_slots. Each DataLoader worker
    process opens its own file handle (h5py.File objects can't cross a
    fork/spawn boundary safely), so this cache is per-worker; sizing it well
    above one batch's worth of windows means repeated epoch passes over the
    same rows increasingly hit this cache instead of re-reading (and, for
    compressed datasets, re-decompressing) from disk on every access.
    """
    data_cfg = getattr(cfg, "data", None)
    nbytes = getattr(data_cfg, "hdf5_cache_bytes", None) if data_cfg is not None else None
    nslots = getattr(data_cfg, "hdf5_cache_slots", None) if data_cfg is not None else None
    kwargs = {}
    if nbytes:
        kwargs["rdcc_nbytes"] = int(nbytes)
    if nslots:
        kwargs["rdcc_nslots"] = int(nslots)
    return kwargs


def subject_wise_split(subjects: np.ndarray, cfg: ConfigNode, seed: int | None = None):
    seed = seed if seed is not None else cfg.project.seed
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subjects)
    rng.shuffle(unique_subjects)

    n = len(unique_subjects)
    n_train = int(n * cfg.dataset_split.train_frac)
    n_val = int(n * cfg.dataset_split.val_frac)

    train_subj = set(unique_subjects[:n_train])
    val_subj = set(unique_subjects[n_train: n_train + n_val])
    test_subj = set(unique_subjects[n_train + n_val:])

    train_idx = np.where(np.isin(subjects, list(train_subj)))[0]
    val_idx = np.where(np.isin(subjects, list(val_subj)))[0]
    test_idx = np.where(np.isin(subjects, list(test_subj)))[0]
    return train_idx, val_idx, test_idx


def _fetch_rows(dset: h5py.Dataset, row_idx: np.ndarray) -> np.ndarray:
    """
    Reads an arbitrary (possibly unsorted, possibly with duplicates) set of
    row indices from an HDF5 dataset in ONE call, instead of one call per
    row. h5py's fancy indexing requires a strictly increasing, duplicate-free
    index array for a point selection, so this sorts+dedupes with
    `np.unique(..., return_inverse=True)`, does the single vectorized read
    against that safe array, then reorders/re-duplicates the result back to
    match the caller's original `row_idx` order via the inverse map -
    `uniq[inverse] == row_idx` by construction of np.unique.
    """
    uniq_idx, inverse = np.unique(row_idx, return_inverse=True)
    uniq_vals = dset[uniq_idx]
    return uniq_vals[inverse]


class LAFNetDataset(Dataset):
    def __init__(self, hdf5_path: str | Path, indices: np.ndarray, cfg: ConfigNode, train: bool,
                 group: str = "/"):
        self.hdf5_path = str(hdf5_path)
        self.indices = np.asarray(indices)
        self.cfg = cfg
        self.train = train
        self.group = group  # "/" for the base ecg/label arrays, "augmented" for precomputed copies
        self._file = None  # lazy open (h5py + multiprocessing workers)
        self._ecg = None
        self._label = None

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r", **_hdf5_cache_kwargs(self.cfg))
            node = self._file[self.group] if self.group != "/" else self._file
            self._ecg = node["ecg"]
            self._label = node["label"]

    def __len__(self):
        return len(self.indices)

    def _to_tensor_pair(self, x: np.ndarray, y: int):
        if self.train and self.cfg.augment.enabled and self.group == "/":
            # Only applied when reading the base (non-precomputed) group -
            # once src.preprocessing.augment_store has baked augmented copies
            # into "/augmented", this path is skipped entirely for the
            # default train split (see build_datasets below), removing the
            # per-epoch augmentation cost from the training hot path.
            x = augment_ecg(x, self.cfg)
        x_t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)  # (1, T) - single ECG channel
        y_t = torch.tensor(int(y), dtype=torch.long)
        return x_t, y_t

    def __getitem__(self, i: int):
        self._ensure_open()
        idx = int(self.indices[i])
        # ecg/label are already stored as float32/int64 in the HDF5 file, so
        # this is a plain read with no dtype conversion or extra copy.
        x = self._ecg[idx]
        y = int(self._label[idx])
        return self._to_tensor_pair(x, y)

    def __getitems__(self, indices: list[int]):
        """
        Batched read path: when a Dataset defines __getitems__, PyTorch's
        default auto-batching fetcher (torch>=1.13) calls it ONCE per batch
        with the whole list of sample indices instead of calling
        __getitem__ once per index. For an HDF5-backed dataset this is the
        single biggest lever available at the loader level: a batch_size=512
        step goes from 512 separate Python-level h5py single-row reads (each
        paying its own call overhead and, for a shuffled/weighted sampler,
        its own random disk-chunk lookup) down to exactly ONE vectorized
        fancy-index read per underlying array for the whole batch.
        """
        self._ensure_open()
        row_idx = self.indices[np.asarray(indices, dtype=np.int64)]
        x_batch = _fetch_rows(self._ecg, row_idx)
        y_batch = _fetch_rows(self._label, row_idx)
        return [self._to_tensor_pair(x_batch[j], y_batch[j]) for j in range(len(indices))]

    def labels_only(self) -> np.ndarray:
        """
        Every label for this split's indices, read directly from the HDF5
        label array in one vectorized fetch - used wherever only the class
        distribution is needed (e.g. class-weighting decisions, weighted
        samplers). Avoids the far more expensive path of going through
        __getitem__ for every index, which would also read the full ECG
        window and run augmentation just to throw the signal away.
        """
        with h5py.File(self.hdf5_path, "r") as f:
            node = f[self.group] if self.group != "/" else f
            return node["label"][:][self.indices].astype(np.int64)


class ConcatLAFNetDataset(Dataset):
    """
    Concatenates a base-group dataset (original windows for the train split)
    with a precomputed-augmented-group dataset (extra rows generated once by
    src.preprocessing.augment_store), presenting them as a single Dataset so
    the rest of the training code (DataLoader, weighted sampler, etc.) doesn't
    need to know the underlying windows live in two different HDF5 groups.
    """

    def __init__(self, base: LAFNetDataset, augmented: LAFNetDataset):
        self.base = base
        self.augmented = augmented

    def __len__(self):
        return len(self.base) + len(self.augmented)

    def __getitem__(self, i: int):
        if i < len(self.base):
            return self.base[i]
        return self.augmented[i - len(self.base)]

    def __getitems__(self, indices: list[int]):
        """
        Splits one batch's indices into "belongs to base" / "belongs to
        augmented" groups, fetches each group with a single vectorized
        __getitems__ call against its own HDF5 group, then reassembles the
        results in the original requested order - so batches that mix
        original and precomputed-augmented rows (the common case once
        src.preprocessing.augment_store has run) still cost exactly one
        HDF5 read per underlying array, not one read per sample.
        """
        idx_arr = np.asarray(indices, dtype=np.int64)
        n_base = len(self.base)
        is_base = idx_arr < n_base

        out: list = [None] * len(idx_arr)
        base_positions = np.nonzero(is_base)[0]
        if len(base_positions) > 0:
            base_items = self.base.__getitems__(idx_arr[base_positions].tolist())
            for pos, item in zip(base_positions, base_items):
                out[pos] = item

        aug_positions = np.nonzero(~is_base)[0]
        if len(aug_positions) > 0:
            aug_items = self.augmented.__getitems__((idx_arr[aug_positions] - n_base).tolist())
            for pos, item in zip(aug_positions, aug_items):
                out[pos] = item

        return out

    def labels_only(self) -> np.ndarray:
        return np.concatenate([self.base.labels_only(), self.augmented.labels_only()])


def _read_cached_split(hdf5_path: str | Path):
    """Returns (train_idx, val_idx, test_idx) from the HDF5 file's "/split"
    group if src.preprocessing.augment_store has already cached it, else None."""
    with h5py.File(hdf5_path, "r") as f:
        if "split" not in f:
            return None
        return (
            f["split"]["train_idx"][:],
            f["split"]["val_idx"][:],
            f["split"]["test_idx"][:],
        )


def _is_augmented(hdf5_path: str | Path) -> bool:
    with h5py.File(hdf5_path, "r") as f:
        return bool(f.attrs.get("augmented", False)) and "augmented" in f


def build_datasets(cfg: ConfigNode):
    hdf5_path = cfg.data.hdf5_path
    cached = _read_cached_split(hdf5_path)
    if cached is not None:
        train_idx, val_idx, test_idx = cached
    else:
        print("[loader] No cached /split found in the HDF5 file - computing the subject-wise split "
              "on the fly. Run `python -m src.preprocessing.augment_store` after "
              "`python -m src.preprocessing.hdf5` to cache this split (and precompute "
              "augmentation) once instead of recomputing it on every run.")
        with h5py.File(hdf5_path, "r") as f:
            subjects = np.array([s.decode() for s in f["subject"][:]])
        train_idx, val_idx, test_idx = subject_wise_split(subjects, cfg)

    if _is_augmented(hdf5_path):
        base_train_ds = LAFNetDataset(hdf5_path, train_idx, cfg, train=False, group="/")
        with h5py.File(hdf5_path, "r") as f:
            n_aug = f["augmented"]["ecg"].shape[0]
        aug_ds = LAFNetDataset(hdf5_path, np.arange(n_aug), cfg, train=False, group="augmented")
        train_ds = ConcatLAFNetDataset(base_train_ds, aug_ds)
    else:
        train_ds = LAFNetDataset(hdf5_path, train_idx, cfg, train=True, group="/")

    val_ds = LAFNetDataset(hdf5_path, val_idx, cfg, train=False, group="/")
    test_ds = LAFNetDataset(hdf5_path, test_idx, cfg, train=False, group="/")
    return train_ds, val_ds, test_ds


def load_hdf5_metadata(cfg: ConfigNode) -> dict:
    """Subject IDs, source-database names, and labels for every window - used by
    the cross-database and cross-validation experiment scripts, which need to
    build custom splits beyond the default subject-wise train/val/test split.
    Deliberately reads only the base "/ecg"-adjacent arrays (never
    "/augmented"), so those scripts' custom group-aware splits are completely
    unaffected by the default split's precomputed augmentation."""
    with h5py.File(cfg.data.hdf5_path, "r") as f:
        subjects = np.array([s.decode() for s in f["subject"][:]])
        dataset_names = np.array([s.decode() for s in f["dataset"][:]])
        labels = f["label"][:]
    return {"subjects": subjects, "dataset_names": dataset_names, "labels": labels}


def indices_for_datasets(dataset_names: np.ndarray, wanted: list[str]) -> np.ndarray:
    """Row indices whose source database name is in `wanted` (e.g. ["afdb"])."""
    return np.where(np.isin(dataset_names, wanted))[0]


def train_val_split_within(indices: np.ndarray, subjects: np.ndarray, cfg: ConfigNode, seed: int | None = None):
    """Subject-wise train/val split restricted to a pre-filtered index set
    (e.g. only AFDB rows), used by cross_database.py which trains on one
    database's subjects only, with no held-out test split needed since the
    other databases ARE the test set."""
    seed = seed if seed is not None else cfg.project.seed
    rng = np.random.default_rng(seed)
    local_subjects = subjects[indices]
    unique_subjects = np.unique(local_subjects)
    rng.shuffle(unique_subjects)

    val_frac = cfg.dataset_split.val_frac / (cfg.dataset_split.train_frac + cfg.dataset_split.val_frac)
    n_val = max(int(len(unique_subjects) * val_frac), 1)
    val_subj = set(unique_subjects[:n_val])
    train_subj = set(unique_subjects[n_val:])

    train_idx = indices[np.isin(local_subjects, list(train_subj))]
    val_idx = indices[np.isin(local_subjects, list(val_subj))]
    return train_idx, val_idx


def build_dataset_from_indices(indices: np.ndarray, cfg: ConfigNode, train: bool) -> LAFNetDataset:
    return LAFNetDataset(cfg.data.hdf5_path, indices, cfg, train=train, group="/")
