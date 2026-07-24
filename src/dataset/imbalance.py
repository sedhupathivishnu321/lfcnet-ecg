"""
Unifies the two imbalance-handling strategies referenced throughout the
project (`sampler.imbalance_strategy` in config.yaml: "weighted_sampler" |
"smote" | "none") behind one function, so train.py and the ablation runner
build a DataLoader the same way regardless of which strategy is active.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ConfigNode
from src.dataset.sampler import build_weighted_sampler

SMOTE_MEMORY_SAFETY_FACTOR = 3.0  # SMOTE needs the original array PLUS the resampled (oversampled) array
                                    # in memory at once; oversampling the minority class up to parity can
                                    # roughly double the row count, so budget generously rather than exactly.


def _worker_init_fn(_worker_id: int) -> None:
    """Pins each DataLoader worker process to a single torch/BLAS thread, so
    training.num_workers worker processes can't each independently spin up a
    full per-core thread pool and oversubscribe the CPU alongside the main
    process's own torch threads (see the OMP_NUM_THREADS block + this same
    worker_init_fn use in src/training/train.py)."""
    torch.set_num_threads(1)


def _materialize(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Pull every (x, y) pair out of a Dataset into flat numpy arrays (train split only - small enough to fit in memory for SMOTE)."""
    xs, ys = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x.numpy().reshape(-1))
        ys.append(int(y))
    return np.stack(xs), np.array(ys)


def _estimate_smote_ram_gb(n_windows: int, win_len: int) -> float:
    bytes_per_window = win_len * 4  # float32
    return (n_windows * bytes_per_window * SMOTE_MEMORY_SAFETY_FACTOR) / (1024 ** 3)


def _window_length(dataset) -> int:
    x, _y = dataset[0]
    return int(x.numel())


def _loader_perf_kwargs(cfg: ConfigNode, num_workers: int) -> dict:
    """
    Shared DataLoader kwargs that keep the CPU cores busy on a CPU-only box:
    persistent_workers avoids tearing down and re-forking (and re-opening
    every worker's own h5py file handle) at the start of every epoch, and
    prefetch_factor keeps several batches queued per worker so the training
    loop is rarely waiting on the next batch. Both only apply (and are only
    valid to pass to DataLoader) when num_workers > 0.
    """
    if num_workers <= 0:
        return {}
    training_cfg = getattr(cfg, "training", None)
    persistent = bool(getattr(training_cfg, "persistent_workers", True)) if training_cfg else True
    prefetch = int(getattr(training_cfg, "prefetch_factor", 4)) if training_cfg else 4
    return {"persistent_workers": persistent, "prefetch_factor": prefetch, "worker_init_fn": _worker_init_fn}


def build_train_loader(
    train_ds,
    cfg: ConfigNode,
    batch_size: int,
    num_workers: int,
    strategy_override: str | None = None,
    precomputed_labels: np.ndarray | None = None,
) -> DataLoader:
    strategy = strategy_override or cfg.sampler.imbalance_strategy
    perf_kwargs = _loader_perf_kwargs(cfg, num_workers)

    if strategy == "weighted_sampler":
        if precomputed_labels is not None:
            labels = precomputed_labels
        elif hasattr(train_ds, "labels_only"):
            labels = train_ds.labels_only()
        else:
            labels = np.array([train_ds[i][1].item() for i in range(len(train_ds))])
        sampler = build_weighted_sampler(labels)
        return DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, **perf_kwargs
        )

    if strategy == "smote":
        hw = getattr(cfg, "hardware", None)
        budget_gb = getattr(hw, "max_smote_ram_gb", None) if hw is not None else None
        if budget_gb is not None:
            win_len = _window_length(train_ds)
            estimated_gb = _estimate_smote_ram_gb(len(train_ds), win_len)
            if estimated_gb > budget_gb:
                print(
                    f"[imbalance] WARNING: SMOTE would materialize an estimated ~{estimated_gb:.1f} GB "
                    f"(train split: {len(train_ds)} windows x {win_len} samples, with oversampling "
                    f"headroom), above hardware.max_smote_ram_gb={budget_gb} GB. Falling back to "
                    "weighted_sampler instead of risking an OOM kill. Lower "
                    "hardware.max_smote_ram_gb's margin or shrink preprocessing.windows.length_seconds "
                    "if you specifically need SMOTE here."
                )
                labels = np.array([train_ds[i][1].item() for i in range(len(train_ds))])
                sampler = build_weighted_sampler(labels)
                return DataLoader(
                    train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, **perf_kwargs
                )

        from imblearn.over_sampling import SMOTE

        X, y = _materialize(train_ds)
        smote = SMOTE(k_neighbors=cfg.sampler.smote_k_neighbors, random_state=cfg.project.seed)
        X_res, y_res = smote.fit_resample(X, y)
        X_t = torch.from_numpy(X_res.astype(np.float32)).unsqueeze(1)  # (N, 1, T)
        y_t = torch.from_numpy(y_res.astype(np.int64))
        tensor_ds = TensorDataset(X_t, y_t)
        # num_workers=0 here: TensorDataset already holds everything in RAM as
        # dense tensors, so there's no HDF5/file-handle work to parallelize.
        return DataLoader(tensor_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    if strategy == "none":
        return DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, **perf_kwargs
        )

    raise ValueError(f"Unknown sampler.imbalance_strategy: {strategy}")
