"""
CPU/I-O profiling harness for the LAFNet pipeline.

Profiles the three stages the optimization effort targeted - HDF5 random
reads, DataLoader throughput (with augmentation on vs. off), and a short
slice of actual training - so any remaining hotspot in data loading,
augmentation, or HDF5 access shows up as concrete numbers instead of guesswork.

Usage:
    python -m scripts.profile_pipeline                  # all stages, cProfile
    python -m scripts.profile_pipeline --stage hdf5      # just raw HDF5 read throughput
    python -m scripts.profile_pipeline --stage loader    # just DataLoader throughput
    python -m scripts.profile_pipeline --stage train     # a handful of real training steps
    python -m scripts.profile_pipeline --batches 50      # profile more/fewer batches

For a live, whole-process view (including native/C extension time that
cProfile under-samples, e.g. inside h5py/HDF5 itself), run this under py-spy
instead, which needs no code changes:
    py-spy record -o profile.svg -- python -m scripts.profile_pipeline --stage train
    py-spy top -- python -m src.training.train
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np
import torch

from src.config import load_config
from src.dataset.imbalance import build_train_loader
from src.dataset.loader import build_datasets
from src.models.lightweight_model import LAFNet
from src.training.losses import build_loss
from src.training.train import resolve_device, resolve_num_workers


def _report(profiler: cProfile.Profile, top_n: int = 25) -> None:
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(top_n)
    print(stream.getvalue())


def profile_hdf5_reads(cfg, n_reads: int = 2000) -> None:
    """Raw random-access read throughput straight off the HDF5 file - isolates
    HDF5/compression cost from everything else (augmentation, collation,
    model forward/backward)."""
    import h5py

    print(f"[profile] HDF5 random reads: n={n_reads}")
    with h5py.File(cfg.data.hdf5_path, "r") as f:
        n_rows = f["ecg"].shape[0]
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n_rows, size=min(n_reads, n_rows))

        profiler = cProfile.Profile()
        profiler.enable()
        t0 = time.perf_counter()
        for i in idx:
            _ = f["ecg"][int(i)]
            _ = f["label"][int(i)]
        elapsed = time.perf_counter() - t0
        profiler.disable()

    print(f"[profile] {len(idx)} reads in {elapsed:.3f}s -> {len(idx) / elapsed:.1f} reads/sec")
    _report(profiler)


def profile_loader(cfg, n_batches: int = 20) -> None:
    """DataLoader throughput end-to-end (HDF5 read + collation + whatever
    augmentation path is active for this HDF5 file's current state)."""
    print(f"[profile] DataLoader throughput: n_batches={n_batches}, batch_size={cfg.training.batch_size}")
    train_ds, _val_ds, _test_ds = build_datasets(cfg)
    num_workers = resolve_num_workers(cfg)
    original_labels = train_ds.labels_only()
    loader = build_train_loader(
        train_ds, cfg, cfg.training.batch_size, num_workers,
        precomputed_labels=original_labels if cfg.sampler.imbalance_strategy == "weighted_sampler" else None,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.perf_counter()
    seen = 0
    for x, _y in loader:
        seen += x.shape[0]
        if seen // cfg.training.batch_size >= n_batches:
            break
    elapsed = time.perf_counter() - t0
    profiler.disable()

    print(f"[profile] {seen} windows in {elapsed:.3f}s -> {seen / elapsed:.1f} windows/sec "
          f"(num_workers={num_workers})")
    _report(profiler)


def profile_train_steps(cfg, n_batches: int = 20) -> None:
    """A handful of real forward+backward steps, same code path as
    src.training.train, to see where wall-clock actually goes once the model
    is in the loop too (not just data loading in isolation)."""
    print(f"[profile] Real training steps: n_batches={n_batches}")
    device = resolve_device(cfg)
    train_ds, _val_ds, _test_ds = build_datasets(cfg)
    num_workers = resolve_num_workers(cfg)
    original_labels = train_ds.labels_only()
    loader = build_train_loader(
        train_ds, cfg, cfg.training.batch_size, num_workers,
        precomputed_labels=original_labels if cfg.sampler.imbalance_strategy == "weighted_sampler" else None,
    )

    model = LAFNet(cfg).to(device)
    loss_fn = build_loss(cfg, class_weight=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.perf_counter()
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        seen += x.shape[0]
        if seen // cfg.training.batch_size >= n_batches:
            break
    elapsed = time.perf_counter() - t0
    profiler.disable()

    print(f"[profile] {seen} windows (train steps) in {elapsed:.3f}s -> {seen / elapsed:.1f} windows/sec")
    _report(profiler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "hdf5", "loader", "train"], default="all")
    parser.add_argument("--batches", type=int, default=20, help="batches/reads to profile per stage")
    args = parser.parse_args()

    cfg = load_config()

    if args.stage in ("all", "hdf5"):
        profile_hdf5_reads(cfg, n_reads=args.batches * cfg.training.batch_size)
    if args.stage in ("all", "loader"):
        profile_loader(cfg, n_batches=args.batches)
    if args.stage in ("all", "train"):
        profile_train_steps(cfg, n_batches=args.batches)


if __name__ == "__main__":
    main()
