"""
Phase: train LAFNet.

AdamW + cosine LR schedule (with warmup) + early stopping on a configurable
validation metric, with a live per-batch progress bar for every epoch AND an
overall run progress bar, e.g.:

  Epoch 003/100: 45%|====>     | loss=0.4213 pct=45%
  Overall training progress: 3%|>   | overall_pct=3.0% val_f1=0.81 val_mcc=0.64

`train_model()` is the reusable core used directly by this script's CLI entry
point AND by the ablation / cross-validation / cross-database experiment
scripts under src/experiments/, so all of them share exactly one training
loop implementation instead of drifting apart.
"""
from __future__ import annotations

import os

# MUST run before numpy/scipy/torch are imported: those libraries size their
# internal BLAS/OpenMP thread pools at import time. Left at their defaults,
# the main process alone would try to use every vCPU for its own tensor ops
# while training.num_workers DataLoader worker PROCESSES each separately
# spin up their own such thread pool for the numpy/h5py calls in reading and
# collating a batch. On an 8 vCPU box that's easily 20-50+ threads all
# fighting over 8 cores - the OS spends more time context-switching than any
# core spends computing, and everything (model compute AND data loading)
# gets dramatically slower than an unparallelized run would be. This is a
# frequent cause of a CPU training run taking far longer than expected even
# after the data pipeline itself has been optimized. Workers get 1 thread
# each (they're I/O/light-numpy bound, not compute bound); the main process's
# torch thread count is set explicitly, after config is loaded below, to the
# cores actually left over.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.dataset.imbalance import build_train_loader
from src.dataset.loader import build_datasets
from src.models.lightweight_model import LAFNet, count_parameters
from src.training.losses import build_loss
from src.training.validate import validate


def _worker_init_fn(_worker_id: int) -> None:
    """Belt-and-suspenders: pin each DataLoader worker process to a single
    torch/BLAS thread even if it imports torch itself (e.g. for the
    torch.from_numpy call in LAFNetDataset.__getitem__), so it can't grow
    back into a multi-threaded pool after fork."""
    torch.set_num_threads(1)


def resolve_device(cfg) -> torch.device:
    if cfg.training.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.training.device)


def resolve_num_workers(cfg) -> int:
    """
    Caps DataLoader worker processes to min(training.num_workers,
    hardware.vcpus - 1, os.cpu_count()) - leaving one core free for the main
    process/OS - rather than trusting training.num_workers blindly. On the
    target 8 vCPU / 32 GB instance this keeps preprocessing worker processes
    (each holding its own h5py file handle + augmentation buffers) from
    oversubscribing the machine if config.yaml's num_workers is ever raised
    without checking the actual instance size.
    """
    requested = cfg.training.num_workers
    detected = os.cpu_count() or 1
    hw = getattr(cfg, "hardware", None)
    hw_cap = max(hw.vcpus - 1, 1) if hw is not None else detected
    return max(min(requested, hw_cap, detected), 0)


def _cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_model(
    cfg,
    train_ds,
    val_ds,
    epochs: int | None = None,
    imbalance_strategy_override: str | None = None,
    checkpoint_path: Path | None = None,
    show_progress: bool = True,
    run_label: str = "train",
):
    """
    Reusable training core. Returns (model, history, best_metric).

    - `epochs`: overrides cfg.training.epochs (used by ablation/CV for shorter runs).
    - `imbalance_strategy_override`: overrides cfg.sampler.imbalance_strategy
      (used by the imbalance-strategy ablation axis: "smote" vs "weighted_sampler").
    - `checkpoint_path`: where to save the best model; if None, no checkpoint is saved
      (used by ablation sweeps, which only care about the metrics, not the weights).
    """
    torch.manual_seed(cfg.project.seed)
    device = resolve_device(cfg)
    total_epochs = epochs or cfg.training.epochs
    num_workers = resolve_num_workers(cfg)
    strategy = imbalance_strategy_override or cfg.sampler.imbalance_strategy

    if device.type == "cpu":
        hw = getattr(cfg, "hardware", None)
        total_cores = (hw.vcpus if hw is not None else os.cpu_count()) or 1
        # Workers are pinned to 1 thread each (see _worker_init_fn / the
        # OMP_NUM_THREADS block at module import time above); give the main
        # process - the only one doing model forward/backward - the cores
        # they're not using, so total concurrent threads stays <= total_cores
        # instead of (main using ALL cores) + (num_workers more processes
        # each ALSO trying to use every core).
        main_threads = max(total_cores - num_workers, 1)
        torch.set_num_threads(main_threads)
        if show_progress:
            print(f"[{run_label}] CPU training: {num_workers} DataLoader workers (1 thread each) + "
                  f"main process using {main_threads} torch threads (hardware.vcpus={total_cores}).")

    # Scan the ORIGINAL train_ds labels exactly once, directly from HDF5 (no
    # augmentation overhead) - reused both for the weighted_sampler branch
    # below (instead of build_train_loader scanning it again independently)
    # and for the loss class-weighting decision.
    original_labels = train_ds.labels_only()

    train_loader = build_train_loader(
        train_ds, cfg, cfg.training.batch_size, num_workers, strategy,
        precomputed_labels=original_labels if strategy == "weighted_sampler" else None,
    )
    val_loader_kwargs = {"worker_init_fn": _worker_init_fn} if num_workers > 0 else {}
    if num_workers > 0:
        val_loader_kwargs["persistent_workers"] = bool(getattr(cfg.training, "persistent_workers", True))
        val_loader_kwargs["prefetch_factor"] = int(getattr(cfg.training, "prefetch_factor", 4))
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=num_workers, **val_loader_kwargs
    )

    model = LAFNet(cfg).to(device)
    if show_progress:
        print(f"[{run_label}] LAFNet parameters: {count_parameters(model):,}")

    # weighted_sampler and smote already correct for class imbalance at the
    # data level - additionally class-weighting the loss on top would stack a
    # second, redundant correction, so that's only applied when neither is
    # active ("none").
    class_weight = None
    if strategy == "none":
        counts = np.bincount(original_labels, minlength=2)
        if counts.sum() > 0:
            weights = counts.sum() / (2.0 * np.clip(counts, 1, None))
            class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_fn = build_loss(cfg, class_weight=class_weight)

    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = _cosine_with_warmup(optimizer, cfg.training.warmup_epochs, total_epochs)

    history = []
    best_metric = -np.inf
    patience_left = cfg.training.early_stopping_patience
    best_state = None

    overall_bar = tqdm(total=total_epochs, desc=f"[{run_label}] Overall training progress", disable=not show_progress)

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_losses = []
        batch_bar = tqdm(
            train_loader,
            desc=f"[{run_label}] Epoch {epoch:03d}/{total_epochs}",
            leave=False,
            disable=not show_progress,
        )
        for x, y in batch_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()
            epoch_losses.append(loss.item())
            if show_progress:
                pct = int(100 * (batch_bar.n + 1) / len(train_loader))
                batch_bar.set_postfix_str(f"loss={np.mean(epoch_losses):.4f} pct={pct}%")

        scheduler.step()
        val_metrics = validate(model, val_loader, loss_fn, device)

        if show_progress:
            overall_pct = 100.0 * epoch / total_epochs
            overall_bar.set_postfix_str(
                f"overall_pct={overall_pct:.1f}% val_f1={val_metrics['f1']:.2f} val_mcc={val_metrics['mcc']:.2f}"
            )
        overall_bar.update(1)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_mcc": val_metrics["mcc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        current_metric = val_metrics[cfg.training.early_stopping_metric.replace("val_", "")]
        if current_metric > best_metric:
            best_metric = current_metric
            patience_left = cfg.training.early_stopping_patience
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                if show_progress:
                    print(f"[{run_label}] Early stopping at epoch {epoch} (no improvement in {cfg.training.early_stopping_metric})")
                break

    overall_bar.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_metric


def train() -> None:
    cfg = load_config()
    train_ds, val_ds, _test_ds = build_datasets(cfg)

    weights_dir = Path(cfg.paths.weights_dir)
    logs_dir = Path(cfg.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model, history, best_metric = train_model(
        cfg, train_ds, val_ds, checkpoint_path=weights_dir / "lafnet_best.pt", run_label="train"
    )

    with open(logs_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), weights_dir / "lafnet_last.pt")
    print(f"[train] Done. Best {cfg.training.early_stopping_metric}={best_metric:.4f}. "
          f"History -> {logs_dir / 'training_history.json'}")


if __name__ == "__main__":
    train()
