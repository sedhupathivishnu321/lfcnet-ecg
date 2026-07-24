"""
Ablation study runner.

Sweeps, one axis at a time (config.yaml: ablation.full_grid=false, the
default) or as a full Cartesian grid (ablation.full_grid=true, expensive):

  - preprocessing.filter.type:        butterworth | bessel | cheby1 | fir
  - preprocessing.windows.length_seconds: 5 | 10 | 15
  - model.use_attention:              true | false
  - model.use_se:                     true | false
  - sampler.imbalance_strategy:       smote | weighted_sampler

For each variant, trains a (shorter, ablation.epochs_override) LAFNet run and
evaluates on the held-out test split, recording every classification +
efficiency metric plus a bootstrap CI, to outputs/ablation_results.csv and
outputs/ablation_report.xlsx.

filter_type and window_seconds axes require re-deriving the HDF5 dataset
(they change the input signal itself), so each such variant - or
combination of variants, in full_grid mode - is cached under
data/ablation_cache/<axis-value>_<axis-value>...h5 and only rebuilt once.
"""
from __future__ import annotations

import copy
from itertools import product
from pathlib import Path

from src.config import load_config
from src.dataset.loader import build_datasets
from src.experiments.utils import evaluate_dataset, write_rows_csv, write_rows_xlsx
from src.preprocessing.hdf5 import build_hdf5_dataset
from src.training.metrics import profile_efficiency
from src.training.train import resolve_device, train_model

AXES_REQUIRING_REBUILD = {"filter_type", "window_seconds"}


def _apply_override(cfg, axis: str, value):
    if axis == "filter_type":
        cfg.preprocessing.filter.type = value
    elif axis == "window_seconds":
        cfg.preprocessing.windows.length_seconds = value
    elif axis == "use_attention":
        cfg.model.use_attention = value
    elif axis == "use_se":
        cfg.model.use_se = value
    elif axis == "imbalance_strategy":
        cfg.sampler.imbalance_strategy = value
    else:
        raise ValueError(f"Unknown ablation axis: {axis}")
    return cfg


def _set_cache_path(cfg, overrides: dict) -> None:
    """
    Points cfg.data.hdf5_path at a cache file keyed by every axis in this
    variant that changes the input signal itself (filter_type, window_seconds).
    Using the *combination* (not just the last-applied axis) avoids two
    different signal-processing variants silently colliding on the same
    cached HDF5 file when both axes vary together, e.g. in full_grid mode.
    """
    rebuild_parts = [f"{axis}-{overrides[axis]}" for axis in sorted(AXES_REQUIRING_REBUILD) if axis in overrides]
    if rebuild_parts:
        cfg.data.hdf5_path = "data/ablation_cache/" + "_".join(rebuild_parts) + ".h5"


def _ensure_dataset_built(cfg) -> None:
    if not Path(cfg.data.hdf5_path).exists():
        print(f"[ablation] building dataset variant -> {cfg.data.hdf5_path}")
        build_hdf5_dataset(cfg)


def run_variant(base_cfg, axis: str, value, epochs: int | None, run_label: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg = _apply_override(cfg, axis, value)
    _set_cache_path(cfg, {axis: value})
    _ensure_dataset_built(cfg)

    train_ds, val_ds, test_ds = build_datasets(cfg)
    device = resolve_device(cfg)

    imbalance_override = value if axis == "imbalance_strategy" else None
    model, history, best_val_metric = train_model(
        cfg, train_ds, val_ds, epochs=epochs, imbalance_strategy_override=imbalance_override,
        checkpoint_path=None, show_progress=True, run_label=run_label,
    )
    model = model.to(device)

    result = evaluate_dataset(model, test_ds, cfg, device)
    fs = cfg.preprocessing.target_fs
    win_len = int(cfg.preprocessing.windows.length_seconds * fs)
    efficiency = profile_efficiency(model, (cfg.model.in_channels, win_len), cfg)

    row = {
        "axis": axis,
        "value": value,
        "n_epochs_run": len(history),
        "best_val_metric": best_val_metric,
        **{f"test_{k}": v for k, v in result["metrics"].items() if k != "confusion_matrix"},
        "params": efficiency["params"],
        "macs": efficiency["macs"],
        "flops": efficiency["flops"],
        "latency_ms_cpu": efficiency["latency_ms_cpu"],
    }
    return row


def run_ablation() -> None:
    base_cfg = load_config()
    a = base_cfg.ablation
    epochs = a.epochs_override

    rows = []
    if a.full_grid:
        axis_names = list(a.axes.__dict__.keys())
        value_lists = [getattr(a.axes, name) for name in axis_names]
        for combo in product(*value_lists):
            label = "_".join(f"{n}={v}" for n, v in zip(axis_names, combo))
            print(f"[ablation] full-grid variant: {label}")
            cfg = copy.deepcopy(base_cfg)
            overrides = dict(zip(axis_names, combo))
            for axis, value in overrides.items():
                cfg = _apply_override(cfg, axis, value)
            _set_cache_path(cfg, overrides)
            _ensure_dataset_built(cfg)
            train_ds, val_ds, test_ds = build_datasets(cfg)
            device = resolve_device(cfg)
            imbalance_override = overrides.get("imbalance_strategy")
            model, history, best_val_metric = train_model(
                cfg, train_ds, val_ds, epochs=epochs, imbalance_strategy_override=imbalance_override,
                checkpoint_path=None, run_label=label,
            )
            model = model.to(device)
            result = evaluate_dataset(model, test_ds, cfg, device)
            fs = cfg.preprocessing.target_fs
            win_len = int(cfg.preprocessing.windows.length_seconds * fs)
            efficiency = profile_efficiency(model, (cfg.model.in_channels, win_len), cfg)
            row = {"variant": label, **overrides}
            row.update({f"test_{k}": v for k, v in result["metrics"].items() if k != "confusion_matrix"})
            row.update({"params": efficiency["params"], "flops": efficiency["flops"]})
            rows.append(row)
    else:
        for axis, values in a.axes.__dict__.items():
            for value in values:
                print(f"[ablation] axis={axis} value={value}")
                rows.append(run_variant(base_cfg, axis, value, epochs, run_label=f"{axis}={value}"))

    write_rows_csv(rows, Path(a.results_csv))
    write_rows_xlsx(rows, Path(a.results_xlsx), sheet_name="Ablation")
    print(f"[ablation] {len(rows)} variants -> {a.results_csv}, {a.results_xlsx}")


if __name__ == "__main__":
    run_ablation()
