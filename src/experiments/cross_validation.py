"""
5-fold subject-wise cross-validation.

Uses sklearn's GroupKFold with subject ID as the group (so no subject's
windows ever appear in both the train and test fold - never window-wise CV,
which would leak). For each fold:

  1. the fold's held-in subjects are further subject-split into train/val
     (train_model's early stopping needs a validation set),
  2. LAFNet is trained fresh (no weight-sharing across folds),
  3. evaluated on the fold's held-out test subjects.

Reports per-fold metrics, mean +/- std across folds, a percentile-bootstrap
95% CI on the pooled out-of-fold predictions, and a McNemar's-test
significance check of the pooled predictions against a trivial
majority-class baseline (a sanity check that LAFNet is doing better than
chance in a way that isn't just an artifact of class imbalance).

Writes outputs/cross_validation_results.json and
outputs/cross_validation_report.xlsx
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from src.config import load_config
from src.dataset.loader import build_dataset_from_indices, load_hdf5_metadata, train_val_split_within
from src.experiments.utils import evaluate_dataset, write_rows_xlsx
from src.training.metrics import bootstrap_ci, mcnemar_test
from src.training.train import resolve_device, train_model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


def run_cross_validation() -> None:
    cfg = load_config()
    cv_cfg = cfg.cross_validation
    device = resolve_device(cfg)

    meta = load_hdf5_metadata(cfg)
    subjects, labels = meta["subjects"], meta["labels"]
    indices = np.arange(len(labels))

    gkf = GroupKFold(n_splits=cv_cfg.n_folds)
    fold_rows = []
    pooled_true, pooled_pred, pooled_prob = [], [], []

    for fold_i, (trainval_idx, test_idx) in enumerate(gkf.split(indices, labels, groups=subjects), start=1):
        print(f"[cross_validation] fold {fold_i}/{cv_cfg.n_folds}: "
              f"{len(np.unique(subjects[trainval_idx]))} train+val subjects, "
              f"{len(np.unique(subjects[test_idx]))} test subjects")

        train_idx, val_idx = train_val_split_within(
            trainval_idx, subjects, cfg, seed=cfg.project.seed + fold_i
        )
        train_ds = build_dataset_from_indices(train_idx, cfg, train=True)
        val_ds = build_dataset_from_indices(val_idx, cfg, train=False)
        test_ds = build_dataset_from_indices(test_idx, cfg, train=False)

        model, history, best_val_metric = train_model(
            cfg, train_ds, val_ds,
            epochs=cv_cfg.epochs_override,
            checkpoint_path=None,
            run_label=f"cv_fold{fold_i}",
        )
        model = model.to(device)

        eval_result = evaluate_dataset(model, test_ds, cfg, device)
        metrics = eval_result["metrics"]
        fold_rows.append(
            {
                "fold": fold_i,
                "n_test_windows": int(len(test_idx)),
                "n_test_subjects": int(len(np.unique(subjects[test_idx]))),
                "best_val_metric": best_val_metric,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            }
        )
        pooled_true.append(eval_result["y_true"])
        pooled_pred.append(eval_result["y_pred"])
        pooled_prob.append(eval_result["y_prob"])

    pooled_true = np.concatenate(pooled_true)
    pooled_pred = np.concatenate(pooled_pred)
    pooled_prob = np.concatenate(pooled_prob)

    metric_names = [k for k in fold_rows[0].keys() if k not in ("fold", "n_test_windows", "n_test_subjects", "best_val_metric")]
    summary_mean_std = {
        m: {"mean": float(np.mean([r[m] for r in fold_rows])), "std": float(np.std([r[m] for r in fold_rows]))}
        for m in metric_names
    }

    bootstrap_fns = {
        "accuracy": accuracy_score,
        "precision": lambda a, b: precision_score(a, b, zero_division=0),
        "recall": lambda a, b: recall_score(a, b, zero_division=0),
        "f1": lambda a, b: f1_score(a, b, zero_division=0),
        "mcc": matthews_corrcoef,
    }
    pooled_ci = {}
    for name, fn in bootstrap_fns.items():
        fn.__name__ = name
        pooled_ci[name] = bootstrap_ci(
            pooled_true, pooled_pred, fn, cv_cfg.bootstrap_iterations, cv_cfg.confidence_level
        )

    majority_class = int(np.round(np.mean(pooled_true)))
    baseline_pred = np.full_like(pooled_pred, majority_class)
    mcnemar_vs_baseline = mcnemar_test(pooled_true, pooled_pred, baseline_pred, correction=True)

    results = {
        "n_folds": cv_cfg.n_folds,
        "folds": fold_rows,
        "summary_mean_std": summary_mean_std,
        "pooled_bootstrap_ci": pooled_ci,
        "mcnemar_vs_majority_class_baseline": mcnemar_vs_baseline,
    }

    out_json = Path(cv_cfg.results_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)

    write_rows_xlsx(fold_rows, Path(cv_cfg.results_xlsx), sheet_name="Per-Fold Results")

    print(f"[cross_validation] Mean +/- std across {cv_cfg.n_folds} folds:")
    for m, v in summary_mean_std.items():
        print(f"  {m}: {v['mean']:.4f} +/- {v['std']:.4f}")
    print(f"[cross_validation] Pooled 95% bootstrap CI: {pooled_ci}")
    print(f"[cross_validation] McNemar vs majority-class baseline: {mcnemar_vs_baseline}")
    print(f"[cross_validation] Results -> {out_json}, {cv_cfg.results_xlsx}")


if __name__ == "__main__":
    run_cross_validation()
