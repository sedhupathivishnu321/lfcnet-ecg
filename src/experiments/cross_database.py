"""
Cross-database evaluation.

Trains LAFNet on one database's subjects only (config.yaml:
cross_database.train_on, default ["afdb"]) and evaluates it, with NO
fine-tuning, on each of the other databases (cross_database.test_on, default
["ltafdb", "shdb-af"]) plus its own in-domain held-out validation subjects,
so you can directly see how much performance degrades when the rhythm model
has to generalize across different recording hardware/populations rather
than just across held-out subjects of the same database.

Writes outputs/cross_database_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.dataset.loader import (
    build_dataset_from_indices,
    indices_for_datasets,
    load_hdf5_metadata,
    train_val_split_within,
)
from src.experiments.utils import evaluate_dataset
from src.training.metrics import mcnemar_test
from src.training.train import resolve_device, train_model


def run_cross_database() -> None:
    cfg = load_config()
    cd = cfg.cross_database
    device = resolve_device(cfg)

    meta = load_hdf5_metadata(cfg)
    subjects, dataset_names = meta["subjects"], meta["dataset_names"]

    train_domain_idx = indices_for_datasets(dataset_names, list(cd.train_on))
    if len(train_domain_idx) == 0:
        raise RuntimeError(
            f"[cross_database] No windows found for training databases {cd.train_on} - "
            "check that data/lafnet_dataset.h5 actually contains those datasets."
        )

    train_idx, val_idx = train_val_split_within(train_domain_idx, subjects, cfg)
    train_ds = build_dataset_from_indices(train_idx, cfg, train=True)
    val_ds = build_dataset_from_indices(val_idx, cfg, train=False)

    weights_dir = Path(cfg.paths.weights_dir)
    model, history, best_val_metric = train_model(
        cfg, train_ds, val_ds,
        epochs=cd.epochs_override,
        checkpoint_path=weights_dir / "lafnet_cross_database_best.pt",
        run_label="cross_db",
    )
    model = model.to(device)

    in_domain_eval = evaluate_dataset(model, val_ds, cfg, device)

    results = {
        "train_on": list(cd.train_on),
        "test_on": list(cd.test_on),
        "best_val_metric": best_val_metric,
        "n_epochs_run": len(history),
        "in_domain_val": in_domain_eval["metrics"],
        "cross_database": {},
    }

    predictions_by_domain = {}
    for test_db in cd.test_on:
        test_idx = indices_for_datasets(dataset_names, [test_db])
        if len(test_idx) == 0:
            print(f"[cross_database] WARNING: no windows found for '{test_db}' - skipping")
            continue
        test_ds = build_dataset_from_indices(test_idx, cfg, train=False)
        eval_result = evaluate_dataset(model, test_ds, cfg, device)
        results["cross_database"][test_db] = eval_result["metrics"]
        predictions_by_domain[test_db] = eval_result

    # Significance check: is the model's performance on each external database
    # statistically distinguishable from its in-domain performance? (paired
    # McNemar's needs the same samples for both arms, so this compares each
    # external database's predictions against the in-domain validation
    # predictions, truncated to matching length, as an illustrative
    # generalization-gap check for the same trained model.)
    results["significance_vs_in_domain"] = {}
    for db_name, eval_result in predictions_by_domain.items():
        n = min(len(in_domain_eval["y_true"]), len(eval_result["y_true"]))
        mc = mcnemar_test(
            eval_result["y_true"][:n],
            eval_result["y_pred"][:n],
            in_domain_eval["y_pred"][:n],
            correction=True,
        )
        results["significance_vs_in_domain"][db_name] = mc

    out_path = Path(cd.results_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"[cross_database] In-domain val: {results['in_domain_val']}")
    for db_name, m in results["cross_database"].items():
        print(f"[cross_database] Zero-shot on {db_name}: {m}")
    print(f"[cross_database] Results -> {out_path}")


if __name__ == "__main__":
    run_cross_database()
