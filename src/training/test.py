"""
Phase: evaluate LAFNet on the held-out test split; writes:
  outputs/test_report.json
  outputs/plots/*.png
  outputs/lafnet_report.xlsx
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset.loader import build_datasets
from src.models.lightweight_model import LAFNet
from src.training.losses import build_loss
from src.training.metrics import (
    bootstrap_ci_all_metrics,
    classification_metrics,
    profile_efficiency,
)
from src.training.report import generate_report


def test() -> None:
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _train_ds, _val_ds, test_ds = build_datasets(cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    model = LAFNet(cfg).to(device)
    weights_path = Path(cfg.paths.weights_dir) / "lafnet_best.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"[test] WARNING: {weights_path} not found - evaluating randomly initialized weights")

    loss_fn = build_loss(cfg)

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    metrics = classification_metrics(y_true, y_pred, y_prob)

    fs = cfg.preprocessing.target_fs
    win_len = int(cfg.preprocessing.windows.length_seconds * fs)
    input_shape = (cfg.model.in_channels, win_len)

    efficiency = profile_efficiency(model, input_shape, cfg)
    confidence_intervals = bootstrap_ci_all_metrics(y_true, y_pred, cfg)

    history_path = Path(cfg.paths.logs_dir) / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    report = {
        "test_metrics": metrics,
        "efficiency": efficiency,
        "confidence_intervals": confidence_intervals,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }

    out_path = Path(cfg.paths.test_report_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[test] Test metrics: {metrics}")
    print(f"[test] Efficiency: {efficiency}")
    print(f"[test] 95% bootstrap CIs: {confidence_intervals}")
    print(f"[test] Report -> {out_path}")

    generate_report(cfg, history, report)


if __name__ == "__main__":
    test()
