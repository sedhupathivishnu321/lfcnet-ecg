"""Shared helpers for src/experiments/*: result writers and a generic held-out evaluator."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from openpyxl import Workbook
from openpyxl.styles import Font
from torch.utils.data import DataLoader

from src.training.metrics import classification_metrics

ARIAL = Font(name="Arial")
ARIAL_BOLD = Font(name="Arial", bold=True)


@torch.no_grad()
def evaluate_dataset(model, dataset, cfg, device) -> dict:
    """Runs the model over an arbitrary held-out Dataset and returns
    classification_metrics() plus raw y_true/y_pred/y_prob (needed for
    McNemar's test / bootstrap CI downstream)."""
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        y_true.append(y.numpy())
        y_pred.append(preds.cpu().numpy())
        y_prob.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    metrics = classification_metrics(y_true, y_pred, y_prob)
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def write_rows_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_xlsx(rows: list[dict], out_path: Path, sheet_name: str = "Results") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    if rows:
        headers = list(rows[0].keys())
        ws.append(headers)
        for row in rows:
            ws.append([row.get(h) for h in headers])
        for row_cells in ws.iter_rows():
            for cell in row_cells:
                cell.font = ARIAL_BOLD if cell.row == 1 else ARIAL
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = 22
    wb.save(out_path)
