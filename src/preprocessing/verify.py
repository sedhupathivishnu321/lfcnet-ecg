"""
Validates every WFDB record across all enabled datasets:
  - record loads cleanly with wfdb.rdrecord
  - the configured ECG channel exists
  - a matching .atr rhythm-annotation file exists (required for real labels)
  - record is not in the per-dataset exclude list

Writes outputs/dataset_report.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import wfdb

from src.config import load_config, enabled_datasets


def _list_records(local_dir: Path):
    return sorted({p.stem for p in local_dir.glob("*.hea")})


def verify_all(cfg) -> list[dict]:
    rows = []
    for ds in enabled_datasets(cfg):
        local_dir = Path(ds.local_dir)
        if not local_dir.exists():
            print(f"[verify] WARNING: {local_dir} does not exist - run scripts/download_data.py first")
            continue
        record_ids = _list_records(local_dir)
        for rid in record_ids:
            row = {"dataset": ds.name, "record": rid, "status": "ok", "reason": ""}
            if rid in getattr(ds, "exclude_records", []):
                row["status"] = "excluded"
                row["reason"] = "in exclude_records list (known missing/bad rhythm annotations)"
                rows.append(row)
                continue
            try:
                rec = wfdb.rdrecord(str(local_dir / rid))
                if ds.ecg_channel >= rec.p_signal.shape[1]:
                    row["status"] = "bad"
                    row["reason"] = f"ecg_channel {ds.ecg_channel} out of range (n_sig={rec.p_signal.shape[1]})"
                    rows.append(row)
                    continue
                ann_path = local_dir / f"{rid}.atr"
                if not ann_path.exists():
                    row["status"] = "bad"
                    row["reason"] = "missing .atr rhythm annotation file"
                    rows.append(row)
                    continue
                _ = wfdb.rdann(str(local_dir / rid), "atr")
                row["reason"] = f"n_sig={rec.p_signal.shape[1]}, fs={rec.fs}, sig_len={rec.sig_len}"
            except Exception as e:  # noqa: BLE001
                row["status"] = "bad"
                row["reason"] = str(e)
            rows.append(row)
    return rows


def main() -> None:
    cfg = load_config()
    rows = verify_all(cfg)
    out_path = Path(cfg.paths.outputs_dir) / "dataset_report.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "record", "status", "reason"])
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    bad = sum(1 for r in rows if r["status"] == "bad")
    excluded = sum(1 for r in rows if r["status"] == "excluded")
    print(f"[verify] {ok} ok, {bad} bad, {excluded} excluded. Report -> {out_path}")


if __name__ == "__main__":
    main()
