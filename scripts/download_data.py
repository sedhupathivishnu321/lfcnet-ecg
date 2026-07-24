"""
Cross-platform, ECG-only WFDB downloader for:

  - MIT-BIH Atrial Fibrillation Database   (afdb,    s3://physionet-open/afdb/1.0.0/)
  - Long Term AF Database                  (ltafdb,  s3://physionet-open/ltafdb/1.0.0/)
  - SHDB-AF (Holter AF Database)            (shdb-af, s3://physionet-open/shdb-af/1.0.1/)
  - SHDB-AF legacy                          (shdb-af, s3://physionet-open/shdb-af/1.0.0/, disabled by default)

All four sources are plain WFDB (.hea/.dat/.atr) - no MATLAB duplicates exist for
these databases, so there is nothing to filter out (unlike EPHNOGRAM's MAT/ folder).

Uses `{sys.executable} -m pip` / subprocess throughout (no wget/find/wc), so it
runs identically on Windows, macOS, and Linux, and does not depend on `aws`
being on PATH once installed into the current environment.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, enabled_datasets  # noqa: E402


def _ensure_awscli() -> None:
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    print("[download_data] awscli not found - installing into current environment...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "awscli"], check=True
    )


def _sync(s3_uri: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download_data] syncing {s3_uri} -> {dest}")
    cmd = [
        sys.executable, "-m", "awscli",
        "s3", "sync", "--no-sign-request", s3_uri, str(dest),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        # Fallback to the `aws` executable directly if `python -m awscli` isn't wired up.
        cmd = ["aws", "s3", "sync", "--no-sign-request", s3_uri, str(dest)]
        subprocess.run(cmd, check=True)


def main() -> None:
    cfg = load_config()
    _ensure_awscli()
    for ds in enabled_datasets(cfg):
        _sync(ds.s3_uri, Path(ds.local_dir))
    print("[download_data] done. ECG-only WFDB records are now under data/WFDB/.")


if __name__ == "__main__":
    main()
