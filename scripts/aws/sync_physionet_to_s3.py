"""
Server-side, bucket-to-bucket sync of AFDB/LTAFDB/SHDB-AF from PhysioNet's
public, no-sign-required S3 bucket directly into your own S3 bucket, WITHOUT
downloading to local disk first.

This is the fastest and cheapest path when your compute runs on AWS (EC2 or
This is the fastest and cheapest path when your compute runs on AWS (EC2):
S3-to-S3 `aws s3 sync` transfers happen entirely inside AWS's network, so a
training run on the instance (via scripts/download_data.py, pointed at your
own bucket) pulls at full intra-region bandwidth instead of re-fetching from
PhysioNet's bucket for every job.

Usage:
    python scripts/aws/sync_physionet_to_s3.py
    # or restrict to specific datasets:
    python scripts/aws/sync_physionet_to_s3.py --datasets afdb ltafdb
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_config, enabled_datasets  # noqa: E402


def _dest_uri(cfg, dataset_name: str) -> str:
    bucket = cfg.aws.s3_bucket
    prefix = cfg.aws.s3_prefix
    raw = cfg.aws.s3_paths.raw_data
    return f"s3://{bucket}/{prefix}/{raw}/{dataset_name}/"


def sync_one(s3_source: str, s3_dest: str, region: str) -> None:
    print(f"[sync_physionet_to_s3] {s3_source} -> {s3_dest}")
    cmd = [
        "aws", "s3", "sync", "--no-sign-request", "--region", region,
        s3_source, s3_dest,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None,
                         help="Restrict to these dataset names (default: all enabled in config.yaml)")
    args = parser.parse_args()

    cfg = load_config()
    datasets = enabled_datasets(cfg)
    if args.datasets:
        datasets = [d for d in datasets if d.name in args.datasets]

    if not datasets:
        print("[sync_physionet_to_s3] No matching enabled datasets in config.yaml - nothing to do.")
        return

    for ds in datasets:
        sync_one(ds.s3_uri, _dest_uri(cfg, ds.name), cfg.aws.region)

    print(f"[sync_physionet_to_s3] Done. Data now under "
          f"s3://{cfg.aws.s3_bucket}/{cfg.aws.s3_prefix}/{cfg.aws.s3_paths.raw_data}/")


if __name__ == "__main__":
    main()
