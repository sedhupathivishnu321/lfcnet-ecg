"""
Generic S3 <-> local directory helpers for backing up/restoring the
processed dataset, checkpoints, and outputs to your own S3 bucket when
running on a plain EC2 instance (e.g. before/after stopping the instance to
save on cost, or to move results back to a laptop for inspection).

Config-driven from the `aws` section of config.yaml - nothing here hardcodes
a bucket name or prefix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_config  # noqa: E402


def _s3_uri(cfg, relative_key: str) -> str:
    bucket = cfg.aws.s3_bucket
    prefix = cfg.aws.s3_prefix.rstrip("/")
    return f"s3://{bucket}/{prefix}/{relative_key.lstrip('/')}"


def upload_dir(local_dir: str | Path, relative_key: str, cfg=None) -> str:
    """Uploads every file under local_dir to s3://<bucket>/<prefix>/<relative_key>/, returns the S3 URI."""
    cfg = cfg or load_config()
    local_dir = Path(local_dir)
    s3 = boto3.client("s3", region_name=cfg.aws.region)
    bucket = cfg.aws.s3_bucket
    prefix = cfg.aws.s3_prefix.rstrip("/")

    n = 0
    for path in local_dir.rglob("*"):
        if path.is_file():
            key = f"{prefix}/{relative_key.strip('/')}/{path.relative_to(local_dir)}"
            s3.upload_file(str(path), bucket, key)
            n += 1
    uri = _s3_uri(cfg, relative_key)
    print(f"[s3_sync] uploaded {n} files from {local_dir} -> {uri}")
    return uri


def download_dir(relative_key: str, local_dir: str | Path, cfg=None) -> None:
    """Downloads every object under s3://<bucket>/<prefix>/<relative_key>/ into local_dir."""
    cfg = cfg or load_config()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=cfg.aws.region)
    bucket = cfg.aws.s3_bucket
    prefix = f"{cfg.aws.s3_prefix.rstrip('/')}/{relative_key.strip('/')}/"

    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            n += 1
    print(f"[s3_sync] downloaded {n} files from s3://{bucket}/{prefix} -> {local_dir}")


def upload_file(local_path: str | Path, relative_key: str, cfg=None) -> str:
    cfg = cfg or load_config()
    s3 = boto3.client("s3", region_name=cfg.aws.region)
    bucket = cfg.aws.s3_bucket
    key = f"{cfg.aws.s3_prefix.rstrip('/')}/{relative_key.lstrip('/')}"
    s3.upload_file(str(local_path), bucket, key)
    uri = f"s3://{bucket}/{key}"
    print(f"[s3_sync] uploaded {local_path} -> {uri}")
    return uri


def download_file(relative_key: str, local_path: str | Path, cfg=None) -> None:
    cfg = cfg or load_config()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=cfg.aws.region)
    bucket = cfg.aws.s3_bucket
    key = f"{cfg.aws.s3_prefix.rstrip('/')}/{relative_key.lstrip('/')}"
    s3.download_file(bucket, key, str(local_path))
    print(f"[s3_sync] downloaded s3://{bucket}/{key} -> {local_path}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Upload/download a local dir to/from S3, using config.yaml's aws.* settings")
    parser.add_argument("direction", choices=["upload", "download"])
    parser.add_argument("local_dir")
    parser.add_argument("relative_key", help="e.g. 'weights', 'outputs', 'data/processed'")
    args = parser.parse_args()

    if args.direction == "upload":
        upload_dir(args.local_dir, args.relative_key)
    else:
        download_dir(args.relative_key, args.local_dir)


if __name__ == "__main__":
    _cli()
