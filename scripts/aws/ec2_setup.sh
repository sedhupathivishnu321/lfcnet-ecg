#!/usr/bin/env bash
# EC2 bootstrap + full-pipeline runner, sized for an 8 vCPU / 64 GB CPU-only
# instance (r5.2xlarge / r6i.2xlarge class - see config.yaml's
# aws.ec2.instance_type). Run this ON the instance after cloning the repo.
#
# Usage:
#   bash scripts/aws/ec2_setup.sh              # full pipeline: download -> verify -> hdf5 -> augment_store -> train -> test
#   bash scripts/aws/ec2_setup.sh --skip-download   # data/WFDB already populated (e.g. via sync_physionet_to_s3.py + download_data.py pointed at your own bucket)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[ec2_setup] repo root: $REPO_ROOT"
echo "[ec2_setup] detected vCPUs: $(nproc), total RAM: $(free -g | awk '/^Mem:/{print $2}')GB"
echo "[ec2_setup] this matches config.yaml's hardware.vcpus=8 / hardware.ram_gb=64 - "
echo "[ec2_setup] if it doesn't, edit that section before running to keep the memory-safety checks accurate."

SKIP_DOWNLOAD=false
for arg in "$@"; do
  if [[ "$arg" == "--skip-download" ]]; then
    SKIP_DOWNLOAD=true
  fi
done

if [[ ! -d ".venv" ]]; then
  echo "[ec2_setup] creating virtualenv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[ec2_setup] installing requirements..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if ! command -v aws >/dev/null 2>&1; then
  echo "[ec2_setup] installing awscli..."
  pip install --quiet awscli
fi

if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
  echo "[ec2_setup] downloading AFDB / LTAFDB / SHDB-AF (WFDB, ECG-only)..."
  python scripts/download_data.py
else
  echo "[ec2_setup] --skip-download passed, assuming data/WFDB is already populated"
fi

echo "[ec2_setup] Phase 3: verifying records + rhythm annotations..."
python -m src.preprocessing.verify

echo "[ec2_setup] Phase 4-6: building the HDF5 dataset (streaming writer, LZF compression)..."
python -m src.preprocessing.hdf5

echo "[ec2_setup] Phase 7: caching the subject-wise split + precomputing train-window augmentation..."
python -m src.preprocessing.augment_store

echo "[ec2_setup] Phase 9: training LAFNet..."
python -m src.training.train

echo "[ec2_setup] Phase 9: evaluating on the held-out test split..."
python -m src.training.test

echo "[ec2_setup] Done. Outputs in outputs/, checkpoints in weights/."
echo "[ec2_setup] To back everything up to S3: python scripts/aws/s3_sync.py upload weights weights && python scripts/aws/s3_sync.py upload outputs outputs"
echo "[ec2_setup] To profile any remaining hotspots: python -m scripts.profile_pipeline"
