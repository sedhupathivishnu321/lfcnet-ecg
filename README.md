# LAFNet: Lightweight Atrial-Fibrillation Network for ECG-only AF Detection

A complete, modular, research-grade PyTorch pipeline for single-lead ECG
atrial-fibrillation detection across **AFDB**, **LTAFDB**, and **SHDB-AF**
(WFDB format only), built around **LAFNet** — the single-modality,
ECG-only successor to LMFCNet, re-architected to be even lighter since
there is no second (PCG) modality to fuse.

- **Model size:** ≈9.0K parameters (well under LMFCNet's 27,970 and the
  100K budget), ~36 KB in float32, expected comfortably under 0.5 MB after
  INT8 quantization — see the honest verification note in Section 8.
- **Real AF / Non-AF labels** derived per-sample from each database's own
  WFDB rhythm annotations (`.atr` `aux_note` tokens like `(AFIB`, `(N`,
  `(AFL`), never a filename or record-level guess — see *"Critical fix"*
  below.
- **All parameters are config-driven** — see `config.yaml`, no hardcoded
  constants anywhere in `src/`.
- Live **% progress bars** during training (per-batch and overall).
- Full evaluation **report** (confusion matrix, MCC, F1 curve, ROC-AUC,
  accuracy/precision/recall, MSE, training/validation loss curves,
  efficiency) exported as both **PNG images** and a **multi-sheet Excel
  workbook**.

---

## 0. Why this is a new model, not just LMFCNet with one branch removed

LMFCNet's parameter budget was dominated by having **two** per-modality
backbones (ECG + PCG) plus a cross-attention stage between them. These
three datasets — AFDB, LTAFDB, SHDB-AF — are **ECG-only**, so:

- There is only **one** backbone (multi-scale depthwise-separable stem +
  residual SE blocks), not two.
- The cross-modal attention/fusion stage is replaced by a single
  **temporal self-attention** stage over the ECG features themselves
  (relating, e.g., an early ectopic complex to a later irregular run),
  since there's no second signal to attend across.
- The classifier head is a smaller 2-layer MLP sized for one feature
  stream instead of a fused pair.

Net effect: **~9.0K parameters**, roughly a third of LMFCNet's size, while
keeping the same multi-scale-stem + residual-SE design that made LMFCNet
effective on physiological waveforms. See `src/models/lightweight_model.py`
for the full architectural rationale in-line.

## 1. Critical fix (carried over from the same lesson as LMFCNet)

AFDB / LTAFDB / SHDB-AF do **not** ship a metadata spreadsheet like
EPHNOGRAM's `ECGPCGSpreadsheet.csv`. Instead, the real ground truth lives
**inside each record's own `.atr` rhythm-annotation stream**: `aux_note`
entries such as `(AFIB`, `(N`, `(AFL`, `(J` mark the start of a rhythm
segment that continues until the next rhythm annotation.

`src/preprocessing/annotations.py` builds a genuine **per-sample** AF /
Non-AF timeline directly from these real annotations (never a filename or
whole-record guess), and `src/preprocessing/hdf5.py` labels every window
by the fraction of AF-labeled samples it contains
(`preprocessing.windows.af_positive_fraction` in `config.yaml`). A single
record can — and usually does — contain both classes. `build_hdf5_dataset`
logs the resulting class distribution and raises a loud error if fewer
than 2 classes end up in the built dataset, the same guardrail LMFCNet
used against a degenerate single-class dataset.

AFDB records `04043` and `04048` are excluded by default in `config.yaml`
(`data.datasets[].exclude_records`) — PhysioNet ships these without rhythm
annotations, so no real label can be derived for them.

## 2. Project Structure

```
lafnet-ecg-af/
├── scripts/
│   ├── download_data.py           # cross-platform WFDB-only PhysioNet S3 download (4 sources)
│   └── aws/
│       ├── sync_physionet_to_s3.py # server-side (bucket-to-bucket) PhysioNet -> your own S3 bucket
│       ├── s3_sync.py              # generic upload/download of data/checkpoints/outputs to/from S3
│       └── ec2_setup.sh            # bootstrap + full-pipeline runner for the 8 vCPU / 32GB target instance
├── data/
│   └── WFDB/                     # AFDB / LTAFDB / SHDB-AF *.hea/*.dat/*.atr land here (gitignored)
├── src/
│   ├── config.py                 # typed YAML config loader
│   ├── preprocessing/
│   │   ├── verify.py              # WFDB record + rhythm-annotation validation, report
│   │   ├── annotations.py          # REAL per-sample AF/Non-AF labels from rhythm annotations
│   │   ├── filter.py               # Bessel/Butterworth/Cheby1/Elliptic/FIR + notch
│   │   ├── quality.py              # NaN/Inf/flat/clip/noise + Signal Quality Index
│   │   ├── normalization.py        # Z-score / Min-Max / Robust / Decimal
│   │   ├── windows.py              # 10s windows, 50% overlap, AF-fraction labeling
│   │   └── hdf5.py                 # full pipeline -> compressed HDF5 dataset
│   ├── dataset/
│   │   ├── loader.py                # PyTorch Dataset + subject-wise split
│   │   ├── augment.py               # noise/shift/scale/stretch/crop/mixup (single-lead ECG)
│   │   └── sampler.py               # class-balanced sampler + feature-space SMOTE
│   ├── models/
│   │   ├── layers.py                 # depthwise-separable conv, multi-scale conv, residual, SE, temporal self-attn
│   │   ├── backbone.py                # single ECG-branch multi-scale CNN backbone
│   │   └── lightweight_model.py       # LAFNet assembly (no fusion.py / cross attention.py - single modality)
│   ├── training/
│   │   ├── losses.py                  # CrossEntropy / label smoothing / focal
│   │   ├── metrics.py                 # accuracy/P/R/F1/MCC/MSE/ROC-AUC/F1-curve/MACs/FLOPs/CPU+GPU latency/energy/bootstrap-CI/McNemar
│   │   ├── train.py                   # AdamW + cosine schedule + early stopping + live % progress bars (train_model() is reused by every experiment script below)
│   │   ├── validate.py                # validation loop
│   │   ├── test.py                    # held-out test + full report (JSON + plots + Excel), now with bootstrap CIs
│   │   └── report.py                  # PNG plots + multi-sheet Excel workbook generator
│   ├── experiments/
│   │   ├── utils.py                    # shared held-out evaluator + CSV/XLSX result writers
│   │   ├── ablation.py                 # filter type / window length / attention / SE / imbalance-strategy ablations
│   │   ├── cross_database.py           # train on AFDB, zero-shot test on LTAFDB + SHDB-AF
│   │   └── cross_validation.py         # 5-fold subject-wise CV + bootstrap CI + McNemar significance test
│   └── export/
│       ├── onnx_export.py             # PyTorch -> ONNX
│       └── tflite_export.py           # ONNX -> TF SavedModel -> TFLite -> INT8
├── outputs/
│   ├── plots/                   # confusion_matrix.png, f1_curve.png, training_curves.png, ...
│   └── lafnet_report.xlsx       # Summary / Training History / Test Metrics / Efficiency sheets
├── weights/                     # checkpoints
├── logs/
├── config.yaml
└── requirements.txt
```

## 3. Setup

```bash
git clone <this-repo-url>
cd lafnet-ecg-af
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (cmd/PowerShell):
.venv\Scripts\activate

pip install -r requirements.txt
```

Download all four ECG-only WFDB sources directly from PhysioNet's public,
no-sign-required AWS S3 buckets (config-driven — see `data.datasets` in
`config.yaml`; `shdb-af` legacy `1.0.0` is listed but disabled by default
since `1.0.1` supersedes it):

```bash
python scripts/download_data.py
```

This runs the equivalent of, for each enabled source:

```bash
aws s3 sync --no-sign-request s3://physionet-open/afdb/1.0.0/      data/WFDB/afdb/
aws s3 sync --no-sign-request s3://physionet-open/ltafdb/1.0.0/    data/WFDB/ltafdb/
aws s3 sync --no-sign-request s3://physionet-open/shdb-af/1.0.1/   data/WFDB/shdb-af-1.0.1/
# shdb-af/1.0.0 available too — set data.datasets[].enabled: true in config.yaml to include it
```

installing `awscli` into the current environment if needed, using plain
Python + `{sys.executable} -m ...` throughout, so it works identically on
Windows, macOS, and Linux.

All records are read exclusively via:

```python
import wfdb
record = wfdb.rdrecord("data/WFDB/afdb/04015")
ann = wfdb.rdann("data/WFDB/afdb/04015", "atr")
```

## 4. Training Guide (Phases 2–9)

Run each phase from the project root (so `src` is importable) or via
`python -m`:

```bash
# Phase 3 — verify every WFDB record + rhythm annotation, produce outputs/dataset_report.csv
python -m src.preprocessing.verify

# Phase 4–6 — filter, normalize, window, build compressed HDF5 dataset
# (uses each record's own .atr rhythm annotations for real AF/Non-AF labels)
python -m src.preprocessing.hdf5

# Phase 9 — train LAFNet (AdamW, cosine LR, early stopping)
# Shows a live progress bar with % complete for every epoch's batches
# AND for the overall run, e.g.:
#   Epoch 003/100: 45%|====>     | loss=0.4213 pct=45%
#   Overall training progress: 3%|>   | overall_pct=3.0% val_f1=0.81 val_mcc=0.64
python -m src.training.train

# Phase 9 — evaluate on the held-out test split; writes:
#   outputs/test_report.json           (raw numbers)
#   outputs/plots/*.png                (confusion matrix, F1 curve, training
#                                        curves, metrics bar, efficiency bar)
#   outputs/lafnet_report.xlsx         (Summary / Training History /
#                                        Test Metrics / Efficiency sheets)
python -m src.training.test
```

All hyperparameters (which datasets are enabled, filter type/order, window
length/overlap, AF-fraction labeling threshold, split ratios, optimizer,
loss, epochs, etc.) live in `config.yaml` — edit that file rather than the
code to change behavior.

## 5. Evaluation Report (images + Excel)

`python -m src.training.test` calls `src/training/report.py`, which produces
**every requested metric as both a chart and a spreadsheet row**:

| Metric | PNG | Excel sheet |
|---|---|---|
| Confusion matrix | `outputs/plots/confusion_matrix.png` | `Test Metrics` (full matrix table) |
| MCC | `outputs/plots/metrics_bar.png` | `Test Metrics`, `Summary` |
| F1 curve (vs. threshold) | `outputs/plots/f1_curve.png` | — (curve is image-only; peak value is in `Test Metrics`) |
| Accuracy / Precision / Recall / F1 | `outputs/plots/metrics_bar.png` | `Test Metrics`, `Summary` |
| MSE | — | `Test Metrics`, `Summary` |
| Training / validation loss curves | `outputs/plots/training_curves.png` | `Training History` (raw per-epoch values + an embedded native Excel line chart) |
| Efficiency (latency, FLOPs, params, size) | `outputs/plots/efficiency.png` | `Efficiency`, `Summary` |

The **`Summary`** sheet's cells are live Excel **formulas** (`=MAX(...)`,
`=INDEX/MATCH(...)`, cross-sheet references) pulling from `Training
History`/`Test Metrics`/`Efficiency` — never hardcoded numbers. If you
edit `report.py` and add new formulas, re-run
`python /mnt/skills/public/xlsx/scripts/recalc.py outputs/lafnet_report.xlsx`
(or open/save once in Excel/LibreOffice) so the cached formula values are
populated.

## 6. Model: LAFNet

```
ECG ─▶ Multi-scale Depthwise CNN (k=3,7,15) ─▶ Residual×4 (+SE) ─▶ Temporal Self-Attention ─▶ GAP ─▶ Classifier
```

- A **multi-scale depthwise-separable stem** — parallel kernel sizes
  (default `[3, 7, 15]`) fused by a 1×1 conv, capturing fine (QRS-complex),
  medium (P/T-wave), and coarse (RR-interval / rhythm-level) temporal
  structure in one layer — followed by 4 residual depthwise-separable
  blocks → Squeeze-Excitation channel attention → progressive 2× average-
  pool downsample per block.
- **Temporal self-attention:** single-head scaled dot-product attention
  over the pooled ECG feature tokens (`attention_dim`, default 32) —
  the single-modality analogue of LMFCNet's ECG↔PCG cross-attention.
- **Classifier:** 2-layer MLP head.

## 7. Research Validation Add-ons

Everything in this section is config-driven from the new sections at the
bottom of `config.yaml` (`ablation`, `cross_database`, `cross_validation`,
`statistics`, `complexity`), and reuses the exact same `train_model()`
training loop as `src/training/train.py` so results are directly comparable.

### 7.1 Ablation study

```bash
python -m src.experiments.ablation
```

One-factor-at-a-time by default (`ablation.full_grid: false`), sweeping:

| Axis | Values | What it isolates |
|---|---|---|
| `preprocessing.filter.type` | butterworth / bessel / cheby1 / fir | choice of bandpass filter family |
| `preprocessing.windows.length_seconds` | 5 / 10 / 15 | window length vs. AF-episode capture |
| `model.use_attention` | true / false | value of the temporal self-attention stage |
| `model.use_se` | true / false | value of Squeeze-Excitation channel attention |
| `sampler.imbalance_strategy` | smote / weighted_sampler | oversampling vs. reweighting for class imbalance |

`filter_type` and `window_seconds` variants change the input signal itself,
so each is cached as its own HDF5 file under `data/ablation_cache/` and only
rebuilt once. Set `ablation.full_grid: true` to instead run the full
Cartesian product of all axes (much more expensive — every combination is
its own training run). Results (accuracy/precision/recall/F1/MCC/ROC-AUC,
params, MACs, FLOPs, CPU latency) -> `outputs/ablation_results.csv` and
`outputs/ablation_report.xlsx`.

### 7.2 Cross-database generalization

```bash
python -m src.experiments.cross_database
```

Trains on `cross_database.train_on` (default: AFDB subjects only) and
evaluates, with **no fine-tuning**, on each database in
`cross_database.test_on` (default: LTAFDB, SHDB-AF), alongside its own
in-domain held-out validation subjects for comparison. Also runs a
McNemar's-test comparison of each external database's predictions against
the in-domain predictions, as an illustrative generalization-gap
significance check for the same trained model. Results ->
`outputs/cross_database_results.json`.

### 7.3 5-fold subject-wise cross-validation

```bash
python -m src.experiments.cross_validation
```

Uses `sklearn.model_selection.GroupKFold` with subject ID as the group, so
no subject's windows ever cross a fold boundary. Reports per-fold metrics,
mean ± std across the 5 folds, a 95% percentile-bootstrap confidence
interval on the pooled out-of-fold predictions
(`cross_validation.bootstrap_iterations`), and a McNemar's test of the
pooled predictions against a trivial majority-class baseline. Results ->
`outputs/cross_validation_results.json` and
`outputs/cross_validation_report.xlsx`.

### 7.4 Model complexity & energy profiling

`src/training/metrics.py::profile_efficiency()` (used automatically by
`src/training/test.py` and every experiment script above) reports:

- **Parameters** and **model size** (fp32 bytes)
- **MACs** (multiply-accumulate count) and **FLOPs** (= 2 × MACs), via a
  forward-hook-based counter with no external profiler dependency
- **CPU inference latency** (always) and **GPU inference latency** (if
  `complexity.measure_gpu: true` and a CUDA device is available)
- **Energy per inference**: measured via Intel RAPL counters through the
  optional `pyRAPL` package when running on supported Linux/Intel hardware;
  otherwise falls back to a clearly-labeled *estimate* from a literature
  energy-per-MAC constant (`complexity.energy.fallback_pj_per_mac_cpu`) —
  every result dict includes an `energy_source` field (`"measured_rapl"` or
  `"estimated_pj_per_mac"`) so measured and estimated figures are never
  silently conflated.

### 7.5 Statistical validation primitives

`src/training/metrics.py` also exposes, for reuse in your own analysis:

- `bootstrap_ci(y_true, y_pred, metric_fn, ...)` — percentile-bootstrap CI
  for any sklearn-style metric.
- `mcnemar_test(y_true, y_pred_a, y_pred_b, correction=True)` — continuity-
  corrected McNemar's test for comparing two classifiers on the *same*
  held-out samples (e.g. attention vs. no-attention).
- `pairwise_mcnemar(y_true, predictions_by_variant)` — runs McNemar's test
  over every pair in a dict of `{variant_name: y_pred}`, useful for
  comparing all ablation arms against each other at once.

## 8. AWS Deployment (8 vCPU / 32 GB, CPU-only)

This repo assumes you're running on a modest single AWS instance — 8 vCPUs,
32 GB RAM, no GPU (e.g. `m5.2xlarge` / `c5.2xlarge` / `m6i.2xlarge`) — not a
GPU or multi-node setup. `config.yaml`'s `hardware:` section records that
profile, and three places in the pipeline actively use it to stay
memory-safe rather than assuming unlimited RAM:

- **`src/preprocessing/hdf5.py`** writes windows to a *resizable* HDF5
  dataset incrementally, one record at a time, instead of accumulating every
  window from every record (AFDB + LTAFDB + SHDB-AF together, at 50%
  overlap, can be tens of GB) into a single Python list before stacking —
  the old pattern peaks at roughly 2x the final dataset size in RAM and can
  exceed 32 GB, especially with LTAFDB's day-plus-length recordings. Peak
  RAM here never holds more than one record's windows at a time.
- **`src/dataset/imbalance.py`**'s SMOTE branch estimates the in-memory size
  of materializing the whole training split as one dense array (which SMOTE
  requires) *before* running it, and falls back to `weighted_sampler` with a
  clear warning instead of risking an OOM kill if it would exceed
  `hardware.max_smote_ram_gb`.
- **`src/training/train.py`**'s `resolve_num_workers()` caps DataLoader
  worker processes to `min(training.num_workers, hardware.vcpus - 1,
  os.cpu_count())` — leaving one core free for the main process/OS — instead
  of trusting `training.num_workers` blindly.

`training.batch_size` (128) and `training.num_workers` (6, further capped at
runtime) are already tuned for this profile; edit `hardware:` first if you
move to a different instance size so these checks stay accurate.

### 8.1 Getting data onto the instance

```bash
# Fastest path when your compute is already on AWS: bucket-to-bucket sync,
# no local download needed on your side, full intra-region bandwidth.
python scripts/aws/sync_physionet_to_s3.py
# then point config.yaml's data.datasets[].s3_uri at your own bucket, or just
# run the normal downloader directly on the instance:
python scripts/download_data.py
```

### 8.2 Running the full pipeline on the instance

```bash
bash scripts/aws/ec2_setup.sh                  # download -> verify -> hdf5 -> train -> test
bash scripts/aws/ec2_setup.sh --skip-download  # if data/WFDB is already populated
```

### 8.3 Backing up checkpoints/outputs to S3

```bash
python scripts/aws/s3_sync.py upload weights weights
python scripts/aws/s3_sync.py upload outputs outputs
python scripts/aws/s3_sync.py download weights weights   # restore on a fresh instance
```

All of this is config-driven from `config.yaml`'s `aws:` section
(`s3_bucket`, `s3_prefix`, `region`, `ec2.instance_type`, etc.) — fill in the
bucket name, key pair, and security group before use.

## 9. Export & Embedded Deployment (Phase 10)

```
PyTorch → ONNX → TensorFlow SavedModel → TFLite → INT8 → CMSIS-NN → STM32
```

```bash
python -m src.export.onnx_export

pip install onnx2tf tensorflow --break-system-packages
python -m src.export.tflite_export
```

`tflite_export.py`'s INT8 path uses a representative-dataset generator for
calibration; swap the random placeholder in `_representative_dataset_factory`
for real training-split windows before deploying.

### STM32 deployment guide (outline)

1. `xxd -i outputs/lafnet_int8.tflite > lafnet_model.h` (or STM32CubeAI's
   converter, which also reports RAM/Flash estimates directly).
2. Import into STM32CubeIDE / STM32CubeAI as a **TFLite Micro** or
   **CMSIS-NN** network.
3. Allocate a static tensor arena per STM32CubeAI's memory report —
   single-branch LAFNet needs meaningfully less RAM/Flash than LMFCNet's
   dual-branch design.
4. On-device inference loop: read a 10s @ target_fs single-lead ECG buffer
   → quantize per the exported model's input scale/zero-point → invoke →
   dequantize logits → argmax.
5. Benchmark actual latency/RAM/Flash on target hardware against
   `outputs/test_report.json`'s efficiency numbers.

## 10. Verification Performed / Honest Limits of This Environment

This sandbox has **no live network access** (so the real AFDB/LTAFDB/SHDB-AF
WFDB files could not be downloaded here) and **no PyTorch, h5py, or tqdm
installed**, so unlike the original LMFCNet delivery, the pipeline in this
environment could **not** be run end-to-end, not even against synthetic
stand-in records. What *was* done:

- ✅ Every `.py` file in this repository was syntax-checked
  (`python -m py_compile`) and imports/module layout were cross-checked by
  hand — all pass cleanly.
- ✅ `LAFNet`'s parameter count was hand-derived layer-by-layer from
  `config.yaml`'s defaults (stem, projection, 4 residual+SE blocks,
  temporal self-attention, classifier): **≈9,008 parameters**. This is an
  arithmetic estimate, not a measured `count_parameters(model)` run — treat
  it as directionally correct rather than exact until you run
  `python -m src.models.lightweight_model` yourself in an environment with
  PyTorch installed.
- ✅ The label-derivation logic in `annotations.py` was reasoned through
  against the documented AFDB/LTAFDB/SHDB-AF annotation conventions
  (rhythm changes marked by `(`-prefixed `aux_note` tokens), but was **not**
  run against real `.atr` files here, since none could be downloaded.

**Before you trust this for anything beyond a starting point:** run
`python -m src.preprocessing.verify` on real downloaded data first, inspect
`outputs/dataset_report.csv`, and spot-check a handful of records' `aux_note`
values against `config.yaml`'s `labels.af_aux_tokens` /
`labels.non_af_aux_tokens` lists — different PhysioNet databases are not
always perfectly consistent in their rhythm-label vocabulary, and this is
exactly the kind of silent-failure risk the original LMFCNet "Critical fix"
was about.

The same caveat applies, doubly, to Section 7's ablation / cross-database /
cross-validation / complexity / statistical-validation scripts: their logic
was written and syntax-checked (`python -m py_compile`) and the pure
statistical functions (`mcnemar_test`, `bootstrap_ci`) were unit-tested in
this sandbox against synthetic prediction arrays (with `torch` stubbed out,
since it isn't installed here) to confirm the arithmetic is sound - but none
of them were run end-to-end against real ECG data or a real trained model,
since neither is available in this environment. Treat `src/experiments/` as
a complete, ready-to-run implementation, not as pre-validated results.

The Section 8 AWS/memory-safety work has the same limit: `h5py` isn't
installed in this sandbox either, so `src/preprocessing/hdf5.py`'s
incremental resizable-dataset writer was validated by re-implementing its
exact resize/append logic against a synthetic in-memory h5py session
description and reasoning through it line-by-line (standard, well-
established h5py usage), not by actually running it here. Similarly,
`resolve_num_workers()`'s vCPU-capping logic and the SMOTE memory-budget
estimate in `src/dataset/imbalance.py` are straightforward arithmetic and
were traced by hand, not executed against a real dataset. Run
`python -m src.preprocessing.hdf5` yourself on the actual instance and watch
memory usage (e.g. `htop` in another terminal) before trusting it on a much
larger combined dataset than was reasoned about here.
