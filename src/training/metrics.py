"""
All evaluation, complexity, energy, and statistical-validation metrics used by
validate.py, test.py, report.py, and the experiment scripts under
src/experiments/.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


# =============================================================================
# Classification metrics
# =============================================================================

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
        "mse": mean_squared_error(y_true, y_prob),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def f1_vs_threshold(y_true: np.ndarray, y_prob: np.ndarray, n_points: int = 100):
    thresholds = np.linspace(0.0, 1.0, n_points)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return thresholds, np.array(f1s), thresholds[best_idx], f1s[best_idx]


def roc_points(y_true: np.ndarray, y_prob: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return np.array([0, 1]), np.array([0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr


# =============================================================================
# Model complexity: MACs, FLOPs, parameters, size
# =============================================================================

def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """
    Multiply-ACcumulate count via a forward hook (no external profiler
    dependency, e.g. fvcore/thop, so this keeps working in constrained/offline
    environments). One MAC = one multiply + one accumulate.
    """
    macs = {"total": 0}

    def hook(module, inp, out):
        if isinstance(module, nn.Conv1d):
            out_t = out.shape[-1]
            k = module.kernel_size[0]
            in_ch_per_group = module.in_channels // module.groups
            macs["total"] += out_t * module.out_channels * in_ch_per_group * k
        elif isinstance(module, nn.Linear):
            macs["total"] += module.in_features * module.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, *input_shape))
    for h in handles:
        h.remove()
    return macs["total"]


def count_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """FLOPs ~= 2 x MACs (one multiply + one add per MAC), the standard convention."""
    return 2 * count_macs(model, input_shape)


def model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


# =============================================================================
# Inference latency: CPU and (if available) GPU
# =============================================================================

def measure_inference_time(
    model: nn.Module,
    input_shape: tuple[int, ...],
    n_runs: int = 100,
    n_warmup: int = 5,
    device: str = "cpu",
) -> float:
    """Mean single-sample inference latency in milliseconds, on the given device."""
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    dummy = torch.randn(1, *input_shape, device=dev)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000.0


def measure_cpu_gpu_latency(model: nn.Module, input_shape: tuple[int, ...], cfg) -> dict:
    """Returns {'latency_ms_cpu': ..., 'latency_ms_gpu': ... or None}."""
    c = cfg.complexity
    result = {
        "latency_ms_cpu": measure_inference_time(
            model, input_shape, n_runs=c.latency_runs, n_warmup=c.latency_warmup, device="cpu"
        )
    }
    if c.measure_gpu and torch.cuda.is_available():
        result["latency_ms_gpu"] = measure_inference_time(
            model, input_shape, n_runs=c.latency_runs, n_warmup=c.latency_warmup, device="cuda"
        )
    else:
        result["latency_ms_gpu"] = None
    return result


# =============================================================================
# Energy consumption: measured (Intel RAPL via pyRAPL) with a clearly labeled
# fallback estimate when no power meter is available.
# =============================================================================

def measure_or_estimate_energy(model: nn.Module, input_shape: tuple[int, ...], macs: int, cfg) -> dict:
    """
    Returns a dict with either:
      {"energy_uj_per_inference": <float>, "energy_source": "measured_rapl"}
    or, when RAPL isn't available (most laptops/CI/cloud VMs, all non-Intel
    CPUs, anything without root access to the RAPL counters):
      {"energy_uj_per_inference": <float>, "energy_source": "estimated_pj_per_mac"}
    The estimated figure is a rough order-of-magnitude number from a literature
    energy-per-MAC constant (config.complexity.energy.fallback_pj_per_mac_*),
    NOT a measurement - treat it as illustrative only.
    """
    e_cfg = cfg.complexity.energy
    if e_cfg.prefer_measured:
        try:
            import pyRAPL  # noqa: WPS433

            pyRAPL.setup()
            meter = pyRAPL.Measurement("lafnet_inference")
            dummy = torch.randn(1, *input_shape)
            model.eval()
            meter.begin()
            with torch.no_grad():
                for _ in range(50):
                    model(dummy)
            meter.end()
            total_uj = sum(meter.result.pkg) if meter.result.pkg else 0.0
            return {
                "energy_uj_per_inference": total_uj / 50.0,
                "energy_source": "measured_rapl",
            }
        except Exception:  # noqa: BLE001 - pyRAPL not installed / no RAPL access / not Linux+Intel
            pass

    pj_per_mac = e_cfg.fallback_pj_per_mac_cpu
    energy_pj = macs * pj_per_mac
    return {
        "energy_uj_per_inference": energy_pj / 1e6,  # pJ -> uJ
        "energy_source": "estimated_pj_per_mac",
    }


def profile_efficiency(model: nn.Module, input_shape: tuple[int, ...], cfg) -> dict:
    """One-stop efficiency profile: params, size, MACs, FLOPs, CPU/GPU latency, energy."""
    macs = count_macs(model, input_shape)
    result = {
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "size_bytes_fp32": model_size_bytes(model),
        "macs": macs,
        "flops": 2 * macs,
    }
    result.update(measure_cpu_gpu_latency(model, input_shape, cfg))
    result.update(measure_or_estimate_energy(model, input_shape, macs, cfg))
    return result


# =============================================================================
# Statistical validation: bootstrap confidence intervals + McNemar's test
# =============================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Percentile-bootstrap confidence interval for any sklearn-style metric_fn(y_true, y_pred).
    Resamples (with replacement) at the window level.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2 and metric_fn.__name__ in ("roc_auc_score", "matthews_corrcoef"):
            continue
        try:
            stats.append(metric_fn(yt, yp))
        except ValueError:
            continue
    stats = np.array(stats)
    alpha = 1.0 - confidence_level
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": float(metric_fn(y_true, y_pred)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "confidence_level": confidence_level,
        "n_iterations": int(len(stats)),
    }


def bootstrap_ci_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, cfg) -> dict:
    s = cfg.statistics
    out = {}
    for name, fn in [
        ("accuracy", accuracy_score),
        ("precision", lambda a, b: precision_score(a, b, zero_division=0)),
        ("recall", lambda a, b: recall_score(a, b, zero_division=0)),
        ("f1", lambda a, b: f1_score(a, b, zero_division=0)),
        ("mcc", matthews_corrcoef),
    ]:
        fn.__name__ = name
        out[name] = bootstrap_ci(y_true, y_pred, fn, s.bootstrap_iterations, s.confidence_level)
    return out


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, correction: bool = True) -> dict:
    """
    McNemar's test for comparing two classifiers' predictions on the SAME
    held-out samples (e.g. attention vs. no-attention, or LAFNet vs. a saved
    baseline). Tests whether the classifiers' disagreement is asymmetric.

    Contingency table:
        b = # samples A correct, B wrong
        c = # samples A wrong,   B correct
    """
    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))

    if correction:
        stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    else:
        stat = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0.0

    p_value = 1.0 - chi2.cdf(stat, df=1) if (b + c) > 0 else 1.0
    return {
        "b_a_correct_b_wrong": b,
        "c_a_wrong_b_correct": c,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def pairwise_mcnemar(y_true: np.ndarray, predictions_by_variant: dict[str, np.ndarray], correction: bool = True) -> list[dict]:
    """Runs mcnemar_test for every pair of variants in predictions_by_variant."""
    results = []
    for name_a, name_b in combinations(predictions_by_variant.keys(), 2):
        res = mcnemar_test(y_true, predictions_by_variant[name_a], predictions_by_variant[name_b], correction)
        res["variant_a"] = name_a
        res["variant_b"] = name_b
        results.append(res)
    return results
