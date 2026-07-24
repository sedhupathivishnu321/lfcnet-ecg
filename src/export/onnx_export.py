"""Phase 10: PyTorch -> ONNX."""
from __future__ import annotations

from pathlib import Path

import torch

from src.config import load_config
from src.models.lightweight_model import LAFNet


def export_onnx() -> None:
    cfg = load_config()
    model = LAFNet(cfg)
    weights_path = Path(cfg.paths.weights_dir) / "lafnet_best.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print(f"[onnx_export] WARNING: {weights_path} not found - exporting randomly initialized weights")
    model.eval()

    fs = cfg.preprocessing.target_fs
    win_len = int(cfg.preprocessing.windows.length_seconds * fs)
    dummy = torch.randn(1, cfg.model.in_channels, win_len)

    out_path = Path(cfg.export.onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["ecg"],
        output_names=["logits"],
        dynamic_axes={"ecg": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=cfg.export.opset,
    )
    print(f"[onnx_export] Exported -> {out_path}")

    # quick validation
    import onnx  # noqa: WPS433 (optional dependency, only needed for export)

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    print("[onnx_export] ONNX graph validated OK")


if __name__ == "__main__":
    export_onnx()
