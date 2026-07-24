"""
Phase 10: ONNX -> TensorFlow SavedModel -> TFLite (float) -> TFLite (INT8) for
STM32 / CMSIS-NN / TFLite Micro deployment.

Requires the heavier "export" extras:
    pip install onnx2tf tensorflow --break-system-packages
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import load_config


def _representative_dataset_factory(cfg, n_samples: int):
    """
    Placeholder representative-dataset generator for INT8 calibration.

    Swap this for real training-split windows (e.g. pulled from
    data/lafnet_dataset.h5) before deploying, so the quantization ranges
    reflect your actual ECG amplitude/noise distribution instead of random
    noise.
    """
    fs = cfg.preprocessing.target_fs
    win_len = int(cfg.preprocessing.windows.length_seconds * fs)

    def gen():
        for _ in range(n_samples):
            x = np.random.randn(1, cfg.model.in_channels, win_len).astype(np.float32)
            yield [x]

    return gen


def export_tflite() -> None:
    cfg = load_config()
    onnx_path = Path(cfg.export.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"{onnx_path} not found - run `python -m src.export.onnx_export` first")

    import onnx2tf  # noqa: WPS433
    import tensorflow as tf  # noqa: WPS433

    saved_model_dir = onnx_path.parent / "lafnet_saved_model"
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        non_verbose=True,
    )

    # float TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_float = converter.convert()
    float_path = Path(cfg.export.tflite_float_path)
    float_path.parent.mkdir(parents=True, exist_ok=True)
    float_path.write_bytes(tflite_float)
    print(f"[tflite_export] Float TFLite -> {float_path}")

    # INT8 TFLite (for CMSIS-NN / STM32)
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = _representative_dataset_factory(
        cfg, cfg.export.representative_samples
    )
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    tflite_int8 = converter_int8.convert()

    int8_path = Path(cfg.export.tflite_int8_path)
    int8_path.write_bytes(tflite_int8)
    print(f"[tflite_export] INT8 TFLite -> {int8_path}")
    print(
        "[tflite_export] Convert to a C array for STM32CubeIDE with: "
        f"xxd -i {int8_path} > lafnet_model.h"
    )


if __name__ == "__main__":
    export_tflite()
