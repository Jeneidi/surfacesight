from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.pipeline import binary_f1, list_samples, split_samples


ROOT = Path(__file__).resolve().parents[1]


def test_metric_math() -> None:
    assert binary_f1([0, 1, 1, 0], [0, 1, 0, 0]) == 2 / 3


def test_split_has_no_filename_overlap() -> None:
    samples = list_samples(ROOT / "data" / "raw" / "mvtec_ad")
    train, holdout = split_samples(samples)
    train_paths = {item["path"] for item in train}
    holdout_paths = {item["path"] for item in holdout}
    assert train_paths.isdisjoint(holdout_paths)
    assert len(train) + len(holdout) == len(samples)


def test_recorded_manifest_has_no_leakage() -> None:
    manifest = json.loads((ROOT / "artifacts" / "split_manifest.json").read_text())
    assert set(manifest["train_paths"]).isdisjoint(manifest["holdout_paths"])


def test_int8_onnx_agrees_with_pytorch_on_fixed_sample() -> None:
    results = json.loads((ROOT / "artifacts" / "results.json").read_text())
    assert results["metrics"]["onnx_max_abs_diff"] <= results["metrics"]["onnx_logit_tolerance"]
    assert results["metrics"]["onnx_fixed_sample_prediction_agrees"] is True
    session = ort.InferenceSession(
        str(ROOT / "artifacts" / "resnet18_finetuned.int8.onnx"),
        providers=["CPUExecutionProvider"],
    )
    assert session.get_inputs()[0].shape[1:] == [3, 128, 128]
    assert np.isfinite(torch.tensor(results["metrics"]["onnx_max_abs_diff"]).item())
