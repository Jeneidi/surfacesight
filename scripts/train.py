from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import (  # noqa: E402
    BottleDataset,
    binary_f1,
    build_resnet18,
    list_samples,
    save_json,
    set_seed,
    split_samples,
)


def collect_outputs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    with torch.inference_mode():
        for images, batch_labels, batch_paths in loader:
            values = model(images.to(device)).cpu().numpy()
            logits.append(values)
            labels.append(batch_labels.numpy())
            paths.extend(batch_paths)
    return np.concatenate(logits), np.concatenate(labels), paths


def collect_features(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            features.append(model(images.to(device)).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(labels)


def benchmark_torch(model: nn.Module, sample: torch.Tensor, device: torch.device, repeats: int = 20) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(sample.to(device))
        values: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(sample.to(device))
            values.append((time.perf_counter_ns() - start) / 1_000_000)
    return float(np.median(values))


def benchmark_onnx(session: ort.InferenceSession, sample: np.ndarray, repeats: int = 20) -> float:
    input_name = session.get_inputs()[0].name
    for _ in range(5):
        session.run(None, {input_name: sample})
    values: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        session.run(None, {input_name: sample})
        values.append((time.perf_counter_ns() - start) / 1_000_000)
    return float(np.median(values))


class CalibrationReader(CalibrationDataReader):
    def __init__(self, dataset: BottleDataset, input_name: str, limit: int = 32):
        self.data = ({input_name: dataset[index][0].unsqueeze(0).numpy().astype(np.float32)} for index in range(min(limit, len(dataset))))

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self.data, None)


def main() -> int:
    set_seed(42)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    data_root = ROOT / "data" / "raw" / "mvtec_ad"
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    site_data = ROOT / "site" / "data"
    site_assets = ROOT / "site" / "assets"
    site_data.mkdir(parents=True, exist_ok=True)
    site_assets.mkdir(parents=True, exist_ok=True)

    all_samples = list_samples(data_root)
    train_samples, holdout_samples = split_samples(all_samples)
    train_set = BottleDataset(train_samples, train=True)
    holdout_set = BottleDataset(holdout_samples, train=False)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(BottleDataset(train_samples), batch_size=32, shuffle=False, num_workers=0)
    holdout_loader = DataLoader(holdout_set, batch_size=32, shuffle=False, num_workers=0)

    manifest = {
        "seed": 42,
        "train_paths": [str(Path(item["path"]).relative_to(ROOT)) for item in train_samples],
        "holdout_paths": [str(Path(item["path"]).relative_to(ROOT)) for item in holdout_samples],
    }
    save_json(artifacts / "split_manifest.json", manifest)

    baseline_model, _ = build_resnet18(weights=True)
    baseline_model.fc = nn.Identity()
    baseline_train_x, baseline_train_y = collect_features(baseline_model, train_eval_loader, device)
    baseline_holdout_x, baseline_holdout_y = collect_features(baseline_model, holdout_loader, device)
    baseline_classifier = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    baseline_classifier.fit(baseline_train_x, baseline_train_y)
    baseline_predictions = baseline_classifier.predict(baseline_holdout_x)
    baseline_f1 = binary_f1(baseline_holdout_y, baseline_predictions)

    model, _ = build_resnet18(weights=True)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(2):
        for images, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}/2 complete", flush=True)

    holdout_logits, holdout_labels, holdout_paths = collect_outputs(model, holdout_loader, device)
    holdout_probabilities = torch.softmax(torch.from_numpy(holdout_logits), dim=1).numpy()
    fine_predictions = holdout_probabilities.argmax(axis=1)
    fine_f1 = binary_f1(holdout_labels, fine_predictions)

    pytorch_path = artifacts / "resnet18_finetuned.pt"
    torch.save(model.state_dict(), pytorch_path)
    sample_tensor = holdout_set[0][0].unsqueeze(0)
    onnx_fp32_path = artifacts / "resnet18_finetuned.onnx"
    model.eval()
    torch.onnx.export(
        model,
        sample_tensor,
        onnx_fp32_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx_int8_path = artifacts / "resnet18_finetuned.int8.onnx"
    calibration_session = ort.InferenceSession(str(onnx_fp32_path), providers=["CPUExecutionProvider"])
    quantize_static(
        str(onnx_fp32_path),
        str(onnx_int8_path),
        CalibrationReader(train_eval_loader.dataset, calibration_session.get_inputs()[0].name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )

    session = ort.InferenceSession(
        str(onnx_int8_path),
        providers=["CPUExecutionProvider"],
        sess_options=ort.SessionOptions(),
    )
    sample_np = sample_tensor.numpy().astype(np.float32)
    torch_sample_logits = model(sample_tensor).detach().numpy()
    onnx_sample_logits = session.run(None, {session.get_inputs()[0].name: sample_np})[0]
    max_abs_diff = float(np.max(np.abs(torch_sample_logits - onnx_sample_logits)))
    prediction_agrees = int(torch_sample_logits.argmax(axis=1)[0]) == int(onnx_sample_logits.argmax(axis=1)[0])
    if max_abs_diff > 1.0 or not prediction_agrees:
        raise AssertionError(f"ONNX int8 fixed-sample check failed: max_abs_diff={max_abs_diff}, prediction_agrees={prediction_agrees}")

    torch_latency = benchmark_torch(model, sample_tensor, device)
    onnx_latency = benchmark_onnx(session, sample_np)
    int8_logits = np.concatenate([session.run(None, {session.get_inputs()[0].name: image.unsqueeze(0).numpy()})[0] for image, _, _ in holdout_set])
    int8_predictions = int8_logits.argmax(axis=1)
    int8_f1 = binary_f1(holdout_labels, int8_predictions)

    misclassified = []
    for path, label, prediction, probabilities in zip(holdout_paths, holdout_labels, fine_predictions, holdout_probabilities):
        if int(label) != int(prediction):
            misclassified.append({
                "path": str(Path(path).relative_to(ROOT)),
                "actual": "defective" if int(label) else "acceptable",
                "predicted": "defective" if int(prediction) else "acceptable",
                "confidence": float(probabilities[int(prediction)]),
            })

    demo_items = []
    demo_indices = list(range(min(5, len(holdout_samples))))
    for index, item in enumerate(holdout_samples):
        if int(holdout_labels[index]) != int(fine_predictions[index]) and index not in demo_indices:
            demo_indices.append(index)
    for index in demo_indices:
        item = holdout_samples[index]
        source = Path(item["path"])
        target = site_assets / f"sample-{index}-{source.parent.name}-{source.name}"
        shutil.copy2(source, target)
        demo_item = {
            "id": index,
            "image": f"assets/{target.name}",
            "actual": "defective" if int(item["label"]) else "acceptable",
            "predicted": "defective" if int(fine_predictions[index]) else "acceptable",
            "confidence": float(holdout_probabilities[index, int(fine_predictions[index])]),
        }
        demo_items.append(demo_item)
        for miss in misclassified:
            if miss["path"] == str(Path(item["path"]).relative_to(ROOT)):
                miss["image"] = demo_item["image"]

    results = {
        "dataset": {
            "name": "MVTec AD bottle category",
            "source": "https://huggingface.co/datasets/foersben/mvtec-ad",
            "official_source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
            "license": "CC BY-NC-SA 4.0",
            "images": len(all_samples),
            "train_images": len(train_samples),
            "holdout_images": len(holdout_samples),
        },
        "metrics": {
            "baseline_f1": baseline_f1,
            "finetuned_f1": fine_f1,
            "int8_f1": int8_f1,
            "int8_f1_change_vs_finetuned": int8_f1 - fine_f1,
            "pytorch_model_bytes": pytorch_path.stat().st_size,
            "onnx_int8_bytes": onnx_int8_path.stat().st_size,
            "pytorch_latency_ms_median": torch_latency,
            "onnx_int8_latency_ms_median": onnx_latency,
            "onnx_max_abs_diff": max_abs_diff,
            "onnx_fixed_sample_prediction_agrees": prediction_agrees,
            "onnx_logit_tolerance": 1.0,
        },
        "misclassified": misclassified,
        "demo": demo_items,
        "cpu_only": True,
        "pretrained_backbone": "ResNet18 ImageNet weights",
        "created_at": time.strftime("%Y-%m-%d"),
    }
    save_json(artifacts / "results.json", results)
    save_json(site_data / "results.json", results)
    print(json.dumps(results["metrics"], indent=2))
    print(f"samples: {len(all_samples)} train: {len(train_samples)} holdout: {len(holdout_samples)}")
    print(f"misclassified: {len(misclassified)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
