from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms


IMAGE_SIZE = 128
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def image_transform(train: bool = False) -> transforms.Compose:
    ops: list[object] = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(ops)


def list_samples(data_root: Path) -> list[dict[str, object]]:
    bottle = data_root / "bottle"
    samples: list[dict[str, object]] = []
    train_good = bottle / "train" / "good"
    for path in sorted(train_good.glob("*.png")):
        samples.append({"path": str(path), "label": 0, "class_name": "good"})
    for class_dir in sorted((bottle / "test").iterdir()):
        if not class_dir.is_dir():
            continue
        label = 0 if class_dir.name == "good" else 1
        for path in sorted(class_dir.glob("*.png")):
            samples.append({"path": str(path), "label": label, "class_name": class_dir.name})
    if not samples:
        raise FileNotFoundError(f"No MVTec bottle PNG files found under {data_root}")
    return samples


def split_samples(samples: list[dict[str, object]], seed: int = 42) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    indices = list(range(len(samples)))
    labels = [int(item["label"]) for item in samples]
    train_idx, holdout_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )
    return [samples[i] for i in train_idx], [samples[i] for i in holdout_idx]


class BottleDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, samples: Iterable[dict[str, object]], train: bool = False):
        self.samples = list(samples)
        self.transform = image_transform(train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        item = self.samples[index]
        image = Image.open(str(item["path"])).convert("RGB")
        return self.transform(image), int(item["label"]), str(item["path"])


def build_resnet18(weights: bool = True) -> tuple[nn.Module, object | None]:
    pretrained = models.ResNet18_Weights.DEFAULT if weights else None
    model = models.resnet18(weights=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if weights:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    return model, pretrained


def binary_f1(labels: Iterable[int], predictions: Iterable[int]) -> float:
    return float(f1_score(list(labels), list(predictions), zero_division=0))


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")

