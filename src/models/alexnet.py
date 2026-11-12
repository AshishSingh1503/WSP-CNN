"""AlexNet for CIFAR-10: pretrained backbone, new deterministic 4096->10 head."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import AlexNet_Weights, alexnet

from src.utils.config import Config, resolve_path
from src.utils.hashing import sha256_file

PENULTIMATE_INDEX = 6  # classifier[:6] ends with the second ReLU -> 4096-d features
WEIGHTS_FILENAME = "alexnet-owt-7be5be79.pth"  # IMAGENET1K_V1


def set_weights_dir(cfg: Config) -> None:
    """Keep downloaded torchvision weights inside the project's data/ folder."""
    torch.hub.set_dir(str(resolve_path(cfg.paths.data_dir) / "torch_hub"))


def pretrained_weights_sha256() -> str | None:
    """SHA-256 of the cached ImageNet weight file, or None if it is not cached yet."""
    path = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_FILENAME
    return sha256_file(path) if path.is_file() else None


def init_head(head: nn.Linear, seed: int) -> None:
    """Deterministic init equal to nn.Linear's default (U(-1/sqrt(in), 1/sqrt(in))), seeded explicitly."""
    gen = torch.Generator().manual_seed(seed)
    bound = 1.0 / math.sqrt(head.in_features)
    with torch.no_grad():
        head.weight.uniform_(-bound, bound, generator=gen)
        head.bias.uniform_(-bound, bound, generator=gen)


def build_model(num_classes: int = 10, seed: int = 0, pretrained: bool = True,
                weights_name: str = "IMAGENET1K_V1") -> nn.Module:
    """AlexNet with ``classifier[6]`` replaced by a seeded ``Linear(4096, num_classes)``.

    ``pretrained=False`` skips the weight download (used when a state dict is loaded afterwards).
    """
    weights = getattr(AlexNet_Weights, weights_name) if pretrained else None
    model = alexnet(weights=weights)
    model.classifier[6] = nn.Linear(4096, num_classes)
    init_head(model.classifier[6], seed)
    return model


def feature_extractor(model: nn.Module) -> nn.Sequential:
    """Layers up to and including ``classifier[5]`` (4096-d, post-ReLU); shares parameters with ``model``."""
    return nn.Sequential(model.features, model.avgpool, nn.Flatten(1),
                         model.classifier[:PENULTIMATE_INDEX])


def describe_model(model: nn.Module) -> list[dict[str, Any]]:
    """One record per parameter: name, owning module type, shape, numel, trainable."""
    modules = dict(model.named_modules())
    records = []
    for name, param in model.named_parameters():
        owner = name.rsplit(".", 1)[0]
        records.append({"name": name, "type": type(modules[owner]).__name__,
                        "shape": list(param.shape), "numel": param.numel(),
                        "trainable": param.requires_grad})
    return records


def save_model_card(model: nn.Module, path: str | Path) -> list[dict[str, Any]]:
    """Write the parameter table (plus totals) to JSON and return the table."""
    table = describe_model(model)
    card = {"total_params": sum(r["numel"] for r in table), "parameters": table}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(card, indent=2), encoding="utf-8")
    return table
