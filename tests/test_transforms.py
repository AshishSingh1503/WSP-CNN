"""Transforms: eval is deterministic; shapes and normalization are right."""
import numpy as np
import torch
from PIL import Image

from src.data.transforms import build_eval_transform, build_train_transform, imagenet_stats
from src.utils.config import load_config

CFG = load_config().data


def _image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))


def test_eval_deterministic() -> None:
    tf, img = build_eval_transform(CFG), _image()
    assert torch.equal(tf(img), tf(img))
    assert torch.equal(build_eval_transform(CFG)(img), tf(img))


def test_eval_shape_dtype() -> None:
    out = build_eval_transform(CFG)(_image())
    assert tuple(out.shape) == (3, 224, 224) and out.dtype == torch.float32


def test_train_shape_and_randomness() -> None:
    tf, img = build_train_transform(CFG), _image()
    torch.manual_seed(0)
    outs = [tf(img) for _ in range(8)]
    assert all(tuple(o.shape) == (3, 224, 224) for o in outs)
    assert any(not torch.equal(outs[0], o) for o in outs[1:])


def test_normalization_stats() -> None:
    mean, std = imagenet_stats(CFG)
    assert np.allclose(mean, (0.485, 0.456, 0.406)) and np.allclose(std, (0.229, 0.224, 0.225))
    white = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
    out = build_eval_transform(CFG)(white)
    assert torch.allclose(out[0], torch.full((224, 224), (1 - mean[0]) / std[0]), atol=1e-4)
