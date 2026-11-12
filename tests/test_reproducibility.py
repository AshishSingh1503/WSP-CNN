"""Seeding and hashing helpers."""
import random

import numpy as np
import torch

from src.utils.reproducibility import make_generator, set_seed, sha256_file, sha256_tensor_dict


def test_set_seed_repeatable() -> None:
    set_seed(5)
    a = (random.random(), np.random.rand(), torch.rand(3))
    set_seed(5)
    b = (random.random(), np.random.rand(), torch.rand(3))
    assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


def test_generator_repeatable() -> None:
    assert torch.equal(torch.randperm(10, generator=make_generator(3)),
                       torch.randperm(10, generator=make_generator(3)))


def test_tensor_dict_hash() -> None:
    sd = {"a": torch.arange(6.0).reshape(2, 3), "b": torch.ones(2, dtype=torch.float16)}
    assert sha256_tensor_dict(sd) == sha256_tensor_dict(dict(reversed(list(sd.items()))))
    changed = {**sd, "a": sd["a"] + 1e-3}
    assert sha256_tensor_dict(changed) != sha256_tensor_dict(sd)
    assert sha256_tensor_dict({"s": torch.tensor(1.0)})  # 0-dim tensors hash fine


def test_sha256_file(tmp_path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
