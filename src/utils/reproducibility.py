"""Seeding, deterministic flags, and hashing helpers."""
from __future__ import annotations

import hashlib
import os
import random
from typing import Mapping

import numpy as np
import torch

from src.utils.hashing import sha256_file  # noqa: F401  (re-exported)


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_deterministic(warn_only: bool = True) -> None:
    """Request deterministic cuDNN/cuBLAS behaviour (best effort on GPU)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive numpy/python seeds from torch's worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A CPU generator seeded for DataLoader shuffling."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def sha256_tensor_dict(state: Mapping[str, torch.Tensor]) -> str:
    """Hex SHA-256 of a state dict: sorted keys, with name, dtype, shape and raw bytes of each tensor."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
