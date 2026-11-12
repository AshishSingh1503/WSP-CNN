"""Bitwise tensor comparison: the edit-size rule counts scalars whose stored bits change (NaN-safe)."""
from __future__ import annotations

import torch

_INT_FOR_SIZE = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def differs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Boolean mask of elements whose bit patterns differ (same shape/dtype required)."""
    int_dtype = _INT_FOR_SIZE[a.element_size()]
    return a.contiguous().view(int_dtype) != b.contiguous().view(int_dtype)
