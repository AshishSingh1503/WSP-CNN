"""Pydantic schema for gradient diagnostics (data only; no autograd code lives here)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChannelGradient(_Frozen):
    index: int
    grad_norm: float


class LayerGradient(_Frozen):
    layer: str
    normalised_grad_norm: float  # ||grad W||_2 / sqrt(numel)
    top_channels: list[ChannelGradient]


class ClassRowGradient(_Frozen):
    class_index: int
    mean_sign: float  # mean of sign(grad) over the classifier[6] row (+1: loss falls if weights shrink)
    grad_norm: float


class ClassPairGradient(_Frozen):
    class_a: int
    class_b: int
    cosine: float


class GradientDiagnostics(_Frozen):
    """Gradient-derived diagnostics of the mean CE loss on the diagnostic subset (Hybrid WSP only)."""

    n_images: int
    loss: float
    layers: list[LayerGradient]
    classifier6_rows: list[ClassRowGradient]
    classifier6_pairs: list[ClassPairGradient]
