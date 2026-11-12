"""Strict Pydantic schema for LLM edit proposals. Structure only: numeric limits live in config and are
enforced by ``edit_validator`` / ``edit_plan`` so there is a single source of truth."""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]  # no numeric strings, no NaN/inf
Idx = Annotated[int, Field(strict=True)]  # no floats, strings or bools


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AddDelta(_Strict):
    """Add ``deltas[i]`` to element ``indices[i]`` (a full index tuple) of ``target_param``."""

    operation: Literal["add_delta"]
    target_param: str
    indices: list[list[Idx]]
    deltas: list[Finite]

    @model_validator(mode="after")
    def _same_length(self) -> "AddDelta":
        if len(self.indices) != len(self.deltas):
            raise ValueError("indices and deltas must have the same length")
        return self


class ScaleOutputChannel(_Strict):
    """Multiply each listed output channel's weights AND bias by its scale factor."""

    operation: Literal["scale_output_channel"]
    target_layer: str
    channel_indices: list[Idx]
    scale_factors: list[Finite]

    @model_validator(mode="after")
    def _same_length(self) -> "ScaleOutputChannel":
        if len(self.channel_indices) != len(self.scale_factors):
            raise ValueError("channel_indices and scale_factors must have the same length")
        return self


class BiasShift(_Strict):
    """Add ``shifts[i]`` to bias element ``indices[i]`` of ``target_layer``."""

    operation: Literal["bias_shift"]
    target_layer: str
    indices: list[Idx]
    shifts: list[Finite]

    @model_validator(mode="after")
    def _same_length(self) -> "BiasShift":
        if len(self.indices) != len(self.shifts):
            raise ValueError("indices and shifts must have the same length")
        return self


class LowRankUpdate(_Strict):
    """Add sum_i u_i (outer) v_i on the ``rows`` x ``cols`` block of a Linear weight (A, B zero-padded)."""

    operation: Literal["low_rank_update"]
    target_param: str
    rows: list[Idx]
    cols: list[Idx]
    u: list[list[Finite]]  # rank x len(rows)
    v: list[list[Finite]]  # rank x len(cols)

    @model_validator(mode="after")
    def _shapes(self) -> "LowRankUpdate":
        if len(self.u) != len(self.v) or not self.u:
            raise ValueError("u and v must be non-empty and have the same rank")
        if any(len(r) != len(self.rows) for r in self.u) or any(len(c) != len(self.cols) for c in self.v):
            raise ValueError("each u_i must have len(rows) entries and each v_i len(cols) entries")
        return self


class NoEdit(_Strict):
    """Explicitly propose no change."""

    operation: Literal["no_edit"]
    reason: str = ""


Operation = Annotated[Union[AddDelta, ScaleOutputChannel, BiasShift, LowRankUpdate, NoEdit],
                      Field(discriminator="operation")]


class Proposal(_Strict):
    """One LLM proposal: up to a few operations plus justification metadata."""

    proposal_id: str
    reason: str
    evidence: list[str]
    expected_effect: str
    confidence: Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
    risk: str
    operations: list[Operation]


def parse_proposal(raw: str | bytes | dict[str, Any]) -> tuple[Proposal | None, list[str]]:
    """Parse JSON text or a dict into a Proposal. Never raises: returns (None, errors) on bad input."""
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (ValueError, TypeError) as err:
        return None, [f"malformed JSON: {err}"]
    if not isinstance(data, dict):
        return None, ["proposal must be a JSON object"]
    try:
        return Proposal.model_validate(data), []
    except ValidationError as err:
        return None, [f"schema: {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()]
