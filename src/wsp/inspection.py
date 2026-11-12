"""Read-only weight inspection offered to the LLM (values only; never gradients, never any write)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.wsp.edit_plan import layer_of


def _round(values: Any, sig: int) -> Any:
    if isinstance(values, list):
        return [_round(v, sig) for v in values]
    return float(f"{values:.{sig}g}")


def inspect_slice(model: nn.Module, request: Any, allowed_layers: tuple[str, ...], max_values: int,
                  sig_figs: int) -> dict[str, Any]:
    """Return a slice of one parameter rounded to ``sig_figs`` significant figures, or an error dict.

    ``request`` is ``{"param": "features.10.weight", "slice": [[start, stop], ...]}`` with one half-open
    [start, stop) pair per dimension and at most ``max_values`` values in total.
    """
    if not isinstance(request, dict) or not isinstance(request.get("param"), str) \
            or not isinstance(request.get("slice"), list):
        return {"ok": False, "error": 'inspect needs {"param": str, "slice": [[start, stop], ...]}'}
    params = dict(model.named_parameters())
    name = request["param"]
    if name not in params:
        return {"ok": False, "error": f"unknown parameter {name!r}"}
    if layer_of(name) not in allowed_layers:
        return {"ok": False, "error": f"parameter {name!r} is not in an allowed layer"}
    p, spec = params[name].detach(), request["slice"]
    if len(spec) != p.dim() or any(not (isinstance(s, list) and len(s) == 2
                                        and all(isinstance(v, int) and not isinstance(v, bool) for v in s))
                                   for s in spec):
        return {"ok": False, "error": f"slice needs {p.dim()} [start, stop] integer pairs for shape {list(p.shape)}"}
    if any(not (0 <= a < b <= n) for (a, b), n in zip(spec, p.shape)):
        return {"ok": False, "error": f"slice out of range for shape {list(p.shape)}"}
    count = 1
    for a, b in spec:
        count *= b - a
    if count > max_values:
        return {"ok": False, "error": f"slice has {count} values; the limit is {max_values}"}
    with torch.no_grad():
        block = p[tuple(slice(a, b) for a, b in spec)].double().cpu().tolist()
    return {"ok": True, "param": name, "slice": spec, "shape": [b - a for a, b in spec],
            "values": _round(block, sig_figs)}
