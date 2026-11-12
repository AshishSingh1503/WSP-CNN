"""Pure packets are structurally gradient-free: schema, static imports/calls, and runtime grads."""
import ast
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from src.diagnostics.packet import HybridPacket, PurePacket, build_hybrid_packet, build_pure_packet
from src.models.alexnet import build_model
from src.training.evaluate import metrics_from_logits
from src.utils.config import REPO_ROOT, load_config

DIAG = REPO_ROOT / "src" / "diagnostics"
PURE_MODULES = [DIAG / n for n in ("weight_inspector.py", "activation_analyzer.py", "error_analyzer.py")]
PURE_MODULES += sorted((REPO_ROOT / "src" / "wsp").glob("pure*.py"))
GRAD_MODULES = ("gradient_analyzer", "gradient_schema")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _imports(tree: ast.AST) -> list[str]:
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            names += [n.module or ""] + [a.name for a in n.names]
    return names


def _autograd_uses(tree: ast.AST) -> list[str]:
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in {"backward", "grad", "autograd"}:
            bad.append(n.attr)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "requires_grad_":
            bad.append("requires_grad_")
    return bad


def test_pure_packet_rejects_gradient_field() -> None:
    assert "gradient_diagnostics" not in PurePacket.model_fields
    assert not any("grad" in name for name in PurePacket.model_fields)
    with pytest.raises(ValidationError):
        PurePacket.model_validate({**_minimal_fields(), "gradient_diagnostics": {}})


def test_hybrid_is_not_a_pure_packet() -> None:
    assert not issubclass(HybridPacket, PurePacket) and "gradient_diagnostics" in HybridPacket.model_fields


def test_static_scan_pure_paths() -> None:
    packet_tree = _parse(DIAG / "packet.py")
    module_level = [n for n in packet_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("gradient_analyzer" in i for n in module_level for i in _imports(n))
    funcs = {f.name: f for f in ast.walk(packet_tree) if isinstance(f, ast.FunctionDef)}
    for name in ("build_pure_packet", "_pure_fields"):
        assert not any("gradient" in i for i in _imports(funcs[name])), name
        assert not _autograd_uses(funcs[name]), name
    assert any("gradient_analyzer" in i for i in _imports(funcs["build_hybrid_packet"]))  # lazy import there only
    for path in PURE_MODULES:
        tree = _parse(path)
        assert not any(g in i for i in _imports(tree) for g in GRAD_MODULES), path
        assert not _autograd_uses(tree), path


def test_scan_helpers_detect_violations() -> None:
    assert _autograd_uses(ast.parse("loss.backward()")) and _autograd_uses(ast.parse("p.grad"))
    assert any("gradient_analyzer" in i for i in _imports(ast.parse("from src.diagnostics import gradient_analyzer")))


def _synthetic():
    torch.manual_seed(0)
    cfg = load_config()
    model = build_model(pretrained=False, seed=0).eval()
    images = torch.randn(10, 3, 224, 224)
    labels = torch.arange(10)
    batches = [(images[:5], labels[:5]), (images[5:], labels[5:])]
    with torch.no_grad():
        logits = torch.cat([model(x) for x, _ in batches])
    kwargs = dict(eval_result=metrics_from_logits(logits, labels), val_edit_indices=torch.arange(10).numpy() + 100,
                  diag_batches=batches, device="cpu", cfg=cfg, seed=0, iteration=1,
                  remaining_edit_budget=57044, remaining_candidate_budget=40, edit_history=[])
    return model, kwargs


def _minimal_fields() -> dict:
    model, kwargs = _synthetic()
    return json.loads(build_pure_packet(model, **kwargs).model_dump_json())


def test_building_pure_packet_leaves_grads_none() -> None:
    model, kwargs = _synthetic()
    assert all(p.grad is None for p in model.parameters())
    packet = build_pure_packet(model, **kwargs)
    assert all(p.grad is None for p in model.parameters())
    assert "gradient" not in packet.model_dump_json().lower()


def test_hybrid_has_gradients_and_cleans_up() -> None:
    model, kwargs = _synthetic()
    flags = [p.requires_grad for p in model.parameters()]
    packet = build_hybrid_packet(model, **kwargs)
    assert packet.gradient_diagnostics.n_images == 10 and len(packet.gradient_diagnostics.layers) == 8
    assert all(p.grad is None for p in model.parameters())
    assert [p.requires_grad for p in model.parameters()] == flags
    assert len(packet.gradient_diagnostics.classifier6_rows) == 10
