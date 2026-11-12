"""Positive-control faults: exact fault sizes, localisation maths, and no leakage to the LLM path."""
import ast
import re

import numpy as np
import pytest
import torch

from src.experiments.positive_control import changed_masks, inject_fault, localisation
from src.models.checkpoint import clone_model
from src.utils.config import REPO_ROOT, load_faults

FAULTS = load_faults()


def test_fault_sizes_match_protocol(model) -> None:
    sizes = {}
    for name, spec in FAULTS.faults.items():
        faulted = clone_model(model)
        record = inject_fault(faulted, spec, seed=0)
        sizes[name] = sum(int(m.sum()) for m in changed_masks(model, faulted).values())
        assert record["kind"] == spec.kind
    assert sizes["F1_negate_features10"] == 8 * 2305
    assert sizes["F2_bias_class3"] == 1
    assert sizes["F3_zero_features3"] == 16 * 1601


def test_fault_channels_are_seeded_and_exact(model) -> None:
    spec = FAULTS.faults["F1_negate_features10"]
    a, b, c = clone_model(model), clone_model(model), clone_model(model)
    ra, rb, rc = inject_fault(a, spec, 0), inject_fault(b, spec, 0), inject_fault(c, spec, 1)
    assert ra["channels"] == rb["channels"] != rc["channels"] and len(set(ra["channels"])) == 8
    ch = ra["channels"][0]
    assert torch.equal(a.features[10].weight[ch], -model.features[10].weight[ch])
    other = next(i for i in range(256) if i not in ra["channels"])
    assert torch.equal(a.features[10].weight[other], model.features[10].weight[other])
    z = clone_model(model)
    zr = inject_fault(z, FAULTS.faults["F3_zero_features3"], 0)
    assert float(z.features[3].weight[zr["channels"][0]].abs().sum()) == 0.0 and float(z.features[3].bias[zr["channels"][0]]) == 0.0


def test_fault_f2_shifts_one_class_bias(model) -> None:
    faulted = clone_model(model)
    inject_fault(faulted, FAULTS.faults["F2_bias_class3"], 0)
    assert float(faulted.classifier[6].bias[3] - model.classifier[6].bias[3]) == pytest.approx(-8.0, abs=1e-5)


def test_localisation_precision_recall() -> None:
    f = {"w": torch.tensor([True, True, False, False])}
    e = {"w": torch.tensor([True, False, True, False])}
    loc = localisation(e, f)
    assert loc == {"edited_scalars": 2, "faulted_scalars": 2, "overlap": 1, "precision": 0.5, "recall": 0.5}
    assert localisation({"w": torch.tensor([False] * 4)}, f)["precision"] == 0.0


def _py_files(*dirs: str):
    return [p for d in dirs for p in (REPO_ROOT / d).rglob("*.py")]


def test_llm_path_never_references_faults() -> None:
    """The LLM/packet/controller code must not import or mention the fault machinery."""
    for path in _py_files("src/wsp", "src/diagnostics"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            assert not any("positive_control" in n or "faults" in n.lower() for n in names), path
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "faults.yaml" not in node.value and "pc_results" not in node.value, path


def test_prompts_and_packets_do_not_mention_faults() -> None:
    for prompt in (REPO_ROOT / "prompts").glob("*.md"):
        assert not re.search(r"fault", prompt.read_text(encoding="utf-8").lower()), prompt
    assert not re.search(r"fault", (REPO_ROOT / "docs" / "EDIT_DSL.md").read_text(encoding="utf-8").lower())
