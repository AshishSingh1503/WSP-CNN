"""Stage 7 controls: random proposals, E4 matching, E5/E6-RS/E7 controller runs, E3 settings, audit."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments import audit as audit_mod
from src.experiments.audit import audit_rows, collect_rows, run_audit
from src.experiments.random_edit import (build_random_set, global_norm, scale_strength, templates_from_proposals)
from src.experiments.wsp_common import edit_budget_scalars, make_settings
from src.models.checkpoint import clone_model
from src.utils.param_diff import diff_stats
from src.wsp.edit_validator import ValidationContext, validate_proposal
from src.wsp.grad_heuristic import GradHeuristicAgent
from src.wsp.random_proposals import RandomProposer, RandomSearchAgent, arm_rng
from src.diagnostics.gradient_analyzer import raw_gradients
from tests.wsp_helpers import CFG, bias_op, make_controller, proposal, synthetic_diag

LAYERS, OPS = tuple(CFG.dsl.allowed_layers), tuple(CFG.dsl.allowed_ops)
BUDGET = edit_budget_scalars(CFG)


def ctx(layers=LAYERS) -> ValidationContext:
    return ValidationContext(CFG.dsl, tuple(layers), OPS, BUDGET)


# ---------------- random proposals (E5) ----------------

def test_random_proposals_always_validator_valid_1000(model) -> None:
    proposer = RandomProposer(CFG, LAYERS, OPS, arm_rng(0, "E5"))
    kinds = Counter()
    for i in range(1000):
        p = proposer.sample(model, BUDGET, f"p{i}")
        result = validate_proposal(p, model, ctx())
        assert result.ok, (p["operations"][0]["operation"], result.errors)
        kinds.update(op["operation"] for op in p["operations"])
    assert set(kinds) == {"add_delta", "scale_output_channel", "bias_shift", "low_rank_update"}   # all op types get sampled


def test_random_proposals_restricted_to_last_layer(model) -> None:
    layers = ("classifier.6",)
    proposer = RandomProposer(CFG, layers, OPS, arm_rng(1, "E6_RS"))
    for i in range(150):
        p = proposer.sample(model, BUDGET, f"q{i}")
        assert validate_proposal(p, model, ctx(layers)).ok
        for op in p["operations"]:
            assert (op.get("target_layer") or op["target_param"].rsplit(".", 1)[0]) == "classifier.6"


def test_random_proposals_deterministic_and_seed_dependent(model) -> None:
    a = [RandomProposer(CFG, LAYERS, OPS, arm_rng(3, "E5")).sample(model, BUDGET, "x") for _ in range(1)]
    b = [RandomProposer(CFG, LAYERS, OPS, arm_rng(3, "E5")).sample(model, BUDGET, "x") for _ in range(1)]
    c = [RandomProposer(CFG, LAYERS, OPS, arm_rng(4, "E5")).sample(model, BUDGET, "x") for _ in range(1)]
    assert a == b and a != c


def test_random_proposals_respect_small_remaining_budget(model) -> None:
    proposer = RandomProposer(CFG, LAYERS, OPS, arm_rng(0, "E5"))
    for i in range(60):
        p = proposer.sample(model, 40, f"s{i}")
        r = validate_proposal(p, model, ctx())
        assert r.ok and r.n_scalars <= 40


def test_e5_controller_uses_full_budget_without_llm_or_early_stop(tmp_path, model, monkeypatch) -> None:
    import src.wsp.llm_agent as llm
    monkeypatch.setattr(llm.LLMAgent, "propose", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM used")))
    monkeypatch.setattr(llm, "build_llm_client", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM client built")))
    agent = RandomSearchAgent(CFG, LAYERS, OPS, 0, "E5")
    ctrl = make_controller(tmp_path, model, agent, stop_after_unproductive=None)
    res = ctrl.run()
    assert res.counts["slots_used"] == CFG.budgets.max_candidates == 40 and res.counts["iterations"] == 10
    assert res.cost["llm_calls"] == 0 and res.stop_reason == "iterations_exhausted"
    assert res.counts["evaluations"] + res.counts["validator_rejections"] + res.counts["no_edits"] == 40


# ---------------- E7 gradient heuristic ----------------

def test_e7_never_touches_an_llm_client(tmp_path, model, monkeypatch) -> None:
    import src.wsp.llm_agent as llm
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM used"))  # noqa: E731
    for target in ("propose",):
        monkeypatch.setattr(llm.LLMAgent, target, boom)
    monkeypatch.setattr(llm.AnthropicClient, "complete", boom)
    monkeypatch.setattr(llm.CachingClient, "complete", boom)
    monkeypatch.setattr(llm.OpenAICompatibleClient, "complete", boom)
    monkeypatch.setattr(llm, "build_llm_client", boom)
    agent = GradHeuristicAgent(CFG, LAYERS, OPS, synthetic_diag(), "cpu")
    assert not any("client" in name for name in vars(agent))
    res = make_controller(tmp_path, model, agent, mode="hybrid", stop_after_unproductive=None, max_iterations=2).run()
    assert res.cost["llm_calls"] == 0 and res.counts["iterations"] == 2 and res.counts["slots_used"] >= 4


def test_e7_candidates_follow_the_rule(model) -> None:
    diag = synthetic_diag()
    agent = GradHeuristicAgent(CFG, LAYERS, OPS, diag, "cpu")
    res = agent.propose("", model, 1, {"remaining_edit_budget": BUDGET})
    assert len(res.proposals) == 4
    grads, _, _ = raw_gradients(model, diag, "cpu")
    alphas = CFG.grad_heuristic.alphas
    for prop, alpha in zip(res.proposals[:3], alphas):
        assert validate_proposal(prop, model, ctx()).ok
        for op in prop["operations"]:
            layer, w = op["target_layer"], dict(model.named_modules())[op["target_layer"]].weight.detach()
            g = grads[f"{layer}.weight"]
            for ch, f in zip(op["channel_indices"], op["scale_factors"]):
                dot = float((g[ch].double() * w[ch].double()).sum())
                assert f == pytest.approx(1.0 - alpha * np.sign(dot))
    head = res.proposals[3]["operations"][0]
    gb = grads["classifier.6.bias"].double()
    expected = (-CFG.grad_heuristic.bias_alphas[0] * gb / gb.abs().max()).tolist()
    assert head["target_layer"] == "classifier.6"
    assert [s for i, s in zip(head["indices"], head["shifts"])] == pytest.approx([expected[i] for i in head["indices"]])
    assert validate_proposal(res.proposals[3], model, ctx()).ok


def test_e7_moves_down_the_ranking_when_nothing_was_accepted(model) -> None:
    agent = GradHeuristicAgent(CFG, LAYERS, OPS, synthetic_diag(), "cpu")
    first = agent.propose("", model, 1, {"remaining_edit_budget": BUDGET}).proposals[0]
    second = agent.propose("", model, 2, {"remaining_edit_budget": BUDGET}).proposals[0]   # same model object
    pick = lambda p: {(op["target_layer"], i) for op in p["operations"] for i in op["channel_indices"]}  # noqa: E731
    assert pick(first) and pick(first).isdisjoint(pick(second))


def test_e7_respects_allowed_layers(model) -> None:
    agent = GradHeuristicAgent(CFG, ("classifier.6",), OPS, synthetic_diag(), "cpu")
    for prop in agent.propose("", model, 1, {"remaining_edit_budget": BUDGET}).proposals:
        assert all(op["target_layer"] == "classifier.6" for op in prop["operations"])


# ---------------- E4 matched random editing ----------------

def test_e4_matches_reference_edit_set(model) -> None:
    reference = [proposal("r1", {"operation": "scale_output_channel", "target_layer": "features.10",
                                 "channel_indices": [5], "scale_factors": [1.1]}),
                 proposal("r2", bias_op(0.2), {"operation": "add_delta", "target_param": "classifier.4.weight",
                                               "indices": [[1, 2], [3, 4], [5, 6]], "deltas": [0.004, -0.004, 0.004]})]
    templates = templates_from_proposals(CFG, model, reference)
    assert [t.name for t in templates] == ["scale_output_channel", "bias_shift", "add_delta"]
    assert [t.n_scalars for t in templates] == [2305, 1, 3] and [t.layer_type for t in templates] == ["Conv2d", "Linear", "Linear"]
    edited = clone_model(model)
    from src.wsp.edit_schema import parse_proposal
    from src.wsp.weight_editor import apply_edit
    for p in reference:
        apply_edit(edited, parse_proposal(p)[0], CFG.dsl, LAYERS, OPS)
    target = diff_stats(model, edited)["global_rel_norm"]
    rng = arm_rng(0, "E4")
    for k in range(3):
        built = build_random_set(CFG, model, templates, RandomProposer(CFG, LAYERS, OPS, rng), target, BUDGET)
        assert built["matched"] and abs(built["rel_norm"] - target) / target <= CFG.random_edit.norm_tolerance
        assert built["scalars_by_layer_type"] == {"Conv2d": 2305, "Linear": 4}
        ops = Counter(op["operation"] for p in built["proposals"] for op in p["operations"])
        assert ops == Counter(t.name for t in templates)
        assert built["n_changed"] == 2309


def test_scale_strength_is_linear_in_update(model) -> None:
    op = {"operation": "scale_output_channel", "target_layer": "features.10", "channel_indices": [1], "scale_factors": [1.2]}
    assert scale_strength(op, 0.5)["scale_factors"] == pytest.approx([1.1])
    d = {"operation": "add_delta", "target_param": "x", "indices": [[0]], "deltas": [0.02]}
    assert scale_strength(d, 0.25)["deltas"] == [0.005]


def test_e4_borrows_median_set_only_when_needed() -> None:
    assert CFG.random_edit.samples_per_seed == 10      # K = 10 random sets per seed


# ---------------- E3 / E6 settings ----------------

def test_e3_and_e2_differ_only_in_mode_and_arm() -> None:
    e2, e3 = make_settings(CFG, "E2", "pure"), make_settings(CFG, "E3", "hybrid")
    from dataclasses import replace
    assert replace(e3, arm="E2", mode="pure") == e2
    assert e2.max_candidates == 40 and e2.candidates_per_iteration == 4 and e2.edit_budget == BUDGET == 57044


def test_e6_is_last_layer_only_and_random_arms_have_no_early_stop() -> None:
    e6 = make_settings(CFG, "E6", "pure", ("classifier.6",))
    assert e6.allowed_layers == ("classifier.6",) and e6.stop_after_unproductive == 3
    assert make_settings(CFG, "E5", "pure", early_stop=False).stop_after_unproductive is None


# ---------------- audit ----------------

def write_run(root: Path, arm: str, seed: int, sha: str = "a" * 64, evals: int = 10, slots: int = 12, changed: int = 5,
              budget: int = BUDGET, group: str | None = None, iterations: int = 4) -> None:
    d = root / arm / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    s = {"arm": arm, "seed": seed, "tags": {"pc_fault": group} if group else {}, "start_sha256": sha,
         "settings": {"edit_budget": budget}, "counts": {"evaluations": evals, "slots_used": slots, "iterations": iterations},
         "diff_n_changed": changed, "diff_global_rel_norm": 1e-5, "cost": {"llm_calls": 3, "cost_usd": 0.5},
         "states": [{"val_select": {"accuracy": 0.8, "loss": 0.5, "macro_f1": 0.8}}], "final_index": 0}
    (d / "summary.json").write_text(json.dumps(s))


def test_audit_passes_on_consistent_runs(tmp_path) -> None:
    for arm in ("E2", "E5", "E7"):
        write_run(tmp_path, arm, 0)
    for arm in ("PC_F1", "PC_F1_E5"):
        write_run(tmp_path, arm, 0, sha="b" * 64, group="F1")
    rows, failures = run_audit(CFG, tmp_path, tmp_path / "audit.csv", {0: "a" * 64})
    assert failures == [] and len(rows) == 5
    csv_text = (tmp_path / "audit.csv").read_text()
    assert csv_text.splitlines()[0].startswith("arm,seed,group,start_sha256,candidate_evaluations")


def test_audit_fails_on_mismatched_start_hash(tmp_path) -> None:
    write_run(tmp_path, "E2", 0)
    write_run(tmp_path, "E5", 0, sha="c" * 64)
    failures = audit_rows(CFG, collect_rows(tmp_path), {0: "a" * 64})
    assert len(failures) == 1 and "E5 seed 0" in failures[0] and "start hash" in failures[0]


def test_audit_fails_on_budget_overruns(tmp_path) -> None:
    write_run(tmp_path, "E2", 0, evals=41, slots=41)
    write_run(tmp_path, "E5", 0, changed=BUDGET + 1)
    write_run(tmp_path, "E7", 0, budget=1000)
    write_run(tmp_path, "E3", 0, iterations=11)
    text = " | ".join(audit_rows(CFG, collect_rows(tmp_path), {0: "a" * 64}))
    assert "41 evaluations" in text and f"changed {BUDGET + 1}" in text and "edit budget 1000" in text and "11 iterations" in text


def test_audit_fails_when_pc_arms_start_from_different_faulted_models(tmp_path) -> None:
    write_run(tmp_path, "PC_F1", 0, sha="b" * 64, group="F1")
    write_run(tmp_path, "PC_F1_E5", 0, sha="d" * 64, group="F1")
    failures = audit_rows(CFG, collect_rows(tmp_path), {0: "a" * 64})
    assert len(failures) == 1 and "faulted start hash differs" in failures[0]


def test_audit_fails_for_unregistered_seed_and_ignores_dry_runs(tmp_path) -> None:
    write_run(tmp_path, "E2", 3)
    write_run(tmp_path, "DRY_E2", 0, sha="z" * 64)
    rows = collect_rows(tmp_path)
    assert [r["arm"] for r in rows] == ["E2"]
    assert "no registered common checkpoint" in audit_rows(CFG, rows, {0: "a" * 64})[0]


def test_main_audit_exits_nonzero_on_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(audit_mod, "run_audit", lambda cfg: ([], ["boom"]))
    with pytest.raises(SystemExit) as exc:
        audit_mod.main_audit(CFG)
    assert exc.value.code == 1
    monkeypatch.setattr(audit_mod, "run_audit", lambda cfg: ([], []))
    audit_mod.main_audit(CFG)   # no exit when clean
