"""Ablations are config overrides: each must validate, change exactly what it claims, and leave the rest alone."""
import json
from pathlib import Path

import pytest

from src.diagnostics.packet import build_pure_packet
from src.experiments import ablations as ab
from src.experiments.wsp_common import edit_budget_scalars, make_settings
from src.utils.config import apply_overrides, load_config
from tests.test_packet_isolation import _synthetic
from tests.test_stage8 import cfg_in

CFG = load_config()


def test_every_ablation_config_validates_and_differs_only_where_intended() -> None:
    base = CFG.model_dump(mode="json")
    for ablation_id, d in CFG.ablations.defs.items():
        cfg = ab.ablation_config(CFG, ablation_id)
        new = cfg.model_dump(mode="json")
        changed = {section for section in base if base[section] != new[section]}
        expected = {k.split(".")[0] for k in d.overrides}
        assert changed == expected, (ablation_id, changed, expected)


def test_budget_ablation_budgets_are_exact() -> None:
    sizes = {i: edit_budget_scalars(ab.ablation_config(CFG, i)) for i in ("A6_0.001pct", "A6_0.01pct", "A6_1pct")}
    assert sizes == {"A6_0.001pct": 570, "A6_0.01pct": 5704, "A6_1pct": 570448}
    assert edit_budget_scalars(CFG) == 57044                   # the default 0.1% is the E2 reference row


def test_single_step_ablation_is_one_iteration_of_four_candidates() -> None:
    s = make_settings(ab.ablation_config(CFG, "A8_single_step"), "ABL_A8", "pure")
    assert (s.max_iterations, s.candidates_per_iteration, s.max_candidates) == (1, 4, 4)
    base = make_settings(CFG, "E2", "pure")
    assert (base.max_iterations, base.max_candidates) == (10, 40)


def test_layer_and_inspection_and_model_switches() -> None:
    conv, cls = CFG.ablations.defs["A5_conv"].allowed_layers, CFG.ablations.defs["A5_classifier"].allowed_layers
    assert set(conv) | set(cls) == set(CFG.dsl.allowed_layers) and not set(conv) & set(cls)
    assert ab.ablation_config(CFG, "A9_no_inspection").llm.inspect_enabled is False and CFG.llm.inspect_enabled is True
    a7 = ab.ablation_config(CFG, "A7_second_llm")
    assert a7.llm.model == "claude-sonnet-5-5" and CFG.ablations.defs["A7_second_llm"].env == {"LLM_MODEL": "claude-sonnet-5-5"}
    assert a7.llm.price_input_per_mtok == 2.0 and CFG.llm.model == "claude-opus-5-5"


@pytest.mark.parametrize("ablation_id,empty_field,kept", [("A1", "activation_summary", ("weight_summary", "confusion")),
                                                          ("A2", "weight_summary", ("activation_summary", "confusion")),
                                                          ("A3", "confusion", ("weight_summary", "activation_summary"))])
def test_packet_ablations_remove_exactly_one_section(ablation_id, empty_field, kept) -> None:
    model, kwargs = _synthetic()
    full = build_pure_packet(model, **kwargs)
    cfg = ab.ablation_config(CFG, ablation_id)
    ablated = build_pure_packet(model, **{**kwargs, "cfg": cfg})
    assert getattr(ablated, empty_field) == {} and getattr(full, empty_field) != {}
    for name in kept:                                                    # untouched sections identical (JSON: NaN-safe)
        assert json.dumps(getattr(ablated, name), sort_keys=True) == json.dumps(getattr(full, name), sort_keys=True)
    assert ablated.metrics == full.metrics and ablated.allowed_layers == full.allowed_layers


def test_override_helper_rejects_unknown_keys_and_inconsistent_budgets() -> None:
    with pytest.raises(KeyError):
        apply_overrides(CFG, {"budgets.no_such_key": 1})
    with pytest.raises(Exception):
        apply_overrides(CFG, {"budgets.max_iterations": 2})              # max_candidates must equal iterations x per-iteration


def write_summary(cfg, arm, seed, acc, accepted=1, cost=0.5):
    d = Path(cfg.paths.results_dir) / arm / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"final_index": 0, "states": [{"val_select": {"accuracy": acc}}],
                                                "counts": {"accepted": accepted}, "diff_n_changed": 10, "cost": {"cost_usd": cost}}))


def test_plan_resume_and_stop_on_failure(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    ids, seeds = ["A1", "A2"], [0, 1]
    assert [(i, s) for i, s, done in ab.plan(cfg, ids, seeds) if not done] == [("A1", 0), ("A1", 1), ("A2", 0), ("A2", 1)]
    write_summary(cfg, "ABL_A1", 0, 0.84)
    calls = []

    def runner(ablation_id, seed):
        calls.append((ablation_id, seed))
        if (ablation_id, seed) == ("A2", 0):
            return 1
        write_summary(cfg, f"ABL_{ablation_id}", seed, 0.84)
        return 0

    assert ab.run_ablations(cfg, ids, seeds, True, runner=runner) == 1
    assert calls == [("A1", 1), ("A2", 0)]                               # A1 seed 0 skipped; A2 seed 1 never started
    assert ab.run_ablations(cfg, ids, seeds, False, runner=runner, input_fn=lambda _: "n") == 2
    assert ab.run_ablations(cfg, ids, seeds, True, plan_only=True) == 0


def test_estimate_from_finished_e2_runs(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    assert ab.estimate(cfg, 3)["usd"] == 3 * cfg.llm.max_usd_per_run          # no E2 runs yet -> cap
    write_summary(cfg, "E2", 0, 0.84, cost=0.4)
    write_summary(cfg, "E2", 1, 0.84, cost=0.6)
    est = ab.estimate(cfg, 4)
    assert est["usd"] == pytest.approx(4 * 0.5) and "2 finished" in est["basis"]


def test_ablation_table_reports_vs_e2(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    for s in (0, 1, 2):
        write_summary(cfg, "E2", s, 0.84, accepted=1, cost=0.5)
        write_summary(cfg, "E3", s, 0.85, cost=0.6)
        write_summary(cfg, "ABL_A9_no_inspection", s, 0.86, accepted=2, cost=0.4)
    t = ab.ablation_table(cfg).set_index("ablation")
    assert t.loc["A9_no_inspection", "vs_E2_pp"] == pytest.approx(2.0) and t.loc["A9_no_inspection", "mean_accepted"] == 2
    assert t.loc["A4_hybrid_E3", "vs_E2_pp"] == pytest.approx(1.0) and t.loc["A9_no_inspection", "llm_usd_per_seed"] == pytest.approx(0.4)
    assert "A1" not in t.index                                               # not run -> not reported
