"""Statistics checked against known answers on synthetic predictions (no real data, no test-set access)."""
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from src.analysis import stats as st
from src.analysis import tables as tb
from src.utils.config import load_config
from tests.test_stage8 import cfg_in

CFG = load_config()
N = 2000


def fast_cfg(cfg, n_boot=1000):
    return cfg.model_copy(update={"stats": cfg.stats.model_copy(update={"n_bootstrap": n_boot})})


class Fake:
    """Predictions source built from explicit correctness arrays."""

    def __init__(self, arrays, metrics=None):
        self.arrays, self._m = arrays, metrics or {}

    def correct(self, arm, seed):
        return self.arrays[(arm, seed)].astype(float)

    def metrics(self, arm, seed):
        return self._m.get((arm, seed), {"macro_f1": float(self.arrays[(arm, seed)].mean()), "accuracy": float(self.arrays[(arm, seed)].mean())})


def arrays_with_gain(seed, base_acc=0.84, gain=0.02, n=N):
    rng = np.random.default_rng(seed)
    base = rng.random(n) < base_acc
    better = base.copy()
    flip = np.flatnonzero(~base)[: int(gain * n)]
    better[flip] = True
    return base, better


def test_bootstrap_is_seeded_and_covers_the_true_difference() -> None:
    base, better = arrays_with_gain(0, gain=0.03)
    d = better.astype(float) - base.astype(float)
    a = st.paired_bootstrap(d, 2000, np.random.default_rng(1), 0.95)
    b = st.paired_bootstrap(d, 2000, np.random.default_rng(1), 0.95)
    assert a[:3] == b[:3] and np.array_equal(a[3], b[3])
    assert a[0] == pytest.approx(0.03) and a[1] < 0.03 < a[2] and a[1] > 0
    zero = st.paired_bootstrap(np.zeros(100), 200, np.random.default_rng(0), 0.95)
    assert zero[:3] == (0.0, 0.0, 0.0)
    assert st.one_sided_bootstrap_p(a[3]) < 0.01 and st.one_sided_bootstrap_p(zero[3]) == 1.0


def test_mcnemar_exact_known_counts() -> None:
    a = np.array([1] * 8 + [0] * 2 + [1] * 5 + [0] * 5)
    b = np.array([0] * 8 + [1] * 2 + [1] * 5 + [0] * 5)
    r = st.mcnemar_exact(a, b)
    assert (r["n10"], r["n01"]) == (8, 2) and r["odds_ratio"] == 4.0
    assert r["p_two_sided"] == pytest.approx(sps.binomtest(8, 10, 0.5).pvalue) == pytest.approx(0.109375)
    same = st.mcnemar_exact(a, a)
    assert same["p_two_sided"] == 1.0 and math.isnan(same["odds_ratio"])
    assert math.isinf(st.mcnemar_exact(np.array([1, 1, 0]), np.array([0, 0, 0]))["odds_ratio"])


def test_holm_known_example() -> None:
    adj = st.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert adj == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})
    assert st.holm({"x": 0.9, "y": 0.8})["y"] == pytest.approx(1.0)   # capped at 1


def test_across_seeds_summary() -> None:
    s = st.across_seeds(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0.95)
    half = sps.t.ppf(0.975, 4) * np.std([1, 2, 3, 4, 5], ddof=1) / math.sqrt(5)
    assert s["mean_pp"] == 3.0 and s["ci_lo_pp"] == pytest.approx(3 - half) and s["ci_hi_pp"] == pytest.approx(3 + half)
    assert s["n_positive"] == 5 and s["cohens_d"] == pytest.approx(3 / np.std([1, 2, 3, 4, 5], ddof=1))
    assert s["wilcoxon_p"] == pytest.approx(0.0625)          # smallest possible two-sided p with n = 5
    z = st.across_seeds(np.zeros(5), 0.95)
    assert z["mean_pp"] == 0 and z["ci_lo_pp"] == z["ci_hi_pp"] == 0 and math.isnan(z["wilcoxon_p"]) and math.isnan(z["cohens_d"])


def test_decision_rule_needs_all_three_conditions() -> None:
    assert st.decision(0.01, 0.5, 5, CFG)["supported"]
    assert not st.decision(0.06, 0.5, 5, CFG)["supported"]      # not significant after Holm
    assert not st.decision(0.01, 0.1, 5, CFG)["supported"]      # below the 0.3 pp minimum effect
    assert not st.decision(0.01, 0.5, 3, CFG)["supported"]      # positive in fewer than 4 seeds


def test_compare_per_seed_and_pooled() -> None:
    cfg, seeds = fast_cfg(CFG), [0, 1, 2]
    arrays = {}
    for s in seeds:
        base, better = arrays_with_gain(s, gain=0.02)
        arrays[("E0", s)], arrays[("E2", s)], arrays[("E5", s)] = base, better, base.copy()
    preds = Fake(arrays)
    gain = st.compare(cfg, preds, "P-a", "E2", "E0", seeds)
    assert list(gain.per_seed["diff_pp"]) == pytest.approx([2.0, 2.0, 2.0])
    assert (gain.per_seed["ci_lo_pp"] > 0).all() and (gain.per_seed["n01"] == 0).all() and (gain.per_seed["n10"] == 40).all()
    assert gain.summary["n_positive"] == 3 and gain.summary["pooled_bootstrap_p_one_sided"] < 0.01
    assert gain.summary["pooled_mean_pp"] == pytest.approx(2.0)
    null = st.compare(cfg, preds, "x", "E5", "E0", seeds)                 # identical predictions
    assert (null.per_seed["diff_pp"] == 0).all() and null.summary["pooled_bootstrap_p_one_sided"] == 1.0
    assert null.per_seed["p_two_sided"].eq(1.0).all() and null.summary["n_positive"] == 0


def test_compare_fractional_arm_has_no_mcnemar() -> None:
    cfg, seeds = fast_cfg(CFG), [0]
    base, _ = arrays_with_gain(0)
    frac = (base.astype(float) + np.roll(base, 1).astype(float)) / 2
    r = st.compare(cfg, Fake({("E4", 0): frac, ("E0", 0): base}), "x", "E4", "E0", seeds)
    assert math.isnan(r.per_seed["p_two_sided"].iloc[0])


def synthetic_costs(arms, seeds):
    rows = []
    for arm in arms:
        for s in seeds:
            rows.append({"arm": arm, "seed": s, "changed_scalars": {"E0": 0, "E2": 2000, "E5": 0}.get(arm, 1000), "rel_norm": 0.0 if arm == "E0" else 1e-3,
                         "llm_usd": 0.5 if arm in ("E2", "E3", "E6") else 0.0, "llm_calls": 4, "tokens": 100, "wall_seconds": 10.0, "gpu_seconds": 5.0})
    return pd.DataFrame(rows)


def test_accuracy_and_efficiency_tables(monkeypatch) -> None:
    seeds = [0, 1]
    arrays = {}
    for s in seeds:
        base, better = arrays_with_gain(s, gain=0.02)
        for arm in CFG.analysis.arms:
            arrays[(arm, s)] = better if arm == "E2" else base
    preds = Fake(arrays)
    acc = st.arm_accuracy_table(CFG, preds, seeds)
    e0, e2 = acc.set_index("arm").loc["E0"], acc.set_index("arm").loc["E2"]
    assert e2["mean_acc"] - e0["mean_acc"] == pytest.approx(2.0)
    eff = st.efficiency_table(acc, synthetic_costs(CFG.analysis.arms, seeds), seeds).set_index("arm")
    assert eff.loc["E2", "gain_pp_per_1000_scalars"] == pytest.approx(2.0 / 2.0)          # 2 pp over 2,000 scalars
    assert eff.loc["E2", "gain_pp_per_unit_rel_norm"] == pytest.approx(2.0 / 1e-3)
    assert math.isnan(eff.loc["E5", "gain_pp_per_1000_scalars"])                           # no edit -> undefined, not zero
    assert "E0" not in eff.index


def write_pc_files(cfg, seeds):
    from src.utils.config import load_faults, resolve_path
    root = resolve_path(cfg.paths.pc_results_dir)
    for fault in load_faults().faults:
        for s in seeds:
            d = root / f"seed{s}" / fault
            d.mkdir(parents=True, exist_ok=True)
            for method in ("pure", "random", "grad"):
                rep = {"val_select": {"recovered_fraction": 0.5}, "precondition_drop_ge_threshold": True,
                       "localisation": {"edited_scalars": 10, "precision": 0.1, "recall": 0.2}}
                (d / f"report_{method}.json").write_text(json.dumps(rep))


def pc_predictions(cfg, seeds, n=N):
    from src.utils.config import load_faults
    arrays, rng = {}, np.random.default_rng(5)
    for s in seeds:
        clean = rng.random(n) < 0.84
        arrays[("E0", s)] = clean
        for fault in load_faults().faults:
            faulted = clean & (rng.random(n) < 0.7)                  # drops accuracy to ~0.59
            arrays[(f"PC_{fault}_faulted", s)] = faulted
            pure = clean | faulted
            pure = pure & (rng.random(n) < 0.97) | faulted           # recovers most of the drop
            arrays[(f"PC_{fault}", s)] = pure
            arrays[(f"PC_{fault}_E5", s)] = faulted.copy()
            arrays[(f"PC_{fault}_E7", s)] = faulted | (clean & (rng.random(n) < 0.3))
    return Fake(arrays)


def test_pc_tables_and_binomial(tmp_path) -> None:
    cfg, seeds = cfg_in(tmp_path), [0, 1]
    write_pc_files(cfg, seeds)
    df, p_d = st.pc_tables(cfg, pc_predictions(cfg, seeds), seeds)
    assert len(df) == 3 * 2 * 3 and set(df["method"]) == {"pure", "random", "grad"}
    pure = df[df["method"] == "pure"]
    assert (pure["test_drop_pp"] > 20).all() and (pure["test_recovery"] > 0.8).all() and pure["success"].all()
    assert p_d["successes"] == p_d["runs"] == 6 and p_d["p_one_sided"] == pytest.approx(0.2 ** 6)
    assert df[df["method"] == "random"]["test_recovery"].abs().max() == pytest.approx(0.0)
    grad = df[df["method"] == "grad"]
    assert (grad["test_recovery"] > 0).all() and (grad["test_recovery"] < pure["test_recovery"].min()).all()


def test_write_tables_end_to_end_and_deterministic(tmp_path, monkeypatch) -> None:
    cfg, seeds = fast_cfg(cfg_in(tmp_path), 500), [0, 1]
    cfg = cfg.model_copy(update={"stats": cfg.stats.model_copy(update={"min_positive_seeds": 2})})   # only 2 seeds here
    write_pc_files(cfg, seeds)
    preds = pc_predictions(cfg, seeds)
    for s in seeds:
        base, better = arrays_with_gain(s, gain=0.02)
        for arm in cfg.analysis.arms:
            preds.arrays[(arm, s)] = better if arm in ("E2", "E3") else base
    monkeypatch.setattr(st, "run_cost_table", lambda c, sd: synthetic_costs(c.analysis.arms, sd))
    out1, out2 = tmp_path / "t1", tmp_path / "t2"
    r1 = tb.write_tables(cfg, preds, out1, seeds)
    tb.write_tables(cfg, preds, out2, seeds)
    names = {"arm_accuracy", "arm_overview", "primary_claims", "per_seed_comparisons", "comparison_summary", "efficiency",
             "run_costs", "pc_runs", "pc_summary"}
    for n in names:
        assert (out1 / f"{n}.csv").read_text() == (out2 / f"{n}.csv").read_text()      # reproducible byte for byte
    assert (out1 / "results.md").read_text() == (out2 / "results.md").read_text()
    primary = r1["primary"].set_index("claim")
    assert set(primary.index) == {"P-a", "P-b", "P-c", "P-d"}
    assert primary.loc["P-a", "mean_pp"] == pytest.approx(2.0) and bool(primary.loc["P-a", "supported"])
    assert bool(primary.loc["P-c", "supported"]) and bool(primary.loc["P-b", "supported"])   # E3 and E2 beat E7 / E5 here
    assert r1["comparisons"]["X-1"].summary["mean_pp"] == 0 and r1["comparisons"]["X-1"].summary["pooled_bootstrap_p_one_sided"] == 1.0
    md = (out1 / "results.md").read_text()
    assert "generated by scripts/reproduce_tables.py" in md and "P-d:" in md and "| P-a |" in md
    assert "E2 Pure WSP" in md and "Holm" in md


def test_md_table_formats_missing_and_bool() -> None:
    df = pd.DataFrame({"a": [1.23456, float("nan"), 0.00001234], "b": [True, False, True], "c": ["x", "y", "z"]})
    text = tb.md_table(df, 2)
    assert "1.23" in text and "–" in text and "yes" in text and "no" in text and "1.23e-05" in text
