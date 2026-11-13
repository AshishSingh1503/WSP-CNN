"""Figure data preparation is exact, and every figure renders to PNG + PDF from synthetic inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import figures as fg
from src.analysis import stats as st
from src.analysis import tables as tb
from src.utils.config import load_config, resolve_path
from tests.test_analysis_stats import (Fake, arrays_with_gain, fast_cfg, pc_predictions, synthetic_costs, synthetic_validation,
                                       write_pc_files)
from tests.test_stage8 import cfg_in

CFG = load_config()
SEEDS = [0, 1]
LAYERS = fg.LAYER_ORDER


def state(it, acc_edit, acc_sel):
    return {"iteration": it, "val_edit": {"accuracy": acc_edit}, "val_select": {"accuracy": acc_sel}}


def write_run_files(cfg):
    root = resolve_path(cfg.paths.results_dir)
    for s in SEEDS:
        for arm in cfg.analysis.arms:
            if arm in ("E0", "E1", "E4"):
                continue
            d = root / arm / f"seed{s}"
            d.mkdir(parents=True, exist_ok=True)
            states = [state(0, 0.80, 0.84), state(3, 0.82, 0.845)] if arm == "E2" else [state(0, 0.80, 0.84)]
            changed = {"classifier.6": {"n_changed": 30}, "features.10": {"n_changed": 10}} if arm == "E2" else {}
            (d / "summary.json").write_text(json.dumps({"states": states, "diff_per_layer": changed, "final_index": len(states) - 1,
                                                        "counts": {"accepted": len(states) - 1}, "diff_n_changed": 40 if arm == "E2" else 0,
                                                        "cost": {"cost_usd": 0.5}}))
        d = root / "E1" / f"seed{s}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps({"param_diff": {"per_layer": {"features.0": {"n_changed": 10}, "classifier.1": {"n_changed": 30}}}}))


def test_trajectory_carries_the_accepted_state_forward(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    write_run_files(cfg)
    edit = fg.trajectory_series(cfg, "E2", SEEDS, "val_edit")
    assert edit.shape == (2, cfg.budgets.max_iterations + 1)
    assert edit[0, :3].tolist() == [80.0, 80.0, 80.0] and edit[0, 3:].tolist() == [82.0] * (cfg.budgets.max_iterations - 2)
    sel = fg.trajectory_series(cfg, "E2", SEEDS, "val_select")
    assert sel[1, 0] == 84.0 and sel[1, 10] == 84.5
    flat = fg.trajectory_series(cfg, "E5", SEEDS, "val_edit")
    assert (flat == 80.0).all()


def test_edit_share_matrix_sums_to_one_or_zero(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    write_run_files(cfg)
    share = fg.edit_share_matrix(cfg, SEEDS)
    assert set(share.index) == {"E1", "E2", "E3", "E5", "E6", "E6_RS", "E7"} and list(share.columns) == fg.LAYER_ORDER
    assert share.loc["E2", "classifier.6"] == pytest.approx(0.75) and share.loc["E2", "features.10"] == pytest.approx(0.25)
    assert share.loc["E1", "classifier.1"] == pytest.approx(0.75)
    assert share.loc["E5"].sum() == 0.0 and (share.drop(index=["E5", "E3", "E6", "E6_RS", "E7"]).sum(axis=1) == 1.0).all()


def test_per_class_change_is_mean_difference_to_e0() -> None:
    base = np.arange(10) / 10
    metrics = {}
    for s in SEEDS:
        for arm in CFG.analysis.arms:
            metrics[(arm, s)] = {"per_class_accuracy": (base + (0.01 * (s + 1) if arm == "E2" else 0.0)).tolist(), "macro_f1": 0.8, "accuracy": 0.8}
    preds = Fake({(a, s): np.ones(10, dtype=bool) for a in CFG.analysis.arms for s in SEEDS}, metrics)
    change = fg.per_class_change(CFG, preds, SEEDS)
    assert "E0" not in change.index and change.shape == (len(CFG.analysis.arms) - 1, 10)
    assert change.loc["E2"].tolist() == pytest.approx([1.5] * 10) and change.loc["E3"].abs().max() == 0


def test_t_ci_helper() -> None:
    mean, half = fg.t_ci(np.array([1.0, 2.0, 3.0]), 0.95)
    assert mean == 2.0 and half == pytest.approx(4.303 * 1.0 / np.sqrt(3), rel=1e-3)
    assert fg.t_ci(np.array([5.0, 5.0]), 0.95) == (5.0, 0.0) and fg.t_ci(np.array([5.0]), 0.95) == (5.0, 0.0)


def test_all_figures_render_to_png_and_pdf(tmp_path, monkeypatch) -> None:
    cfg = fast_cfg(cfg_in(tmp_path), 300)
    write_pc_files(cfg, SEEDS)
    write_run_files(cfg)
    preds = pc_predictions(cfg, SEEDS)
    for s in SEEDS:
        base, better = arrays_with_gain(s, gain=0.02)
        for arm in cfg.analysis.arms:
            preds.arrays[(arm, s)] = better if arm == "E2" else base
            preds._m[(arm, s)] = {"macro_f1": 0.8, "accuracy": 0.8, "per_class_accuracy": (np.arange(10) / 10 + (0.02 if arm == "E2" else 0)).tolist()}
    monkeypatch.setattr(st, "run_cost_table", lambda c, sd: synthetic_costs(c.analysis.arms, sd))
    monkeypatch.setattr(st, "validation_table", synthetic_validation)
    out = tmp_path / "figs"
    res = tb.write_tables(cfg, preds, tmp_path / "tables", SEEDS)
    paths = fg.make_all(cfg, preds, res, res["frames"], out)
    names = {p.stem for p in paths}
    assert names == {"fig1_accuracy_by_arm", "fig2_forest_paired_differences", "fig3_gain_vs_changed_scalars", "fig4_validation_trajectories",
                     "fig5_per_class_change", "fig6_edit_location_heatmap", "fig7_pc_recovery", "fig8_cost"}
    assert len(paths) == 16 and all(p.stat().st_size > 1000 for p in paths)
    assert {p.suffix for p in paths} == {".png", ".pdf"}
    assert open(out / "fig1_accuracy_by_arm.png", "rb").read(8) == b"\x89PNG\r\n\x1a\n" and open(out / "fig8_cost.pdf", "rb").read(4) == b"%PDF"


def test_label_spreading_keeps_order_and_minimum_gap() -> None:
    ys = fg.spread([84.0, 84.0, 80.0, 84.01], 0.3)
    assert ys[2] == 80.0 and sorted(ys) == sorted(set(ys))
    order = sorted(range(4), key=lambda i: ys[i])
    assert all(ys[b] - ys[a] >= 0.3 - 1e-12 for a, b in zip(order, order[1:]))
    assert fg.spread([1.0, 5.0], 0.3) == [1.0, 5.0]


def test_reproduce_is_byte_identical_and_check_detects_changes(tmp_path, monkeypatch) -> None:
    from src.analysis import reproduce as rp
    cfg = fast_cfg(cfg_in(tmp_path), 300)
    write_pc_files(cfg, SEEDS)
    write_run_files(cfg)
    preds = pc_predictions(cfg, SEEDS)
    for s in SEEDS:
        base, better = arrays_with_gain(s, gain=0.02)
        for arm in cfg.analysis.arms:
            preds.arrays[(arm, s)] = better if arm == "E2" else base
            preds._m[(arm, s)] = {"macro_f1": 0.8, "accuracy": 0.8, "per_class_accuracy": (np.arange(10) / 10).tolist()}
    monkeypatch.setattr(st, "run_cost_table", lambda c, sd: synthetic_costs(c.analysis.arms, sd))
    monkeypatch.setattr(st, "validation_table", synthetic_validation)
    first = rp.reproduce(cfg, preds, tmp_path / "a" / "tables", tmp_path / "a" / "figures", SEEDS)
    second = rp.reproduce(cfg, preds, tmp_path / "b" / "tables", tmp_path / "b" / "figures", SEEDS)
    assert first == second and any(k.endswith(".png") for k in first) and any(k.endswith(".pdf") for k in first)
    assert any(k.endswith("results.md") for k in first) and any(k.endswith("ablations.csv") for k in first)
    # check(): regenerate into scratch and compare with the stored manifest of cfg.analysis.tables_dir
    tables = Path(cfg.analysis.tables_dir if Path(cfg.analysis.tables_dir).is_absolute() else tmp_path / "results" / "tables")
    stored = tmp_path / "stored_tables"
    stored.mkdir()
    monkeypatch.setattr(rp, "resolve_path", lambda p: stored if p == cfg.analysis.tables_dir else Path(p))
    (stored / rp.MANIFEST_NAME).write_text(json.dumps(first))
    assert rp.check(cfg, preds, tmp_path / "scratch", SEEDS) == []
    changed = dict(first)
    changed[next(iter(changed))] = "0" * 64
    (stored / rp.MANIFEST_NAME).write_text(json.dumps(changed))
    assert rp.check(cfg, preds, tmp_path / "scratch2", SEEDS) == [next(iter(changed))]
