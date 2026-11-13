"""Figures (PNG + PDF) generated from the computed tables and the run summaries. No numbers are typed here.

Style follows the dataviz reference: validated categorical palette fixed per arm (colour follows the entity),
thin marks, recessive grid, identity carried by labels and marker shape as well as colour, no dual axes,
blue-grey-red diverging scale for signed change, one-hue sequential ramp for magnitude."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats as sps  # noqa: E402

from src.utils.config import Config, resolve_path  # noqa: E402

MARKERS = {"E0": "o", "E1": "D", "E2": "s", "E3": "^", "E4": "v", "E5": "P", "E6": "X", "E6_RS": "h", "E7": "*"}
LAYER_ORDER = ["features.0", "features.3", "features.6", "features.8", "features.10", "classifier.1", "classifier.4", "classifier.6"]
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]                       # blue - neutral grey - red (palette reference)
SEQUENTIAL = ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]           # blue ramp, light to dark


def style(cfg: Config) -> None:
    """Global matplotlib look: light surface, recessive grid, thin marks, system sans."""
    a = cfg.analysis
    plt.rcParams.update({"figure.facecolor": a.surface, "axes.facecolor": a.surface, "savefig.facecolor": a.surface,
                         "axes.edgecolor": a.grid, "axes.labelcolor": a.ink_secondary, "text.color": a.ink,
                         "xtick.color": a.ink_secondary, "ytick.color": a.ink_secondary, "grid.color": a.grid,
                         "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "regular", "lines.linewidth": 1.6, "legend.frameon": False})


def save(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=160, bbox_inches="tight", metadata={"Software": None})            # no timestamps: byte-reproducible
    fig.savefig(paths[1], bbox_inches="tight", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)
    return paths


def t_ci(values: np.ndarray, level: float) -> tuple[float, float]:
    """Mean and half-width of a t-interval (0 when there is no spread or a single value)."""
    n = len(values)
    if n < 2 or np.std(values, ddof=1) == 0:
        return float(np.mean(values)), 0.0
    return float(np.mean(values)), float(sps.t.ppf(0.5 + level / 2, n - 1) * np.std(values, ddof=1) / math.sqrt(n))


def _ring(ax: plt.Axes, cfg: Config) -> dict[str, Any]:
    return {"edgecolors": cfg.analysis.surface, "linewidths": 1.0, "zorder": 3}


# ---------------- 1. accuracy by arm ----------------

def fig_accuracy_by_arm(cfg: Config, acc: pd.DataFrame, seeds: list[int], out: Path) -> list[Path]:
    """Test accuracy per arm: per-seed points, mean and 95% t-interval."""
    style(cfg)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    arms = list(acc["arm"])
    for i, arm in enumerate(arms):
        y = len(arms) - 1 - i
        vals = acc.loc[acc["arm"] == arm, [f"seed{s}" for s in seeds]].to_numpy(dtype=float).ravel()
        mean, half = t_ci(vals, cfg.analysis.ci_level)
        col = cfg.analysis.colors[arm]
        ax.hlines(y, mean - half, mean + half, color=col, lw=2.0, zorder=2)
        ax.scatter(vals, np.full_like(vals, y) + np.linspace(-0.12, 0.12, len(vals)), s=14, color=col, alpha=0.55, marker=MARKERS[arm], **_ring(ax, cfg))
        ax.scatter([mean], [y], s=52, color=col, marker=MARKERS[arm], **_ring(ax, cfg))
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([cfg.analysis.arm_labels[a] for a in reversed(arms)])
    ax.set_xlabel("test accuracy (%)  -  small points: seeds; large: mean with 95% t-interval")
    ax.set_title("Test accuracy by arm")
    ax.grid(axis="y", visible=False)
    return save(fig, out, "fig1_accuracy_by_arm")


# ---------------- 2. forest plot ----------------

def fig_forest(cfg: Config, per_seed: pd.DataFrame, summary: pd.DataFrame, claim_rows: pd.DataFrame, out: Path) -> list[Path]:
    """Per-seed paired differences (pp) and across-seed mean with 95% t-interval for each comparison."""
    style(cfg)
    ids = list(summary["comparison"])
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(ids) + 1.6))
    primary = set(claim_rows["claim"])
    for i, cid in enumerate(ids):
        y = len(ids) - 1 - i
        row = summary[summary["comparison"] == cid].iloc[0]
        col = cfg.analysis.colors[row["a"]]
        d = per_seed[per_seed["comparison"] == cid]["diff_pp"].to_numpy()
        ax.scatter(d, np.full_like(d, y) + np.linspace(-0.1, 0.1, len(d)), s=16, color=col, alpha=0.5, marker=MARKERS[row["a"]], **_ring(ax, cfg))
        ax.hlines(y, row["ci_lo_pp"], row["ci_hi_pp"], color=col, lw=2.0, zorder=2)
        ax.scatter([row["mean_pp"]], [y], s=64, color=col, marker=MARKERS[row["a"]], **_ring(ax, cfg))
    ax.axvline(0, color=cfg.analysis.ink_secondary, lw=1.0, zorder=1)
    ax.axvline(cfg.stats.min_effect_pp, color=cfg.analysis.grid, lw=1.0, ls="--", zorder=1)
    labels = [f"{('primary ' if cid in primary else '')}{cid}: {summary[summary['comparison'] == cid].iloc[0]['a']} - "
              f"{summary[summary['comparison'] == cid].iloc[0]['b']}" for cid in ids]
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel(f"paired test-accuracy difference (percentage points); dashed line = {cfg.stats.min_effect_pp} pp minimum effect")
    ax.set_title("Paired differences between arms (points: seeds; bar: mean with 95% t-interval over seeds)")
    ax.grid(axis="y", visible=False)
    return save(fig, out, "fig2_forest_paired_differences")


# ---------------- 3. gain vs changed scalars ----------------

def fig_gain_vs_scalars(cfg: Config, eff: pd.DataFrame, out: Path) -> list[Path]:
    """Mean accuracy gain over E0 against mean changed scalars (symlog x so a 0-scalar arm is visible)."""
    style(cfg)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    groups: dict[tuple[float, float], list[str]] = {}
    for _, r in eff.iterrows():
        ax.scatter([r["mean_changed_scalars"]], [r["mean_gain_pp"]], s=70, color=cfg.analysis.colors[r["arm"]], marker=MARKERS[r["arm"]], **_ring(ax, cfg))
        groups.setdefault((round(float(r["mean_changed_scalars"]), 6), round(float(r["mean_gain_pp"]), 6)), []).append(r["arm"])
    for (x, y), arms in groups.items():                # one combined label per coincident group, so labels never overprint
        ax.annotate(", ".join(arms), (x, y), xytext=(8, 6), textcoords="offset points", fontsize=8, color=cfg.analysis.ink)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.axhline(0, color=cfg.analysis.ink_secondary, lw=1.0)
    ax.set_xlabel("mean changed scalars (symlog; 0 = no edit)")
    ax.set_ylabel("test accuracy gain over E0 (pp)")
    ax.set_title("Accuracy gain versus edit size")
    handles = [plt.Line2D([], [], color=cfg.analysis.colors[a], marker=MARKERS[a], ls="", ms=7, label=cfg.analysis.arm_labels[a]) for a in eff["arm"]]
    ax.legend(handles=handles, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    return save(fig, out, "fig3_gain_vs_changed_scalars")


# ---------------- 4. validation trajectories ----------------

def trajectory_series(cfg: Config, arm: str, seeds: list[int], split: str, max_iter: int | None = None) -> np.ndarray:
    """Seeds x (iterations+1) accuracy (%) of the current accepted state after each iteration (carried forward)."""
    max_iter = max_iter if max_iter is not None else cfg.budgets.max_iterations
    root, rows = resolve_path(cfg.paths.results_dir), []
    for seed in seeds:
        s = json.loads((root / arm / f"seed{seed}" / "summary.json").read_text(encoding="utf-8"))
        curve, current, states = [], None, {st["iteration"]: st for st in s["states"]}
        for it in range(max_iter + 1):
            if it in states:                      # a new state was accepted at this iteration; otherwise carry the last one
                current = states[it][split]
            curve.append(100 * current["accuracy"])
        rows.append(curve)
    return np.array(rows)


def fig_trajectories(cfg: Config, seeds: list[int], out: Path, arms: tuple[str, ...] = ("E2", "E3", "E5", "E7")) -> list[Path]:
    """Mean val_edit (solid) and val_select (dashed) accuracy of the accepted state per iteration, same axes."""
    style(cfg)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(cfg.budgets.max_iterations + 1)
    ends: dict[str, list[float]] = {"val_edit": [], "val_select": []}
    for arm in arms:
        col = cfg.analysis.colors[arm]
        edit, sel = (trajectory_series(cfg, arm, seeds, sp).mean(axis=0) for sp in ("val_edit", "val_select"))
        ax.plot(x, edit, color=col, lw=1.8, marker=MARKERS[arm], ms=5, mec=cfg.analysis.surface)
        ax.plot(x, sel, color=col, lw=1.4, ls="--", marker=MARKERS[arm], ms=5, mfc=cfg.analysis.surface)
        ends["val_edit"].append(edit[-1])
        ends["val_select"].append(sel[-1])
    for key, ys in ends.items():                       # direct labels at the right end, nudged apart so they never collide
        gap = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for arm, y in zip(arms, spread(ys, gap)):
            ax.annotate(f"{arm} {'edit' if key == 'val_edit' else 'sel'}", (x[-1], y), xytext=(8, 0), textcoords="offset points",
                        fontsize=7, color=cfg.analysis.ink, va="center", annotation_clip=False)
    handles = [plt.Line2D([], [], color=cfg.analysis.colors[a], marker=MARKERS[a], lw=1.8, ms=5, label=cfg.analysis.arm_labels[a]) for a in arms]
    handles += [plt.Line2D([], [], color=cfg.analysis.ink_secondary, lw=1.8, label="val_edit (used for acceptance)"),
                plt.Line2D([], [], color=cfg.analysis.ink_secondary, lw=1.4, ls="--", label="val_select (held out)")]
    ax.legend(handles=handles, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_xlabel("iteration")
    ax.set_ylabel("validation accuracy of the accepted state (%), mean over seeds")
    ax.set_title("Validation trajectories")
    ax.set_xlim(0, x[-1] + 0.2)
    fig.subplots_adjust(right=0.86)
    return save(fig, out, "fig4_validation_trajectories")


def spread(values: list[float], min_gap: float) -> list[float]:
    """Move label positions apart (keeping order) so neighbours are at least ``min_gap`` apart."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for prev, cur in zip(order, order[1:]):
        out[cur] = max(out[cur], out[prev] + min_gap)
    return out


# ---------------- 5. per-class change ----------------

def per_class_change(cfg: Config, preds: Any, seeds: list[int]) -> pd.DataFrame:
    """Arms x classes: mean over seeds of (per-class test accuracy - E0's), in percentage points."""
    rows = {}
    for arm in cfg.analysis.arms:
        if arm == "E0":
            continue
        diffs = [100 * (np.array(preds.metrics(arm, s)["per_class_accuracy"]) - np.array(preds.metrics("E0", s)["per_class_accuracy"])) for s in seeds]
        rows[arm] = np.mean(diffs, axis=0)
    return pd.DataFrame(rows).T


def fig_per_class(cfg: Config, change: pd.DataFrame, out: Path) -> list[Path]:
    """Diverging heatmap (blue = better than E0, red = worse) with the values written in the cells."""
    style(cfg)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(change) + 1.6))
    lim = max(float(np.abs(change.to_numpy()).max()), 1e-9)
    im = ax.imshow(change.to_numpy(), cmap=LinearSegmentedColormap.from_list("div", DIVERGING), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(change.shape[1]))
    ax.set_xticklabels([f"class {c}" for c in change.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(change)))
    ax.set_yticklabels([cfg.analysis.arm_labels[a] for a in change.index])
    for (i, j), v in np.ndenumerate(change.to_numpy()):
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color=cfg.analysis.ink)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="change in per-class test accuracy vs E0 (pp)", shrink=0.8)
    ax.set_title("Per-class accuracy change relative to E0 (mean over seeds)")
    return save(fig, out, "fig5_per_class_change")


# ---------------- 6. edit location heatmap ----------------

def edit_share_matrix(cfg: Config, seeds: list[int]) -> pd.DataFrame:
    """Arms x layers: share (0-1) of the arm's changed scalars falling in each layer (summed over seeds)."""
    root, rows = resolve_path(cfg.paths.results_dir), {}
    for arm in cfg.analysis.arms:
        if arm in ("E0", "E4"):
            continue
        totals = dict.fromkeys(LAYER_ORDER, 0)
        for seed in seeds:
            path = (root / "E1" / f"seed{seed}" / "metrics.json") if arm == "E1" else (root / arm / f"seed{seed}" / "summary.json")
            doc = json.loads(path.read_text(encoding="utf-8"))
            per_layer = doc["param_diff"]["per_layer"] if arm == "E1" else doc["diff_per_layer"]
            for layer in LAYER_ORDER:
                totals[layer] += per_layer.get(layer, {}).get("n_changed", 0)
        total = sum(totals.values())
        rows[arm] = {k: (v / total if total else 0.0) for k, v in totals.items()}
    return pd.DataFrame(rows).T


def fig_edit_heatmap(cfg: Config, share: pd.DataFrame, out: Path) -> list[Path]:
    """One-hue sequential heatmap of where each arm edited (arms with no edits show an empty row)."""
    style(cfg)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(share) + 1.6))
    im = ax.imshow(share.to_numpy(), cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(share.shape[1]))
    ax.set_xticklabels(share.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels([cfg.analysis.arm_labels[a] for a in share.index])
    for (i, j), v in np.ndenumerate(share.to_numpy()):
        ax.text(j, i, f"{100 * v:.0f}%" if v > 0 else "", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.5 else cfg.analysis.ink)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="share of the arm's changed scalars", shrink=0.8)
    ax.set_title("Where each arm edited (summed over seeds; blank row = no edits)")
    return save(fig, out, "fig6_edit_location_heatmap")


# ---------------- 7. positive-control recovery ----------------

def fig_pc_recovery(cfg: Config, pc: pd.DataFrame, threshold: float, out: Path) -> list[Path]:
    """Test recovery fraction per fault and method (bars: mean over seeds; points: seeds)."""
    style(cfg)
    faults, methods = list(dict.fromkeys(pc["fault"])), ["pure", "random", "grad"]
    arm_of = {"pure": "E2", "random": "E5", "grad": "E7"}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.24
    for k, m in enumerate(methods):
        col = cfg.analysis.colors[arm_of[m]]
        for i, f in enumerate(faults):
            vals = pc[(pc["fault"] == f) & (pc["method"] == m)]["test_recovery"].to_numpy(dtype=float)
            x = i + (k - 1) * (width + 0.04)
            mean = float(np.nanmean(vals))
            ax.bar(x, mean, width=width, color=col, alpha=0.85, zorder=2, label={"pure": "Pure WSP", "random": "E5 random search", "grad": "E7 gradient heuristic"}[m] if i == 0 else None)
            ax.scatter(np.full_like(vals, x), vals, s=26, color=cfg.analysis.ink, marker=MARKERS[arm_of[m]], **_ring(ax, cfg))
            ax.annotate(f"{mean:.2f}", (x, max(mean, 0.0)), xytext=(0, 8 if len(vals) else 0), textcoords="offset points", ha="center", fontsize=7.5, color=cfg.analysis.ink)
    ax.axhline(threshold, color=cfg.analysis.ink_secondary, lw=1.0, ls="--")
    ax.axhline(0, color=cfg.analysis.ink_secondary, lw=1.0)
    ax.set_xticks(range(len(faults)))
    ax.set_xticklabels(faults, rotation=12)
    ax.set_ylabel("fraction of the accuracy drop recovered (test)")
    ax.set_title(f"Positive control: recovery by fault and method (dashed: success threshold {threshold})")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="x", visible=False)
    return save(fig, out, "fig7_pc_recovery")


# ---------------- 8. cost ----------------

def fig_cost(cfg: Config, costs: pd.DataFrame, out: Path) -> list[Path]:
    """Two panels (no dual axis): LLM USD per seed, and wall-clock minutes per run, per arm."""
    style(cfg)
    g = costs.groupby("arm", sort=False).agg(usd=("llm_usd", "mean"), minutes=("wall_seconds", lambda s: s.mean() / 60)).reindex(cfg.analysis.arms)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, col, label in ((axes[0], "usd", "LLM cost per seed (USD)"), (axes[1], "minutes", "wall-clock per run (minutes)")):
        y = np.arange(len(g))[::-1]
        ax.barh(y, g[col].to_numpy(), height=0.55, color=[cfg.analysis.colors[a] for a in g.index], zorder=2)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(g))[::-1])
    axes[0].set_yticklabels([cfg.analysis.arm_labels[a] for a in g.index])
    fig.suptitle("Compute and cost per arm", x=0.5, y=1.0)
    return save(fig, out, "fig8_cost")


def make_all(cfg: Config, preds: Any, res: dict[str, Any], frames: dict[str, pd.DataFrame], out_dir: Path | None = None) -> list[Path]:
    """Draw every figure from the computed results; returns all written paths."""
    out = out_dir or resolve_path(cfg.analysis.figures_dir)
    seeds = res["seeds"]
    paths = fig_accuracy_by_arm(cfg, res["accuracy"], seeds, out)
    paths += fig_forest(cfg, frames["per_seed_comparisons"], frames["comparison_summary"], res["primary"], out)
    paths += fig_gain_vs_scalars(cfg, res["efficiency"], out)
    paths += fig_trajectories(cfg, seeds, out)
    paths += fig_per_class(cfg, per_class_change(cfg, preds, seeds), out)
    paths += fig_edit_heatmap(cfg, edit_share_matrix(cfg, seeds), out)
    from src.utils.config import load_faults
    paths += fig_pc_recovery(cfg, res["pc"], load_faults().repair_success_fraction, out)
    paths += fig_cost(cfg, res["costs"], out)
    return paths
