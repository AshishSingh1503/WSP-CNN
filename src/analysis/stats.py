"""Paired statistics on saved test predictions (PROTOCOL.md Section 12). Every number comes from files."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.utils.config import Config, load_faults, resolve_path


class Predictions(Protocol):
    """Source of per-example correctness (0/1, or the mean over several samples for E4) per arm and seed."""

    def correct(self, arm: str, seed: int) -> np.ndarray: ...
    def metrics(self, arm: str, seed: int) -> dict[str, Any]: ...


class FilePredictions:
    """Reads results/test/{arm}/seed{S}/{correct.npy, metrics.json}; E4 averages its sample folders."""

    def __init__(self, cfg: Config) -> None:
        self.root = resolve_path(cfg.paths.results_dir) / "test"

    def _dirs(self, arm: str, seed: int) -> list[Path]:
        base = self.root / arm / f"seed{seed}"
        samples = sorted(base.glob("sample*"))
        return samples if samples else [base]

    def correct(self, arm: str, seed: int) -> np.ndarray:
        return np.mean([np.load(d / "correct.npy").astype(float) for d in self._dirs(arm, seed)], axis=0)

    def metrics(self, arm: str, seed: int) -> dict[str, Any]:
        runs = [json.loads((d / "metrics.json").read_text(encoding="utf-8")) for d in self._dirs(arm, seed)]
        if len(runs) == 1:
            return runs[0]
        keys = ("accuracy", "loss", "macro_f1", "top5_accuracy")
        out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
        out["per_class_accuracy"] = np.mean([r["per_class_accuracy"] for r in runs], axis=0).tolist()
        return out


# ---------------- paired tests ----------------

def paired_bootstrap(d: np.ndarray, n_boot: int, rng: np.random.Generator, level: float, chunk: int = 250
                     ) -> tuple[float, float, float, np.ndarray]:
    """Mean of per-example differences ``d`` with a percentile CI from ``n_boot`` resamples; also returns the resampled means."""
    n = len(d)
    means = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        means[start:start + size] = d[rng.integers(0, n, size=(size, n))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(d.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)), means


def one_sided_bootstrap_p(boot_means: np.ndarray) -> float:
    """P(mean difference <= 0) with +1 smoothing: (1 + #{resamples <= 0}) / (1 + n_boot)."""
    return float((1 + np.sum(boot_means <= 0)) / (1 + len(boot_means)))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Exact (binomial) McNemar test on binary correctness; discordant counts n10 (A right, B wrong) and n01."""
    a, b = a.astype(bool), b.astype(bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n10 + n01
    p = 1.0 if n == 0 else float(sps.binomtest(n10, n, 0.5).pvalue)
    odds = math.nan if n == 0 else (math.inf if n01 == 0 else n10 / n01)
    return {"n10": n10, "n01": n01, "p_two_sided": p, "odds_ratio": odds}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


def across_seeds(diffs_pp: np.ndarray, level: float) -> dict[str, float]:
    """Mean, SD, t-interval, sign count, Cohen's d and Wilcoxon p over per-seed differences (percentage points)."""
    n = len(diffs_pp)
    mean = float(diffs_pp.mean())
    sd = float(diffs_pp.std(ddof=1)) if n > 1 else math.nan
    half = float(sps.t.ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n)) if n > 1 and sd > 0 else 0.0
    nonzero = diffs_pp[diffs_pp != 0]
    wilcoxon = float(sps.wilcoxon(diffs_pp, zero_method="wilcox").pvalue) if len(nonzero) > 0 else math.nan
    return {"n_seeds": n, "mean_pp": mean, "sd_pp": sd, "ci_lo_pp": mean - half, "ci_hi_pp": mean + half,
            "n_positive": int(np.sum(diffs_pp > 0)), "n_negative": int(np.sum(diffs_pp < 0)),
            "cohens_d": mean / sd if sd and sd > 0 else math.nan, "wilcoxon_p": wilcoxon}


def decision(adj_p: float, mean_pp: float, n_positive: int, cfg: Config) -> dict[str, Any]:
    """PROTOCOL.md Section 12 rule: Holm p < alpha AND mean >= min effect AND positive in >= k seeds."""
    s = cfg.stats
    checks = {"holm_p_below_alpha": adj_p < s.alpha, "mean_at_least_min_effect": mean_pp >= s.min_effect_pp,
              "positive_in_enough_seeds": n_positive >= s.min_positive_seeds}
    return {**checks, "supported": all(checks.values())}


@dataclass
class Comparison:
    """Per-seed rows plus the across-seed summary of one A-vs-B comparison on the test set."""

    id: str
    a: str
    b: str
    per_seed: pd.DataFrame
    summary: dict[str, Any]


def compare(cfg: Config, preds: Predictions, cid: str, a: str, b: str, seeds: list[int]) -> Comparison:
    """Paired comparison of arm ``a`` against arm ``b`` over ``seeds`` (all numbers from the prediction files)."""
    level, n_boot = cfg.analysis.ci_level, cfg.stats.n_bootstrap
    rows, per_example = [], []
    for seed in seeds:
        ca, cb = preds.correct(a, seed), preds.correct(b, seed)
        d = ca - cb
        per_example.append(d)
        rng = np.random.default_rng([cfg.stats.bootstrap_seed, seed])
        mean, lo, hi, _ = paired_bootstrap(d, n_boot, rng, level, cfg.analysis.bootstrap_chunk)
        binary = bool(np.all((ca == 0) | (ca == 1)) and np.all((cb == 0) | (cb == 1)))
        mc = mcnemar_exact(ca, cb) if binary else {"n10": math.nan, "n01": math.nan, "p_two_sided": math.nan, "odds_ratio": math.nan}
        rows.append({"seed": seed, "diff_pp": 100 * mean, "ci_lo_pp": 100 * lo, "ci_hi_pp": 100 * hi, **mc})
    per_seed = pd.DataFrame(rows)
    pooled = np.mean(per_example, axis=0)                  # per-example difference averaged over seeds
    rng = np.random.default_rng([cfg.stats.bootstrap_seed, 10_000])
    mean, lo, hi, boots = paired_bootstrap(pooled, n_boot, rng, level, cfg.analysis.bootstrap_chunk)
    summary = {**across_seeds(per_seed["diff_pp"].to_numpy(), level), "pooled_mean_pp": 100 * mean,
               "pooled_ci_lo_pp": 100 * lo, "pooled_ci_hi_pp": 100 * hi, "pooled_bootstrap_p_one_sided": one_sided_bootstrap_p(boots)}
    return Comparison(cid, a, b, per_seed, summary)


# ---------------- tables ----------------

def arm_accuracy_table(cfg: Config, preds: Predictions, seeds: list[int]) -> pd.DataFrame:
    """Test accuracy per arm and seed, with mean and SD over seeds (E4 = mean of its random edit sets)."""
    rows = []
    for arm in cfg.analysis.arms:
        accs = [100 * float(preds.correct(arm, s).mean()) for s in seeds]
        f1 = [100 * preds.metrics(arm, s)["macro_f1"] for s in seeds]
        rows.append({"arm": arm, **{f"seed{s}": a for s, a in zip(seeds, accs)}, "mean_acc": float(np.mean(accs)),
                     "sd_acc": float(np.std(accs, ddof=1)), "mean_macro_f1": float(np.mean(f1))})
    return pd.DataFrame(rows)


def run_cost_table(cfg: Config, seeds: list[int]) -> pd.DataFrame:
    """Changed scalars, relative norm, LLM USD/calls/tokens and wall time per arm and seed from the run summaries."""
    root, rows = resolve_path(cfg.paths.results_dir), []
    for arm in cfg.analysis.arms:
        for seed in seeds:
            if arm == "E0":
                rows.append({"arm": arm, "seed": seed, "changed_scalars": 0, "rel_norm": 0.0, "llm_usd": 0.0, "llm_calls": 0,
                             "tokens": 0, "wall_seconds": 0.0, "gpu_seconds": 0.0})
            elif arm == "E1":
                m = json.loads((root / "E1" / f"seed{seed}" / "metrics.json").read_text(encoding="utf-8"))
                rows.append({"arm": arm, "seed": seed, "changed_scalars": m["param_diff"]["n_changed"],
                             "rel_norm": m["param_diff"]["global_rel_norm"], "llm_usd": 0.0, "llm_calls": 0, "tokens": 0,
                             "wall_seconds": m["compute"]["wall_seconds"], "gpu_seconds": m["compute"]["gpu_seconds"]})
            else:
                s = json.loads((root / arm / f"seed{seed}" / "summary.json").read_text(encoding="utf-8"))
                c = s.get("cost", {})
                rows.append({"arm": arm, "seed": seed, "changed_scalars": s["diff_n_changed"], "rel_norm": s["diff_global_rel_norm"],
                             "llm_usd": c.get("cost_usd", 0.0), "llm_calls": c.get("llm_calls", 0),
                             "tokens": c.get("input_tokens", 0) + c.get("output_tokens", 0),
                             "wall_seconds": c.get("wall_seconds", 0.0), "gpu_seconds": math.nan})
    return pd.DataFrame(rows)


def validation_table(cfg: Config, acc: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Validation accuracy (%) of each arm's final state, mean over seeds, next to test accuracy.

    Editing arms: the state at ``final_index`` of the run summary. E0/E1: ``metrics.json``. E4: the mean over its random sets
    (val_select only; it stores no per-state val_edit).
    """
    root, rows = resolve_path(cfg.paths.results_dir), []
    test = acc.set_index("arm")["mean_acc"]
    for arm in cfg.analysis.arms:
        edit, select = [], []
        for seed in seeds:
            d = root / arm / f"seed{seed}"
            if arm in ("E0", "E1"):
                m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
                edit.append(m["val_edit"]["accuracy"]); select.append(m["val_select"]["accuracy"])
            elif arm == "E4":
                s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                edit.append(math.nan); select.append(s["aggregate"]["accuracy"]["mean"])
            else:
                s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                final = s["states"][s["final_index"]]
                edit.append(final["val_edit"]["accuracy"]); select.append(final["val_select"]["accuracy"])
        sel_mean = 100 * float(np.mean(select))
        rows.append({"arm": arm, "val_edit_acc": 100 * float(np.mean(edit)), "val_select_acc": sel_mean, "test_acc": float(test[arm]),
                     "val_select_minus_test_pp": sel_mean - float(test[arm])})
    return pd.DataFrame(rows)


def efficiency_table(acc: pd.DataFrame, costs: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Gain over E0 (pp), per 1,000 changed scalars and per unit relative update norm (mean over seeds)."""
    base = acc.set_index("arm").loc["E0", [f"seed{s}" for s in seeds]].to_numpy(dtype=float)
    rows = []
    for arm in acc["arm"]:
        if arm == "E0":
            continue
        gain = acc.set_index("arm").loc[arm, [f"seed{s}" for s in seeds]].to_numpy(dtype=float) - base
        c = costs[costs["arm"] == arm]
        scalars, norm = float(c["changed_scalars"].mean()), float(c["rel_norm"].mean())
        rows.append({"arm": arm, "mean_gain_pp": float(gain.mean()), "mean_changed_scalars": scalars, "mean_rel_norm": norm,
                     "gain_pp_per_1000_scalars": float(gain.mean() / (scalars / 1000)) if scalars > 0 else math.nan,
                     "gain_pp_per_unit_rel_norm": float(gain.mean() / norm) if norm > 0 else math.nan})
    return pd.DataFrame(rows)


def pc_tables(cfg: Config, preds: Predictions, seeds: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Positive-control rows (test and validation recovery, localisation) and the P-d binomial test for Pure WSP."""
    faults, root, rows = load_faults(), resolve_path(cfg.paths.pc_results_dir), []
    suffix = {"pure": "", "random": "_E5", "grad": "_E7"}
    for fault in faults.faults:
        for seed in seeds:
            clean = float(preds.correct("E0", seed).mean())
            faulted = float(preds.correct(f"PC_{fault}_faulted", seed).mean())
            for method, suf in suffix.items():
                rep = json.loads((root / f"seed{seed}" / fault / f"report_{method}.json").read_text(encoding="utf-8"))
                final = float(preds.correct(f"PC_{fault}{suf}", seed).mean())
                rho = (final - faulted) / (clean - faulted) if clean > faulted else math.nan
                rows.append({"fault": fault, "seed": seed, "method": method, "test_clean": 100 * clean, "test_faulted": 100 * faulted,
                             "test_drop_pp": 100 * (clean - faulted), "test_final": 100 * final, "test_recovery": rho,
                             "val_recovery": rep["val_select"]["recovered_fraction"], "precondition_met": rep["precondition_drop_ge_threshold"],
                             "edited_scalars": rep["localisation"]["edited_scalars"], "precision": rep["localisation"]["precision"],
                             "recall": rep["localisation"]["recall"], "success": bool(rho >= faults.repair_success_fraction)})
    df = pd.DataFrame(rows)
    pure = df[(df["method"] == "pure") & df["precondition_met"]]
    k, n = int(pure["success"].sum()), len(pure)
    p = float(sps.binomtest(k, n, faults.h4_null_rate, alternative="greater").pvalue) if n else math.nan
    return df, {"successes": k, "runs": n, "null_rate": faults.h4_null_rate, "p_one_sided": p,
                "excluded_runs": int((df["method"] == "pure").sum() - n)}


def build_all(cfg: Config, preds: Predictions, seeds: list[int] | None = None) -> dict[str, Any]:
    """Compute every table (as DataFrames / dicts). Holm is applied over the primary p-values incl. P-d."""
    seeds = seeds or list(cfg.seeds.primary)
    comps = {c.id: compare(cfg, preds, c.id, c.a, c.b, seeds) for c in [*cfg.analysis.primary, *cfg.analysis.exploratory]}
    pc_df, p_d = pc_tables(cfg, preds, seeds)
    raw = {cid: comps[cid].summary["pooled_bootstrap_p_one_sided"] for cid in (c.id for c in cfg.analysis.primary)}
    raw["P-d"] = p_d["p_one_sided"]
    adj = holm({k: (1.0 if math.isnan(v) else v) for k, v in raw.items()})
    primary = []
    for c in cfg.analysis.primary:
        s = comps[c.id].summary
        primary.append({"claim": c.id, "a": c.a, "b": c.b, "mean_pp": s["mean_pp"], "ci_lo_pp": s["ci_lo_pp"], "ci_hi_pp": s["ci_hi_pp"],
                        "pooled_p_one_sided": raw[c.id], "holm_p": adj[c.id], "n_positive_seeds": s["n_positive"],
                        **decision(adj[c.id], s["mean_pp"], s["n_positive"], cfg)})
    pd_row = {"claim": "P-d", "a": "Pure WSP on PC", "b": f"null rate {p_d['null_rate']}", "mean_pp": math.nan, "ci_lo_pp": math.nan,
              "ci_hi_pp": math.nan, "pooled_p_one_sided": raw["P-d"], "holm_p": adj["P-d"], "n_positive_seeds": p_d["successes"],
              "holm_p_below_alpha": adj["P-d"] < cfg.stats.alpha, "mean_at_least_min_effect": True, "positive_in_enough_seeds": True,
              "supported": adj["P-d"] < cfg.stats.alpha}
    acc = arm_accuracy_table(cfg, preds, seeds)
    costs = run_cost_table(cfg, seeds)
    return {"seeds": seeds, "comparisons": comps, "primary": pd.DataFrame([*primary, pd_row]), "p_d": p_d, "pc": pc_df,
            "accuracy": acc, "costs": costs, "efficiency": efficiency_table(acc, costs, seeds),
            "validation": validation_table(cfg, acc, seeds)}
