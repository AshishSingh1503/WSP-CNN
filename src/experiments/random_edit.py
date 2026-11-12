"""E4: random editing without feedback, matched to the E2 final edit set (op types, scalars per layer type, norm)."""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.experiments.wsp_common import WSPData, run_dirs
from src.models.checkpoint import clone_model, require_common_checkpoint
from src.utils.config import Config, resolve_path
from src.utils.logger import JsonlLogger
from src.utils.param_diff import diff_stats
from src.utils.reproducibility import enable_deterministic, set_seed, sha256_tensor_dict
from src.wsp.edit_plan import layer_of, plan_edit
from src.wsp.edit_schema import parse_proposal
from src.wsp.edit_validator import ValidationContext, validate_proposal
from src.wsp.random_proposals import RandomProposer, arm_rng, wrap
from src.wsp.weight_editor import apply_edit


@dataclass
class OpTemplate:
    """One operation of the reference edit set: what it was, where (by layer type) and how big."""

    name: str
    layer_type: str          # "Conv2d" | "Linear"
    n_scalars: int
    param_kind: str = "weight"   # add_delta: weight or bias


def _op_layer(op: dict[str, Any]) -> str:
    return op.get("target_layer") or layer_of(op["target_param"])


def layer_type(model: nn.Module, layer: str) -> str:
    return type(dict(model.named_modules())[layer]).__name__


def templates_from_proposals(cfg: Config, start: nn.Module, proposals: list[dict[str, Any]]) -> list[OpTemplate]:
    """Replay accepted proposals op by op on a clone of ``start`` and record each op's exact scalar count."""
    work, out = clone_model(start), []
    layers, ops = tuple(cfg.dsl.allowed_layers), tuple(cfg.dsl.allowed_ops)
    for prop in proposals:
        for op in prop["operations"]:
            parsed, errors = parse_proposal(wrap("replay", [op]))
            if parsed is None:
                raise ValueError(f"cannot replay op {op}: {errors}")
            rec = apply_edit(work, parsed, cfg.dsl, layers, ops)
            kind = "bias" if op.get("target_param", "").endswith(".bias") else "weight"
            out.append(OpTemplate(op["operation"], layer_type(start, _op_layer(op)), rec.n_scalars, kind))
    return out


def load_e2_reference(cfg: Config, seed: int) -> tuple[list[dict[str, Any]], float, int]:
    """(accepted proposals up to E2's final state, E2 global relative norm, final_index) for ``seed``."""
    res_dir, _ = run_dirs(cfg, "E2", seed)
    if not (res_dir / "summary.json").is_file():
        raise FileNotFoundError(f"E2 results for seed {seed} not found in {res_dir}; run pure_wsp first")
    summary = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
    accepted = [r["proposal"] for r in (json.loads(line) for line in (res_dir / "history.jsonl").read_text().splitlines())
                if r["type"] == "candidate" and r["decision"] == "accepted"]
    return accepted[:summary["final_index"]], summary["diff_global_rel_norm"], summary["final_index"]


def global_norm(model: nn.Module) -> float:
    return sum(float(p.detach().double().pow(2).sum()) for p in model.parameters()) ** 0.5


def scale_strength(op: dict[str, Any], m: float) -> dict[str, Any]:
    """Multiply an op's magnitude by ``m`` (all four op types are linear in their update)."""
    op = json.loads(json.dumps(op))
    if op["operation"] == "add_delta":
        op["deltas"] = [d * m for d in op["deltas"]]
    elif op["operation"] == "bias_shift":
        op["shifts"] = [d * m for d in op["shifts"]]
    elif op["operation"] == "scale_output_channel":
        op["scale_factors"] = [1.0 + m * (f - 1.0) for f in op["scale_factors"]]
    elif op["operation"] == "low_rank_update":
        op["u"] = [[x * m for x in row] for row in op["u"]]
    return op


def _unit_op(cfg: Config, model: nn.Module, proposer: RandomProposer, t: OpTemplate) -> dict[str, Any]:
    """A random op matching template ``t`` in type and size, at unit magnitude (fractions of the bounds)."""
    mods = {n: m for n, m in model.named_modules() if n in proposer.layers and type(m).__name__ == t.layer_type}
    if not mods:
        raise ValueError(f"no allowed {t.layer_type} layer for template {t}")
    names = sorted(mods)
    mag = cfg.random_edit.unit_magnitude
    rng = proposer.rng
    if t.name == "add_delta":
        return proposer.make_add_delta(model, str(rng.choice(names)), t.n_scalars, mag, t.param_kind == "weight")
    if t.name == "bias_shift":
        return proposer.make_bias_shift(model, str(rng.choice(names)), t.n_scalars, mag)
    if t.name == "scale_output_channel":
        sizes = {n: mods[n].weight.numel() // mods[n].weight.shape[0] + 1 for n in names}
        cap = cfg.dsl.scale_output_channel.max_channels
        fit = {n: (max(1, min(cap, round(t.n_scalars / sizes[n]))), ) for n in names}
        err = {n: abs(fit[n][0] * sizes[n] - t.n_scalars) for n in names}
        best = min(err.values())
        layer = str(rng.choice([n for n in names if err[n] == best]))
        return proposer.make_scale(model, layer, fit[layer][0])
    if t.name == "low_rank_update":
        c = cfg.dsl.low_rank_update
        rows = next((r for r in range(min(c.max_rows, t.n_scalars), 0, -1) if t.n_scalars % r == 0 and t.n_scalars // r <= c.max_cols),
                    min(c.max_rows, t.n_scalars))
        cols = max(1, min(c.max_cols, -(-t.n_scalars // rows)))
        return proposer.make_low_rank(model, str(rng.choice(names)), rows, cols,
                                      int(rng.integers(1, cfg.random_search.low_rank_max_rank + 1)), float(rng.uniform(*mag)))
    raise ValueError(f"unknown op {t.name}")


def _delta_sq(model: nn.Module, op: dict[str, Any], cfg: Config) -> float:
    parsed = parse_proposal(wrap("probe", [op]))[0]
    plan = plan_edit(model, parsed, cfg.dsl, tuple(cfg.dsl.allowed_layers), tuple(cfg.dsl.allowed_ops))
    return plan.delta_sq


def build_random_set(cfg: Config, start: nn.Module, templates: list[OpTemplate], proposer: RandomProposer,
                     target_rel_norm: float, edit_budget: int) -> dict[str, Any]:
    """One random edit set matched to ``templates`` and ``target_rel_norm`` (within the configured tolerance)."""
    e4, theta = cfg.random_edit, global_norm(start)
    ctx = ValidationContext(cfg.dsl, tuple(cfg.dsl.allowed_layers), tuple(cfg.dsl.allowed_ops), edit_budget)
    best: dict[str, Any] | None = None
    for attempt in range(1, e4.max_attempts + 1):
        unit = [_unit_op(cfg, start, proposer, t) for t in templates]
        unit_norm = sum(_delta_sq(start, op, cfg) for op in unit) ** 0.5 / theta
        if unit_norm == 0:
            continue
        m = target_rel_norm / unit_norm
        ops = [scale_strength(op, m) for op in unit]
        work, props, ok = clone_model(start), [], True
        for i in range(0, len(ops), cfg.dsl.max_ops_per_proposal):
            prop = wrap(f"e4_a{attempt}_{i}", ops[i:i + cfg.dsl.max_ops_per_proposal])
            res = validate_proposal(prop, work, ctx)
            if not res.ok:
                ok = False
                break
            apply_edit(work, res.proposal, cfg.dsl, tuple(cfg.dsl.allowed_layers), tuple(cfg.dsl.allowed_ops))
            props.append(prop)
        if not ok:
            continue
        stats = diff_stats(start, work)
        rel_err = abs(stats["global_rel_norm"] - target_rel_norm) / target_rel_norm
        per_type: dict[str, int] = {}
        for name, entry in stats["per_tensor"].items():
            lt = layer_type(start, layer_of(name))
            per_type[lt] = per_type.get(lt, 0) + entry["n_changed"]
        cand = {"model": work, "proposals": props, "attempts": attempt, "rel_norm": stats["global_rel_norm"],
                "rel_norm_error": rel_err, "n_changed": stats["n_changed"], "scalars_by_layer_type": per_type,
                "matched": rel_err <= e4.norm_tolerance}
        if best is None or rel_err < best["rel_norm_error"]:
            best = cand
        if cand["matched"]:
            return cand
    if best is None:
        raise RuntimeError("could not build any valid random edit set")
    return best


def run_random_edit(cfg: Config, seed: int, log: JsonlLogger) -> dict[str, Any]:
    """E4 for one seed: K matched random edit sets, each evaluated once on val_select."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.deterministic:
        enable_deterministic(cfg.training.warn_only_deterministic)
    set_seed(seed)
    start, _ = require_common_checkpoint(cfg, seed, device)
    start_sha = sha256_tensor_dict(start.state_dict())
    ref_seed, note = seed, ""
    proposals, target, final_index = load_e2_reference(cfg, seed)
    if final_index == 0:
        others = []
        for s in cfg.seeds.primary:
            try:
                p, n, fi = load_e2_reference(cfg, s)
            except FileNotFoundError:
                continue
            if fi > 0:
                others.append((n, s, p))
        if not others:
            raise RuntimeError("E2 made no edits for this seed and no other seed has E2 edits to borrow")
        target, ref_seed, proposals = sorted(others)[len(others) // 2]
        note = f"E2 made no edits for seed {seed}; used the median E2 edit set (from seed {ref_seed})"
    ref_start = start if ref_seed == seed else require_common_checkpoint(cfg, ref_seed, "cpu")[0]
    templates = templates_from_proposals(cfg, ref_start, proposals)
    data = WSPData(cfg, device)
    from src.experiments.wsp_common import edit_budget_scalars
    budget = edit_budget_scalars(cfg)
    rng = arm_rng(seed, "E4")
    proposer = RandomProposer(cfg, tuple(cfg.dsl.allowed_layers), tuple(cfg.dsl.allowed_ops), rng)
    sel0 = data.eval_select(start)
    reference_types: dict[str, int] = {}
    for t in templates:
        reference_types[t.layer_type] = reference_types.get(t.layer_type, 0) + t.n_scalars
    samples = []
    for k in range(cfg.random_edit.samples_per_seed):
        built = build_random_set(cfg, start, templates, proposer, target, budget)
        sel = data.eval_select(built["model"])
        samples.append({"sample": k, "val_select": {"accuracy": sel.accuracy, "loss": sel.loss, "macro_f1": sel.macro_f1},
                        "checkpoint_sha256": sha256_tensor_dict(built["model"].state_dict()),
                        **{key: built[key] for key in ("attempts", "rel_norm", "rel_norm_error", "n_changed",
                                                       "scalars_by_layer_type", "matched")}})
        log.log("e4_sample", **samples[-1])
        del built
    accs = [s["val_select"]["accuracy"] for s in samples]
    def agg(key: str) -> dict[str, float]:
        v = [s["val_select"][key] for s in samples]
        return {"mean": statistics.fmean(v), "sd": statistics.pstdev(v), "min": min(v), "max": max(v)}
    summary = {"arm": "E4", "seed": seed, "start_sha256": start_sha, "note": note,
               "reference": {"seed": ref_seed, "target_rel_norm": target, "templates": [asdict(t) for t in templates],
                             "scalars_by_layer_type": reference_types},
               "start_val_select": {"accuracy": sel0.accuracy, "loss": sel0.loss, "macro_f1": sel0.macro_f1},
               "samples": samples, "aggregate": {k: agg(k) for k in ("accuracy", "loss", "macro_f1")},
               "all_matched": all(s["matched"] for s in samples), "val_select_evals": len(samples) + 1,
               "settings": {"edit_budget": budget}, "counts": {"evaluations": 0, "slots_used": 0},
               "diff_n_changed": int(max(s["n_changed"] for s in samples)),
               "diff_global_rel_norm": statistics.fmean(s["rel_norm"] for s in samples)}
    res_dir, _ = run_dirs(cfg, "E4", seed)
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    a = summary["aggregate"]["accuracy"]
    print(f"E4 seed {seed}: val_select acc {a['mean']:.4f} +- {a['sd']:.4f} (min {a['min']:.4f}, max {a['max']:.4f}) over "
          f"{len(samples)} matched random edit sets; start {sel0.accuracy:.4f}; all matched: {summary['all_matched']}")
    return summary
