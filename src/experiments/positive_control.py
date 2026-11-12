"""Positive control (PC): inject a known localized fault, then see whether Pure WSP repairs it.

The fault specification lives only here, in configs/faults.yaml and in results/PC/. It is never passed to
the packet builder, the prompt or the agent (a test scans the source to enforce this)."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch import nn

from src.experiments.arms import run_controller
from src.experiments.wsp_common import WSPData, print_run_table, run_dirs
from src.models.checkpoint import clone_model, require_common_checkpoint
from src.utils.config import Config, FaultSpecConfig, FaultsConfig, load_faults, resolve_path
from src.utils.logger import JsonlLogger
from src.utils.reproducibility import enable_deterministic, set_seed
from src.wsp.bitwise import differs


@torch.no_grad()
def inject_fault(model: nn.Module, spec: FaultSpecConfig, seed: int) -> dict[str, Any]:
    """Modify ``model`` in place according to ``spec``; return the record of what was done."""
    params = dict(model.named_parameters())
    if spec.kind in ("negate_channels", "zero_channels"):
        w = params[f"{spec.tensor}.weight"]
        rng = np.random.default_rng(spec.seed_offset + seed)
        channels = sorted(int(c) for c in rng.choice(w.shape[0], spec.n_channels, replace=False))
        factor = -1.0 if spec.kind == "negate_channels" else 0.0
        for suffix in ("weight", "bias"):
            t = params[f"{spec.tensor}.{suffix}"]
            t[channels] = 0.0 if factor == 0.0 else t[channels] * factor
        return {"kind": spec.kind, "tensor": spec.tensor, "channels": channels, "factor": factor}
    if spec.kind == "bias_shift":
        params[spec.tensor + ".bias"][spec.class_index] += spec.shift
        return {"kind": spec.kind, "tensor": spec.tensor, "class_index": spec.class_index, "shift": spec.shift}
    raise ValueError(f"unknown fault kind {spec.kind!r}")


def changed_masks(a: nn.Module, b: nn.Module) -> dict[str, torch.Tensor]:
    """Per-parameter boolean masks of scalars whose bits differ between two models."""
    pb = dict(b.named_parameters())
    return {n: differs(p.detach().cpu(), pb[n].detach().cpu()) for n, p in a.named_parameters()}


def localisation(edited: dict[str, torch.Tensor], faulted: dict[str, torch.Tensor]) -> dict[str, float]:
    """Precision/recall of the edited scalar positions against the faulted ones."""
    e = sum(int(m.sum()) for m in edited.values())
    f = sum(int(m.sum()) for m in faulted.values())
    both = sum(int((edited[n] & faulted[n]).sum()) for n in edited)
    return {"edited_scalars": e, "faulted_scalars": f, "overlap": both,
            "precision": both / e if e else 0.0, "recall": both / f if f else 0.0}


def run_fault(cfg: Config, faults: FaultsConfig, name: str, seed: int, data: WSPData, clean: nn.Module,
              log: JsonlLogger, dry_run: bool, device: torch.device, method: str = "pure") -> dict[str, Any]:
    """One fault: inject, measure the drop, run ``method`` from the faulted model, report recovery."""
    suffix = {"pure": "", "hybrid": "_E3", "random": "_E5", "grad": "_E7"}[method]
    arm = f"PC_{name}{suffix}"
    faulted = clone_model(clean)
    record = inject_fault(faulted, faults.faults[name], seed)
    f_masks = changed_masks(clean, faulted)
    spec_dir = resolve_path(cfg.paths.pc_results_dir) / f"seed{seed}" / name
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "fault_spec.json").write_text(json.dumps({"fault": name, "seed": seed, **record,
                                                          "n_faulted_scalars": sum(int(m.sum()) for m in f_masks.values())}, indent=1))
    np.savez_compressed(spec_dir / "fault_positions.npz", **{n: np.flatnonzero(m.numpy().reshape(-1)) for n, m in f_masks.items() if m.any()})
    clean_sel, faulted_sel = data.eval_select(clean), data.eval_select(faulted)
    clean_edit, faulted_edit = data.eval_edit(clean), data.eval_edit(faulted)
    drop_pp = 100 * (clean_sel.accuracy - faulted_sel.accuracy)
    result = run_controller(cfg, arm, method, seed, faulted, data, device, log, None, dry_run,
                            tags={"pc_fault": name, "pc_method": method})
    res_dir = run_dirs(cfg, result.arm, seed)[0]
    final_sel = result.states[result.final_index].val_select["accuracy"]
    final_edit = result.states[result.final_index].val_edit["accuracy"]
    rho = lambda clean_a, fault_a, final_a: (final_a - fault_a) / (clean_a - fault_a) if clean_a > fault_a else float("nan")  # noqa: E731
    report = {"fault": name, "method": method, "seed": seed, "arm": result.arm, "injected": record,
              "precondition_drop_ge_threshold": drop_pp >= faults.min_val_select_drop_pp,
              "val_select": {"clean": clean_sel.accuracy, "faulted": faulted_sel.accuracy, "final": final_sel,
                             "drop_pp": drop_pp, "recovered_fraction": rho(clean_sel.accuracy, faulted_sel.accuracy, final_sel)},
              "val_edit": {"clean": clean_edit.accuracy, "faulted": faulted_edit.accuracy, "final": final_edit,
                           "recovered_fraction": rho(clean_edit.accuracy, faulted_edit.accuracy, final_edit)},
              "localisation": localisation(changed_masks(faulted, result.final_model), f_masks),
              "stop_reason": result.stop_reason, "counts": result.counts, "cost": result.cost,
              "repair_success_threshold": faults.repair_success_fraction}
    (spec_dir / f"report_{method}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print_run_table(res_dir / "history.jsonl")
    return report


def run_positive_control(cfg: Config, seed: int, log: JsonlLogger, fault: str | None = None,
                         dry_run: bool = False, method: str = "pure") -> list[dict[str, Any]]:
    """All faults (or one) for one seed; prints the recovery table."""
    faults = load_faults()
    names = [fault] if fault else list(faults.faults)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.deterministic:
        enable_deterministic(cfg.training.warn_only_deterministic)
    set_seed(seed)
    clean, _ = require_common_checkpoint(cfg, seed, device)
    data = WSPData(cfg, device)
    methods = ["pure", "random", "grad"] if method == "all" else [method]
    reports = [run_fault(cfg, faults, n, seed, data, clean, log, dry_run, device, m) for n in names for m in methods]
    print(f"\nPOSITIVE CONTROL seed {seed} (val_select; test set not touched)")
    print(f"{'fault':<22}{'method':<7}{'clean':>7}{'faulted':>9}{'drop pp':>9}{'final':>7}{'recovered':>10}{'prec':>6}{'recall':>7}{'scalars':>8}")
    for r in reports:
        v, loc = r["val_select"], r["localisation"]
        print(f"{r['fault']:<22}{r['method']:<7}{v['clean']:>7.4f}{v['faulted']:>9.4f}{v['drop_pp']:>9.2f}{v['final']:>7.4f}"
              f"{100 * v['recovered_fraction']:>9.1f}%{loc['precision']:>6.2f}{loc['recall']:>7.2f}{loc['edited_scalars']:>8}")
    return reports
