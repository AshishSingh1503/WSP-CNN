"""Final evaluation on the official CIFAR-10 test set. The ONLY module that may import ``get_test_loader``.

Every frozen checkpoint is verified against the lock, then evaluated once in a single pass over the test set
(models are switched between batches via sparse overlays), and its results are written read-only."""
from __future__ import annotations

import json
import os
import statistics
import stat
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.data.dataset import get_loader, get_test_loader
from src.experiments.positive_control import inject_fault
from src.models.alexnet import build_model
from src.models.checkpoint import load_checkpoint
from src.models.delta import Overlay, SwitchingModel, compose, load_delta, overlay_between, state_hash_with
from src.training.evaluate import metrics_from_logits
from src.utils.config import REPO_ROOT, Config, config_hash, load_faults, resolve_path
from src.utils.hashing import sha256_file
from src.utils.logger import JsonlLogger
from src.utils.reproducibility import enable_deterministic, sha256_tensor_dict
from src.wsp.edit_schema import parse_proposal
from src.wsp.weight_editor import apply_edit

DEVIATION_FLAG = "--i-understand-this-is-a-deviation"


class Variant:
    """One frozen checkpoint to evaluate: an overlay on its seed's E0, or a full model (E1)."""

    def __init__(self, entry: dict[str, Any], overlay: Overlay | None, full_model: nn.Module | None) -> None:
        self.entry, self.overlay, self.full_model = entry, overlay, full_model
        sample = entry.get("sample")
        self.key = f"{entry['arm']}/seed{entry['seed']}" + (f"/sample{sample}" if sample is not None else "")

    @property
    def seed(self) -> int:
        return self.entry["seed"]


def test_dir(cfg: Config) -> Path:
    return resolve_path(cfg.paths.results_dir) / "test"


def out_dir_of(cfg: Config, entry: dict[str, Any]) -> Path:
    base = test_dir(cfg) / entry["arm"] / f"seed{entry['seed']}"
    return base / f"sample{entry['sample']}" if entry.get("sample") is not None else base


def load_lock(cfg: Config, token: str) -> dict[str, Any]:
    """Read the lock after checking that ``token`` is its SHA-256."""
    path = resolve_path(cfg.paths.frozen_lock)
    if not path.is_file():
        raise PermissionError(f"{path} does not exist: the protocol has not been frozen")
    if sha256_file(path) != token:
        raise PermissionError("final_eval_token does not match the lock file")
    return json.loads(path.read_text(encoding="utf-8"))


def check_environment(cfg: Config, lock: dict[str, Any]) -> None:
    """The code-side inputs must still be exactly what was frozen."""
    problems = []
    if config_hash(cfg) != lock["config_hash"]:
        problems.append("config hash changed since the freeze")
    for name, digest in lock["prompts"].items():
        if sha256_file(resolve_path("prompts") / name) != digest:
            problems.append(f"prompt {name} changed since the freeze")
    if sha256_file(resolve_path(cfg.paths.splits_dir) / "manifest.json") != lock["split_manifest_sha256"]:
        problems.append("split manifest changed since the freeze")
    if problems:
        raise RuntimeError("; ".join(problems))


def _cpu_base(entry: dict[str, Any]) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    model = load_checkpoint(resolve_path(entry["path"]), entry["state_sha256"], model=build_model(pretrained=False), device="cpu")
    return model, {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def _verify(entry: dict[str, Any], base: dict[str, torch.Tensor], overlay: Overlay) -> None:
    got = state_hash_with(base, overlay)
    if got != entry["state_sha256"]:
        raise RuntimeError(f"{entry['arm']} seed {entry['seed']}: rebuilt state hash {got[:12]} != frozen {entry['state_sha256'][:12]}")


def build_variants(cfg: Config, lock: dict[str, Any], device: torch.device) -> tuple[dict[int, SwitchingModel], list[Variant]]:
    """Rebuild and VERIFY every frozen checkpoint against its recorded SHA-256 (no test data is touched here)."""
    entries = lock["checkpoints"]
    bases: dict[int, tuple[nn.Module, dict[str, torch.Tensor]]] = {e["seed"]: _cpu_base(e) for e in entries if e["arm"] == "E0"}
    faults, variants = load_faults().faults, []
    faulted: dict[tuple[int, str], Overlay] = {}
    for e in entries:
        seed, base_model, base = e["seed"], bases[e["seed"]][0], bases[e["seed"]][1]
        if e["arm"] == "E0":
            variants.append(Variant(e, None, None))
        elif e["kind"] == "full":
            model = load_checkpoint(resolve_path(e["path"]), e["state_sha256"], model=build_model(pretrained=False), device=device)
            variants.append(Variant(e, None, model.eval()))
        elif e["kind"] == "faulted_rebuild" or (e["kind"] == "delta" and e.get("pc_fault")):
            key = (seed, e["pc_fault"])
            if key not in faulted:
                work = build_model(pretrained=False)
                work.load_state_dict(base_model.state_dict())
                inject_fault(work, faults[e["pc_fault"]], seed)
                faulted[key] = overlay_between(base, work)
                want = e["state_sha256"] if e["kind"] == "faulted_rebuild" else e["start_sha256"]
                if state_hash_with(base, faulted[key]) != want:
                    raise RuntimeError(f"faulted model for {e['pc_fault']} seed {seed} does not match the frozen hash")
            if e["kind"] == "faulted_rebuild":
                variants.append(Variant(e, faulted[key], None))
            else:
                if sha256_file(Path(e["path"])) != e["file_sha256"]:
                    raise RuntimeError(f"delta file hash mismatch for {e['arm']} seed {seed}")
                overlay = compose(faulted[key], load_delta(Path(e["path"])))
                _verify(e, base, overlay)
                variants.append(Variant(e, overlay, None))
        elif e["kind"] == "delta":
            if sha256_file(Path(e["path"])) != e["file_sha256"]:
                raise RuntimeError(f"delta file hash mismatch for {e['arm']} seed {seed}")
            overlay = load_delta(Path(e["path"]))
            _verify(e, base, overlay)
            variants.append(Variant(e, overlay, None))
        elif e["kind"] == "e4_rebuild":
            if sha256_file(Path(e["path"])) != e["file_sha256"]:
                raise RuntimeError(f"E4 summary hash mismatch for seed {seed}")
            samples = json.loads(Path(e["path"]).read_text(encoding="utf-8"))["samples"]
            work = build_model(pretrained=False)
            work.load_state_dict(base_model.state_dict())
            for prop in samples[e["sample"]]["proposals"]:
                apply_edit(work, parse_proposal(prop)[0], cfg.dsl, tuple(cfg.dsl.allowed_layers), tuple(cfg.dsl.allowed_ops))
            overlay = overlay_between(base, work)
            _verify(e, base, overlay)
            variants.append(Variant(e, overlay, None))
        else:
            raise RuntimeError(f"unknown checkpoint kind {e['kind']!r}")
    switchers = {s: SwitchingModel(bases[s][0], bases[s][1], device) for s in bases}
    for v in variants:
        if v.overlay is not None:
            v.overlay = {n: (i.to(device), val.to(device)) for n, (i, val) in v.overlay.items()}
    return switchers, variants


def _read_only(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def existing_results(cfg: Config, variants: list[Variant]) -> list[str]:
    return [v.key for v in variants if (out_dir_of(cfg, v.entry) / "metrics.json").exists()]


def append_deviation(reasons: list[str], today: date | None = None) -> None:
    """Add a dated entry to PROTOCOL.md Section 16 for a re-run of the test evaluation."""
    path = REPO_ROOT / "PROTOCOL.md"
    text = path.read_text(encoding="utf-8").rstrip()
    day = (today or date.today()).isoformat()
    entry = (f"\n\n### {day} — Test evaluation re-run (automatic entry)\n"
             f"- What changed: final_eval was run again with {DEVIATION_FLAG} although test results already existed for: "
             f"{', '.join(reasons[:20])}{' ...' if len(reasons) > 20 else ''}.\n"
             f"- Were any test results seen before this change? yes (they already existed).\n"
             f"- Affected claims / tables: every analysis that uses these checkpoints must be flagged.\n")
    path.write_text(text + entry, encoding="utf-8")


@torch.no_grad()
def collect_logits(loader: Any, switchers: dict[int, SwitchingModel], variants: list[Variant],
                   device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One pass over ``loader``: every variant sees every batch. Returns ({variant key: logits}, labels)."""
    store: dict[str, list[torch.Tensor]] = {v.key: [] for v in variants}
    labels: list[torch.Tensor] = []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        labels.append(y)
        for v in variants:
            if v.full_model is not None:
                logits = v.full_model(images)
            else:
                sm = switchers[v.seed]
                sm.set_variant(v.overlay)
                logits = sm(images)
            store[v.key].append(logits.float().cpu())
    return {k: torch.cat(parts) for k, parts in store.items()}, torch.cat(labels)


def write_results(cfg: Config, v: Variant, logits: torch.Tensor, labels: torch.Tensor, lock_sha: str) -> dict[str, Any]:
    """metrics.json + predictions.npy (+ correct.npy, logits.npy), all read-only."""
    res = metrics_from_logits(logits, labels, cfg.data.num_classes)
    out = out_dir_of(cfg, v.entry)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "predictions.npy", "correct.npy", "logits.npy"):
        if (out / name).exists():
            os.chmod(out / name, stat.S_IWRITE)
    metrics = {**res.to_dict(), "arm": v.entry["arm"], "seed": v.entry["seed"], "sample": v.entry.get("sample"),
               "state_sha256": v.entry["state_sha256"], "lock_sha256": lock_sha, "n_test": int(len(labels)),
               "batch_size": cfg.loader.batch_size_eval, "precision": "fp32"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    np.save(out / "predictions.npy", res.predictions)
    np.save(out / "correct.npy", (res.predictions == res.labels))
    np.save(out / "logits.npy", logits.numpy())
    for name in ("metrics.json", "predictions.npy", "correct.npy", "logits.npy"):
        _read_only(out / name)
    return metrics


def measure_latency(cfg: Config, switchers: dict[int, SwitchingModel], variants: list[Variant], device: torch.device) -> dict[str, Any]:
    """Per arm (seed-0 model): batch 128, 20 warm-up batches, median of 100 timed batches. Uses val_edit images."""
    images = next(iter(get_loader("val_edit", cfg, batch_size=cfg.loader.batch_size_eval))) [0].to(device)
    out: dict[str, Any] = {}
    seed0 = sorted({v.seed for v in variants})[0]
    for v in [x for x in variants if x.seed == seed0 and x.entry.get("sample") in (None, 0)]:
        run = v.full_model if v.full_model is not None else switchers[seed0]
        if v.full_model is None:
            switchers[seed0].set_variant(v.overlay)
        times = []
        with torch.no_grad():
            for i in range(120):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                run(images)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 20:
                    times.append((time.perf_counter() - start) * 1000)
        med = statistics.median(times)
        out[v.entry["arm"]] = {"median_ms_per_batch": med, "images_per_sec": 1000 * len(images) / med, "batch_size": len(images)}
    return out


def run_final_eval(cfg: Config, token: str, log: JsonlLogger, allow_rerun: bool = False) -> dict[str, Any]:
    """Verify everything, evaluate each frozen checkpoint once on the test set, write results and the ledger."""
    lock = load_lock(cfg, token)
    check_environment(cfg, lock)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.deterministic:
        enable_deterministic(cfg.training.warn_only_deterministic)
    switchers, variants = build_variants(cfg, lock, device)          # verifies every hash; test data not touched yet
    print(f"verified {len(variants)} frozen checkpoints against the lock")
    done = existing_results(cfg, variants)
    if done and not allow_rerun:
        raise SystemExit(f"REFUSING: test results already exist for {len(done)} checkpoints (first: {done[0]}). "
                         f"Re-running is a protocol deviation; pass {DEVIATION_FLAG} to do it and record it.")
    if done:
        append_deviation(done)
    loader = get_test_loader(token, cfg)                              # the single opening of the test set
    logits, labels = collect_logits(loader, switchers, variants, device)
    ledger = test_dir(cfg) / "eval_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    with ledger.open("a", encoding="utf-8") as fh:
        for v in variants:
            summary[v.key] = write_results(cfg, v, logits[v.key], labels, token)
            fh.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "key": v.key, "arm": v.entry["arm"],
                                 "seed": v.seed, "sample": v.entry.get("sample"), "state_sha256": v.entry["state_sha256"],
                                 "lock_sha256": token, "n_test": int(len(labels))}) + "\n")
    latency = measure_latency(cfg, switchers, variants, device)
    lat_path = test_dir(cfg) / "latency.json"
    if lat_path.exists():
        os.chmod(lat_path, stat.S_IWRITE)
    lat_path.write_text(json.dumps(latency, indent=1), encoding="utf-8")
    _read_only(lat_path)
    log.log("final_eval", n_checkpoints=len(variants), lock_sha256=token)
    return summary


def ledger_counts(cfg: Config) -> dict[str, int]:
    """Evaluations recorded per checkpoint key (each should be exactly 1)."""
    path = test_dir(cfg) / "eval_ledger.jsonl"
    counts: dict[str, int] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key = json.loads(line)["key"]
            counts[key] = counts.get(key, 0) + 1
    return counts
