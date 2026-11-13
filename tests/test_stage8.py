"""Stage 8 machinery: sparse checkpoints, run_all resume/estimate, freeze refusals, final_eval (all on fakes)."""
import json
import os
import stat
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from src.experiments import final_eval as fe
from src.experiments import freeze as fz
from src.experiments import run_all as ra
from src.models.checkpoint import clone_model
from src.models.delta import (SwitchingModel, compose, load_delta, overlay_between, realize, save_delta, state_hash_with)
from src.utils.config import DEFAULT_CONFIG_PATH, REPO_ROOT, Config, load_config
from src.utils.hashing import sha256_file
from src.utils.logger import JsonlLogger
from src.utils.reproducibility import sha256_tensor_dict
from src.wsp.edit_schema import parse_proposal
from src.wsp.weight_editor import apply_edit
from tests.wsp_helpers import CFG, bias_op, make_controller, proposal
from src.wsp.llm_agent import FakeLLM, LLMAgent

DSL = CFG.dsl
LAYERS, OPS = tuple(DSL.allowed_layers), tuple(DSL.allowed_ops)


def cfg_in(tmp: Path) -> Config:
    """Real config with every output path redirected into ``tmp``."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for key in ("results_dir", "checkpoints_dir", "finetune_dir", "e1_results_dir", "e0_results_dir", "pc_results_dir",
                "baseline_dir"):
        raw["paths"][key] = str(tmp / key)
    raw["paths"]["registry"] = str(tmp / "registry.json")
    raw["paths"]["frozen_lock"] = str(tmp / "FROZEN_PROTOCOL.lock")
    raw["run_all"]["progress_file"] = str(tmp / "progress.md")
    raw["run_all"]["progress_json"] = str(tmp / "progress.json")
    raw["run_all"]["logs_dir"] = str(tmp / "logs")
    return Config.model_validate(raw)


# ---------------- sparse checkpoints ----------------

def edited(model, *ops):
    out = clone_model(model)
    for i, op in enumerate(ops):
        apply_edit(out, parse_proposal(proposal(f"e{i}", op))[0], DSL, LAYERS, OPS)
    return out


def test_delta_round_trip_and_immutability(tmp_path, model) -> None:
    final = edited(model, bias_op(0.4), {"operation": "scale_output_channel", "target_layer": "features.10",
                                         "channel_indices": [3], "scale_factors": [1.2]})
    info = save_delta(tmp_path / "d.npz", model, final, {"arm": "x"})
    base = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    overlay = load_delta(tmp_path / "d.npz")
    assert info["n_changed"] == 1 + 2305 and sum(len(i) for i, _ in overlay.values()) == 2306
    assert state_hash_with(base, overlay) == info["final_state_sha256"] == sha256_tensor_dict(final.state_dict())
    assert info["start_state_sha256"] == sha256_tensor_dict(model.state_dict())
    assert not os.access(tmp_path / "d.npz", os.W_OK) and (tmp_path / "d.json").is_file()
    with pytest.raises(FileExistsError):
        save_delta(tmp_path / "d.npz", model, final, {})
    assert info["file_sha256"] == sha256_file(tmp_path / "d.npz")


def test_compose_later_overlay_wins(model) -> None:
    a = edited(model, bias_op(0.4, idx=3))
    b = clone_model(a)
    apply_edit(b, parse_proposal(proposal("x", bias_op(0.1, idx=3), bias_op(0.2, idx=5)))[0], DSL, LAYERS, OPS)
    base = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    first, second = overlay_between(model, a), overlay_between(a, b)
    assert state_hash_with(base, compose(first, second)) == sha256_tensor_dict(b.state_dict())
    assert set(compose(first, {}).keys()) == set(first.keys())


def test_switching_model_matches_separate_models_and_restores_base(model) -> None:
    v1 = edited(model, bias_op(0.5))
    v2 = edited(model, {"operation": "scale_output_channel", "target_layer": "features.0", "channel_indices": [1], "scale_factors": [0.6]})
    base = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    sm = SwitchingModel(clone_model(model), base, "cpu")
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for variant, ref in ((overlay_between(model, v1), v1), (overlay_between(model, v2), v2), (None, model), (overlay_between(model, v1), v1)):
            sm.set_variant(variant)
            assert torch.equal(sm(x), ref.eval()(x))
    sm.set_variant(None)
    assert sha256_tensor_dict(sm.model.state_dict()) == sha256_tensor_dict(model.state_dict())


def test_controller_delta_matches_reported_hash(tmp_path, model) -> None:
    fake = FakeLLM([json.dumps([proposal("g", bias_op(0.5))]), json.dumps([proposal("n", {"operation": "no_edit", "reason": ""})])] * 2)
    agent = LLMAgent(fake, CFG, "pure", LAYERS, OPS)
    res = make_controller(tmp_path, model, agent, select_slope=0.05, max_iterations=2).run()
    base = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["checkpoint_sha256"] == res.checkpoint_sha256 == state_hash_with(base, load_delta(Path(summary["delta_path"])))
    assert summary["delta_file_sha256"] == sha256_file(Path(summary["delta_path"])) and summary["start_sha256"] == sha256_tensor_dict(model.state_dict())


# ---------------- run_all ----------------

def fake_outputs(cfg: Config, step: str, seed: int, tag: str = "a") -> None:
    d = ra.step_dir(cfg, step, seed)
    d.mkdir(parents=True, exist_ok=True)
    delta = d / "final_delta.npz"
    delta.write_bytes(tag.encode())
    (d / "summary.json").write_text(json.dumps({"seed": seed, "start_sha256": "s" * 64, "checkpoint_sha256": tag * 64,
                                                "delta_path": str(delta), "cost": {"cost_usd": 0.5}}))


def test_done_marker_requires_matching_hashes(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    assert not ra.is_done(cfg, "E2", 0)
    fake_outputs(cfg, "E2", 0)
    with pytest.raises(RuntimeError):
        ra.mark_done(cfg, "E5", 0)                      # outputs missing -> refuse
    payload = ra.mark_done(cfg, "E2", 0)
    assert ra.is_done(cfg, "E2", 0) and payload["evidence"]["delta_file_sha256"]
    (ra.step_dir(cfg, "E2", 0) / "final_delta.npz").write_bytes(b"tampered")
    assert not ra.is_done(cfg, "E2", 0)                  # changed output invalidates the marker


def test_run_all_skips_done_stops_on_failure_and_writes_progress(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    fake_outputs(cfg, "E2", 0)
    ra.mark_done(cfg, "E2", 0)
    calls = []

    def runner(c, step, seed):
        calls.append((step, seed))
        if step == "E6":
            return 1                                     # a failing step stops the run
        fake_outputs(c, step, seed, tag="b")
        return 0

    code = ra.run_all(cfg, [0], ["E2", "E3", "E5", "E6", "E7"], assume_yes=True, runner=runner)
    assert code == 1 and calls == [("E3", 0), ("E5", 0), ("E6", 0)]    # E2 skipped; E7 never started
    text = (tmp_path / "progress.md").read_text()
    assert "failed" in text and "done" in text and "pending" in text and "| E3 |" in text
    calls.clear()
    code = ra.run_all(cfg, [0], ["E2", "E3", "E5"], assume_yes=True, runner=runner)   # resume: everything done
    assert code == 0 and calls == []


def test_run_all_asks_for_confirmation(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    calls = []
    assert ra.run_all(cfg, [0], ["E5"], assume_yes=False, runner=lambda *a: calls.append(a) or 0, input_fn=lambda _: "n") == 2
    assert calls == []
    assert ra.run_all(cfg, [0], ["E5"], assume_yes=False, runner=lambda c, s, sd: (fake_outputs(c, s, sd), 0)[1], input_fn=lambda _: "y") == 0
    assert ra.run_all(cfg, [0], ["E5"], assume_yes=False, plan_only=True) == 0


def test_estimate_uses_pilot_else_caps(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    plan = ra.build_plan(cfg, [0, 1], ["E2", "E3", "E5", "PC"])
    no_pilot = ra.estimate(cfg, plan, {"runs": []}, None)
    assert no_pilot["llm_runs"] == 2 * (1 + 1 + 3) and no_pilot["usd"] == no_pilot["llm_runs"] * cfg.llm.max_usd_per_run and "UPPER BOUND" in no_pilot["usd_basis"]
    pilot = {"llm_calls": 20, "input_tokens": 100_000, "output_tokens": 30_000, "cost_usd": 2.0}
    est = ra.estimate(cfg, plan, {"runs": [{"step": "E5", "status": "done", "minutes": 4.0}]}, pilot)
    assert est["llm_calls"] == 20 * est["llm_runs"] and est["usd"] == pytest.approx(2.0 * est["llm_runs"]) and est["pending_runs"] == 8
    expected_minutes = 2 * (cfg.run_all.default_minutes["E2"] + cfg.run_all.default_minutes["E3"] + 4.0 + cfg.run_all.default_minutes["PC"])
    assert est["gpu_hours"] == pytest.approx(expected_minutes / 60)


# ---------------- freeze ----------------

def test_freeze_refusals_and_success(tmp_path, monkeypatch) -> None:
    cfg = cfg_in(tmp_path)
    ok_git, ok_audit = (lambda: ("abc123", False)), (lambda c: ([], []))
    with pytest.raises(SystemExit, match="not complete"):                       # nothing has run
        fz.freeze(cfg, [0], ok_git, ok_audit)
    monkeypatch.setattr(fz, "is_done", lambda *a: True)
    monkeypatch.setattr(fz, "collect_entries", lambda c, s: [{"arm": "E0", "seed": 0, "state_sha256": "x"}])
    with pytest.raises(SystemExit, match="audit failed"):
        fz.freeze(cfg, [0], ok_git, lambda c: ([], ["start hash differs"]))
    with pytest.raises(SystemExit, match="dirty"):
        fz.freeze(cfg, [0], lambda: ("abc123", True), ok_audit)
    assert not (tmp_path / "FROZEN_PROTOCOL.lock").exists()
    with pytest.raises(SystemExit, match="env.json"):
        fz.freeze(cfg, [0], ok_git, ok_audit)
    (tmp_path / "results_dir").mkdir(exist_ok=True)
    (tmp_path / "results_dir" / "env.json").write_text(json.dumps({"python": "3.12", "torch": "2.5.1"}))
    token = fz.freeze(cfg, [0], ok_git, ok_audit)
    lock = tmp_path / "FROZEN_PROTOCOL.lock"
    data = json.loads(lock.read_text())
    assert token == sha256_file(lock) and not os.access(lock, os.W_OK)
    assert data["git_commit"] == "abc123" and data["config_hash"] and set(data["prompts"]) == {"pure_wsp_v1.md", "hybrid_wsp_v1.md"}
    assert data["split_manifest_sha256"] and data["env"] and data["checkpoints"][0]["arm"] == "E0"
    with pytest.raises(SystemExit, match="already exists"):
        fz.freeze(cfg, [0], ok_git, ok_audit)


def test_git_state_parses_dirty_and_clean() -> None:
    class R:
        def __init__(self, out): self.stdout = out
    def runner(cmd, **kw):
        return R("deadbeef\n" if cmd[1] == "rev-parse" else " M file.py\n")
    assert fz.git_state(runner) == ("deadbeef", True)
    assert fz.git_state(lambda cmd, **kw: R("deadbeef\n" if cmd[1] == "rev-parse" else ""))[1] is False


# ---------------- final_eval ----------------

def tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(12, 10))


def tiny_setup(cfg):
    base_model = tiny()
    base = {n: p.detach().cpu().clone() for n, p in base_model.named_parameters()}
    variant_model = clone_model(base_model)
    with torch.no_grad():
        variant_model[1].bias[2] += 0.7
    overlay = overlay_between(base_model, variant_model)
    other = tiny()
    with torch.no_grad():
        other[1].weight.mul_(2.0)
    sw = {0: SwitchingModel(clone_model(base_model), base, "cpu")}
    h = lambda m: sha256_tensor_dict(m.state_dict())  # noqa: E731
    variants = [fe.Variant({"arm": "E0", "seed": 0, "state_sha256": h(base_model)}, None, None),
                fe.Variant({"arm": "E2", "seed": 0, "state_sha256": h(variant_model)}, overlay, None),
                fe.Variant({"arm": "E1", "seed": 0, "state_sha256": h(other)}, None, other.eval())]
    refs = {"E0": base_model, "E2": variant_model, "E1": other}
    return sw, variants, refs


def fake_batches(n_batches=3):
    g = torch.Generator().manual_seed(1)
    return [(torch.randn(4, 3, 2, 2, generator=g), torch.randint(0, 10, (4,), generator=g)) for _ in range(n_batches)]


def test_collect_logits_equals_separate_evaluation(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    sw, variants, refs = tiny_setup(cfg)
    batches = fake_batches()
    logits, labels = fe.collect_logits(batches, sw, variants, torch.device("cpu"))
    for v in variants:
        expected = torch.cat([refs[v.entry["arm"]].eval()(x) for x, _ in batches])
        assert torch.equal(logits[v.key], expected), v.key
    assert labels.shape == (12,)


def test_write_results_read_only_and_complete(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    sw, variants, _ = tiny_setup(cfg)
    logits, labels = fe.collect_logits(fake_batches(), sw, variants, torch.device("cpu"))
    m = fe.write_results(cfg, variants[1], logits[variants[1].key], labels, "tok")
    out = fe.out_dir_of(cfg, variants[1].entry)
    assert out == tmp_path / "results_dir" / "test" / "E2" / "seed0"
    for name in ("metrics.json", "predictions.npy", "correct.npy", "logits.npy"):
        assert (out / name).is_file() and not os.access(out / name, os.W_OK)
    assert m["n_test"] == 12 and m["lock_sha256"] == "tok" and {"accuracy", "top5_accuracy", "loss", "confusion_matrix"} <= m.keys()
    assert np.load(out / "predictions.npy").shape == (12,) and np.load(out / "correct.npy").dtype == bool


def run_fake_final_eval(cfg, monkeypatch, allow=False, opened=None):
    sw, variants, _ = tiny_setup(cfg)
    opened = opened if opened is not None else []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)      # the fake models live on the CPU
    monkeypatch.setattr(fe, "load_lock", lambda c, t: {"checkpoints": []})
    monkeypatch.setattr(fe, "check_environment", lambda c, l: None)
    monkeypatch.setattr(fe, "build_variants", lambda c, l, d: (sw, variants))
    monkeypatch.setattr(fe, "get_test_loader", lambda t, c: opened.append(t) or fake_batches())
    monkeypatch.setattr(fe, "measure_latency", lambda *a, **k: {"E0": {"median_ms_per_batch": 1.0}})
    return fe.run_final_eval(cfg, "tok", JsonlLogger(Path(cfg.paths.results_dir) / "log.jsonl", "r", 0, "h"), allow_rerun=allow)


def test_final_eval_runs_once_and_refuses_rerun(tmp_path, monkeypatch) -> None:
    cfg = cfg_in(tmp_path)
    protocol = tmp_path / "PROTOCOL.md"
    protocol.write_text("# P\n\n## 16. Deviations\n")
    monkeypatch.setattr(fe, "REPO_ROOT", tmp_path)
    opened: list[str] = []
    summary = run_fake_final_eval(cfg, monkeypatch, opened=opened)
    assert set(summary) == {"E0/seed0", "E2/seed0", "E1/seed0"} and opened == ["tok"]
    assert fe.ledger_counts(cfg) == {"E0/seed0": 1, "E2/seed0": 1, "E1/seed0": 1}
    assert (tmp_path / "results_dir" / "test" / "latency.json").is_file()
    # a second run without the flag is refused BEFORE the test set is opened
    with pytest.raises(SystemExit, match="already exist"):
        run_fake_final_eval(cfg, monkeypatch, opened=opened)
    assert opened == ["tok"] and "re-run" not in protocol.read_text(encoding="utf-8")
    # with the flag it re-runs, records a dated deviation, and the ledger shows 2 evaluations
    run_fake_final_eval(cfg, monkeypatch, allow=True, opened=opened)
    assert opened == ["tok", "tok"] and set(fe.ledger_counts(cfg).values()) == {2}
    text = protocol.read_text(encoding="utf-8")
    assert f"### {date.today().isoformat()} — Test evaluation re-run" in text and "E2/seed0" in text


def test_lock_and_environment_checks(tmp_path) -> None:
    cfg = cfg_in(tmp_path)
    with pytest.raises(PermissionError, match="not been frozen"):
        fe.load_lock(cfg, "x")
    lock = Path(cfg.paths.frozen_lock)
    lock.write_text(json.dumps({"config_hash": "other", "prompts": {}, "split_manifest_sha256": "z"}))
    with pytest.raises(PermissionError, match="does not match"):
        fe.load_lock(cfg, "wrong")
    data = fe.load_lock(cfg, sha256_file(lock))
    with pytest.raises(RuntimeError, match="config hash changed"):
        fe.check_environment(cfg, data)


def test_final_eval_is_the_only_importer_of_get_test_loader() -> None:
    import ast
    importers = []
    for path in [*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "scripts").glob("*.py"), REPO_ROOT / "main.py"]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and any(a.name == "get_test_loader" for a in node.names):
                importers.append(path.relative_to(REPO_ROOT).as_posix())
    assert importers == ["src/experiments/final_eval.py"]
