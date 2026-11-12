"""Full controller runs with a scripted FakeLLM on a random-init AlexNet with scripted evaluation functions."""
import copy
import json

import pytest
import torch

from src.utils.reproducibility import sha256_tensor_dict
from src.wsp.llm_agent import BudgetExceeded, CachingClient, CostGuard, FakeLLM, LLMAgent
from tests.wsp_helpers import (BAD_OP, CFG, NO_EDIT, bias_op, make_controller, proposal, read_history)


def script():
    good1 = proposal("good1", bias_op(0.5))
    good2 = proposal("good2", bias_op(0.3, idx=3))
    return [json.dumps([good1, proposal("bad1", BAD_OP), proposal("none1", NO_EDIT)]),   # iteration 1
            json.dumps([good2, proposal("none2", NO_EDIT)]),                              # iteration 2
            "oops, not json", "still not json",                                          # iteration 3: reply + repair both bad
            json.dumps([proposal("none3", NO_EDIT)]),                                     # iteration 4
            json.dumps([proposal("bad2", BAD_OP)])]                                       # iteration 5


def agent_for(fake, mode="pure"):
    return LLMAgent(fake, CFG, mode, tuple(CFG.dsl.allowed_layers), tuple(CFG.dsl.allowed_ops))


@pytest.mark.parametrize("select_slope,expected_final", [(-0.05, 0), (0.05, 2)])
def test_scripted_run_accept_reject_budget_history_and_selection(tmp_path, model, select_slope, expected_final) -> None:
    start_hash = sha256_tensor_dict(model.state_dict())
    ctrl = make_controller(tmp_path, model, agent_for(FakeLLM(script())), select_slope=select_slope)
    res = ctrl.run()
    # control flow
    assert res.stop_reason.startswith("unproductive_x3")
    assert res.counts["accepted"] == 2 and res.counts["iterations"] == 5
    assert res.counts["slots_used"] == 3 + 2 + 0 + 1 + 1
    assert res.counts["evaluations"] == 2                        # only valid, non-no_edit proposals are evaluated
    assert res.counts["validator_rejections"] == 2 and res.counts["no_edits"] == 3
    assert res.counts["failed_llm_outputs"] == 1
    # accepted states: val_edit improves, selection happens on val_select (best accuracy incl. start)
    assert [round(s.val_edit["loss"], 3) for s in res.states] == [2.0, 1.5, 1.2]
    assert res.final_index == expected_final
    assert res.states[res.final_index].val_select["accuracy"] == max(s.val_select["accuracy"] for s in res.states)
    # final model and edit size
    expected_bias = [0.0, 0.5, 0.8][expected_final]
    assert float(res.final_model.classifier[6].bias[3]) == pytest.approx(float(model.classifier[6].bias[3]) + expected_bias, abs=1e-6)
    assert res.diff["n_changed"] == (0 if expected_final == 0 else 1)
    assert sha256_tensor_dict(model.state_dict()) == start_hash   # the start model was never modified
    assert (tmp_path / "ckpt" / "final.pt").is_file() and (tmp_path / "run" / "summary.json").is_file()
    # audit trail
    hist = read_history(tmp_path)
    cands = [r for r in hist if r["type"] == "candidate"]
    assert len(cands) == 7 and [r["decision"] for r in cands[:3]] == ["accepted", "invalid", "no_edit"]
    assert [r["decision"] for r in cands[3:5]] == ["accepted", "no_edit"]
    for r in cands:
        assert {"proposal", "status", "decision", "errors", "n_scalars", "rel_norm", "cumulative_after", "edit_npz",
                "checkpoint_sha256", "val_edit", "val_select"} <= r.keys()
    ok = cands[0]
    assert ok["n_scalars"] == 1 and ok["edit_npz"].endswith("good1.npz") and len(ok["checkpoint_sha256"]) == 64
    assert any("outside" in e for e in cands[1]["errors"])
    assert sum(r["type"] == "llm_failure" for r in hist) == 1 and sum(r["type"] == "accepted_state" for r in hist) == 2
    assert hist[0]["type"] == "start" and hist[-1]["type"] == "end"
    iters = [r for r in hist if r["type"] == "iteration"]
    assert all(u["prompt_sha256"] for r in iters for u in r["usage"]) and len(iters[2]["usage"]) == 2  # reply + repair


def test_history_is_fed_back_to_the_llm(tmp_path, model) -> None:
    fake = FakeLLM(script())
    make_controller(tmp_path, model, agent_for(fake)).run()
    second_user_msg = fake.calls[1]["messages"][0]["content"]       # iteration 2 packet
    packet = json.loads(second_user_msg.split("Diagnostic packet:\n")[1].split("\n\nReply with")[0])
    decisions = [h["decision"] for h in packet["edit_history"]]
    assert decisions == ["accepted", "invalid", "no_edit"]
    assert packet["remaining_candidate_budget"] == CFG.budgets.max_candidates - 3
    assert "gradient_diagnostics" not in packet


def test_pure_mode_never_calls_backward(tmp_path, model, monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("backward() was called in a pure run")
    monkeypatch.setattr(torch.Tensor, "backward", boom)
    fake = FakeLLM([json.dumps([proposal("g", bias_op(0.5))]), json.dumps([proposal("n", NO_EDIT)])] * 2)
    res = make_controller(tmp_path, model, agent_for(fake), max_iterations=2).run()
    assert res.counts["accepted"] == 1


def test_hybrid_mode_does_call_backward(tmp_path, model, monkeypatch) -> None:
    """Sanity check that the monkeypatch above really would catch gradient code."""
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **k: (_ for _ in ()).throw(AssertionError("backward")))
    fake = FakeLLM([json.dumps([proposal("n", NO_EDIT)])])
    with pytest.raises(AssertionError, match="backward"):
        make_controller(tmp_path, model, agent_for(fake, "hybrid"), mode="hybrid", max_iterations=1).run()


def test_cost_cap_stops_the_run_cleanly(tmp_path, model) -> None:
    guard = CostGuard(max_calls=100, max_usd=1.0)
    fake = FakeLLM([json.dumps([proposal("n", NO_EDIT)])], repeat_last=True, cost_per_call=0.6)
    client = CachingClient(fake, tmp_path / "cache", guard, {})
    res = make_controller(tmp_path, model, agent_for(client), stop_after_unproductive=None, max_iterations=10).run()
    assert res.stop_reason.startswith("cost_cap") and len(fake.calls) == 2
    assert res.counts["iterations"] == 3 and res.cost["real_calls"] == 2
    assert (tmp_path / "ckpt" / "final.pt").is_file()           # a clean stop still saves the final state


def test_call_cap_guard() -> None:
    guard = CostGuard(max_calls=1, max_usd=100)
    from src.wsp.llm_agent import Usage
    guard.check()
    guard.record(Usage("m", cost_usd=0.01))
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_start_model_protected_and_unchanged(tmp_path, model) -> None:
    from src.wsp.weight_editor import is_protected
    ctrl = make_controller(tmp_path, model, agent_for(FakeLLM([json.dumps([proposal("n", NO_EDIT)])] * 3)))
    ctrl.run()
    assert is_protected(model)


def test_agent_failure_does_not_crash(tmp_path, model) -> None:
    fake = FakeLLM([])  # exhausted immediately -> LLMError inside propose
    res = make_controller(tmp_path, model, agent_for(fake), max_iterations=2).run()
    assert res.counts["accepted"] == 0 and res.counts["failed_llm_outputs"] == 2


def test_existing_run_is_not_overwritten(tmp_path, model) -> None:
    fake = FakeLLM([json.dumps([proposal("n", NO_EDIT)])] * 4)
    make_controller(tmp_path, model, agent_for(fake), max_iterations=1).run()
    with pytest.raises(FileExistsError):
        make_controller(tmp_path, copy.deepcopy(model), agent_for(fake), max_iterations=1).run()
