"""LLM agent: reply parsing, repair retry, inspection, cache, prompts, provider retries."""
import json
import re

import pytest

from src.wsp.inspection import inspect_slice
from src.wsp.llm_agent import (AnthropicClient, CachingClient, CostGuard, FakeLLM, LLMAgent, LLMError, Usage,
                               compute_cost, parse_reply)
from tests.wsp_helpers import CFG, NO_EDIT, bias_op, proposal

LAYERS, OPS = tuple(CFG.dsl.allowed_layers), tuple(CFG.dsl.allowed_ops)


def agent(fake, mode="pure", cfg=CFG):
    return LLMAgent(fake, cfg, mode, LAYERS, OPS)


def packet_json() -> str:
    return json.dumps({"metrics": {"loss": 1.0}})


@pytest.mark.parametrize("text,kind", [
    ('[{"operations": []}]', "proposals"), ('{"proposals": [{"a": 1}]}', "proposals"),
    ('{"operations": [], "proposal_id": "x"}', "proposals"), ('```json\n[{"a": 1}]\n```', "proposals"),
    ('Sure! Here you go: [{"a": 1}]', "proposals"), ('{"inspect": {"param": "p"}}', "inspect"),
    ("no json here", "error"), ("[1, 2]", "error"), ("[]", "error"), ('{"x": 1}', "error"), ("[{", "error"),
])
def test_parse_reply(text, kind) -> None:
    assert parse_reply(text)[0] == kind


def test_valid_reply_one_call(model) -> None:
    fake = FakeLLM([json.dumps([proposal("a", bias_op(0.1))])])
    res = agent(fake).propose(packet_json(), model, 1)
    assert len(res.proposals) == 1 and not res.failed and len(fake.calls) == 1


def test_one_repair_retry_then_success(model) -> None:
    fake = FakeLLM(["garbage", json.dumps([proposal("a", bias_op(0.1))])])
    res = agent(fake).propose(packet_json(), model, 1)
    assert len(res.proposals) == 1 and not res.failed and len(fake.calls) == 2
    assert "invalid" in fake.calls[1]["messages"][-1]["content"]       # the validation error is sent back


def test_two_bad_replies_become_a_failed_proposal(model) -> None:
    fake = FakeLLM(["garbage", "more garbage", "never used"])
    res = agent(fake).propose(packet_json(), model, 1)
    assert not res.proposals and len(res.failed) == 1 and len(fake.calls) == 2


def test_extra_proposals_are_dropped(model) -> None:
    many = [proposal(f"p{i}", bias_op(0.1, idx=i)) for i in range(6)]
    res = agent(FakeLLM([json.dumps(many)])).propose(packet_json(), model, 1)
    assert len(res.proposals) == CFG.budgets.candidates_per_iteration and res.extra_dropped == 2


def test_inspection_flow_and_limit(model) -> None:
    ask = json.dumps({"inspect": {"param": "classifier.6.weight", "slice": [[0, 2], [0, 3]]}})
    fake = FakeLLM([ask, ask, ask, ask, json.dumps([proposal("a", bias_op(0.1))])])
    res = agent(fake).propose(packet_json(), model, 1)
    assert len(res.inspections) == CFG.llm.n_inspect == 3 and len(res.proposals) == 1
    reply = json.loads(fake.calls[1]["messages"][-1]["content"].split("Inspection result:\n")[1])
    assert reply["ok"] and reply["shape"] == [2, 3] and len(reply["values"]) == 2
    assert len(fake.calls) == 5                                         # 4 inspect asks (3 served) + proposals


def test_inspection_disabled_by_config(model) -> None:
    import yaml
    from src.utils.config import DEFAULT_CONFIG_PATH, Config
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["llm"]["inspect_enabled"] = False
    cfg = Config.model_validate(raw)
    ask = json.dumps({"inspect": {"param": "classifier.6.weight", "slice": [[0, 1], [0, 1]]}})
    fake = FakeLLM([ask, json.dumps([proposal("a", bias_op(0.1))])])
    res = agent(fake, cfg=cfg).propose(packet_json(), model, 1)
    assert not res.inspections and len(res.proposals) == 1 and "disabled" in agent(fake, cfg=cfg).system


def test_inspect_slice_rules(model) -> None:
    ok = inspect_slice(model, {"param": "features.10.weight", "slice": [[0, 4], [0, 8], [0, 3], [0, 3]]}, LAYERS, 2048, 4)
    assert ok["ok"] and ok["shape"] == [4, 8, 3, 3]
    import numpy as np
    leaves = np.array(ok["values"]).reshape(-1).tolist()
    assert leaves == [float(f"{x:.4g}") for x in leaves] and len(leaves) == 4 * 8 * 3 * 3
    too_big = inspect_slice(model, {"param": "features.10.weight", "slice": [[0, 256], [0, 8], [0, 3], [0, 3]]}, LAYERS, 2048, 4)
    assert not too_big["ok"] and "limit" in too_big["error"]
    for bad in ({"param": "nope", "slice": []}, {"param": "classifier.6.weight", "slice": [[0, 99], [0, 1]]},
                {"param": "classifier.6.weight", "slice": [[0, 1]]}, {"param": "classifier.6.weight", "slice": [[2, 2], [0, 1]]},
                "text", {"param": "features.10.weight"}):
        assert not inspect_slice(model, bad, LAYERS, 2048, 4)["ok"]
    assert not inspect_slice(model, {"param": "features.10.weight", "slice": [[0, 1]] * 4}, ("classifier.6",), 2048, 4)["ok"]


def test_inspection_is_read_only(model) -> None:
    before = {k: v.clone() for k, v in model.state_dict().items()}
    inspect_slice(model, {"param": "classifier.6.weight", "slice": [[0, 2], [0, 2]]}, LAYERS, 2048, 4)
    assert all(bool((before[k] == v).all()) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_cache_hit_returns_identical_proposals(tmp_path, model) -> None:
    fake = FakeLLM([json.dumps([proposal("a", bias_op(0.1))])])
    guard = CostGuard(10, 10.0)
    first = agent(CachingClient(fake, tmp_path / "c", guard, {"effort": "medium"})).propose(packet_json(), model, 1)
    second_agent = agent(CachingClient(FakeLLM([]), tmp_path / "c", guard, {"effort": "medium"}))  # empty script: must not be called
    second = second_agent.propose(packet_json(), model, 1)
    assert second.proposals == first.proposals and len(fake.calls) == 1
    assert second.usages[0].cached and second.usages[0].cost_usd == 0.0 and guard.calls == 1
    assert len(list((tmp_path / "c").glob("*.json"))) == 1


def test_cache_key_depends_on_request(tmp_path, model) -> None:
    fake = FakeLLM([json.dumps([proposal("a", bias_op(0.1))])], repeat_last=True)
    client = CachingClient(fake, tmp_path / "c", CostGuard(10, 10.0), {})
    agent(client).propose(packet_json(), model, 1)
    agent(client).propose(json.dumps({"metrics": {"loss": 2.0}}), model, 1)
    assert len(fake.calls) == 2 and len(list((tmp_path / "c").glob("*.json"))) == 2


def test_prompts_versioned_hashed_and_separated(model) -> None:
    pure, hybrid = agent(FakeLLM([]), "pure"), agent(FakeLLM([]), "hybrid")
    assert pure.prompt_version == "pure_wsp_v1" and hybrid.prompt_version == "hybrid_wsp_v1"
    assert len(pure.prompt_sha256) == 64 and pure.prompt_sha256 != hybrid.prompt_sha256
    assert "gradient" not in pure.system.lower() and "gradient_diagnostics" in hybrid.system
    assert not re.search(r"fault", pure.system.lower()) and not re.search(r"fault", hybrid.system.lower())
    for text in (pure.system, hybrid.system):
        assert "scale_output_channel" in text and "evidence" in text and "{{" not in text
        assert str(CFG.budgets.candidates_per_iteration) in text and "classifier.6" in text
    fake = FakeLLM([json.dumps([proposal("a", NO_EDIT)])])
    res = agent(fake).propose(packet_json(), model, 1)
    assert res.usages[0].prompt_sha256 == pure.prompt_sha256 and res.usages[0].prompt_version == "pure_wsp_v1"


def test_cost_computation() -> None:
    u = Usage("m", input_tokens=1_000_000, output_tokens=500_000, cache_read_tokens=1_000_000, cache_write_tokens=0)
    assert compute_cost(CFG.llm, u) == pytest.approx(4.0 + 10.0 + 0.2)


def test_no_temperature_sent_by_default() -> None:
    client = AnthropicClient(CFG.llm, api_key="test")
    req = client._request([{"role": "user", "content": "hi"}], "sys")
    assert "temperature" not in req and req["output_config"] == {"effort": "medium"} and "thinking" not in req


def test_provider_backoff_and_giving_up(monkeypatch) -> None:
    import anthropic
    sleeps: list[float] = []
    client = AnthropicClient(CFG.llm, api_key="test", sleep=sleeps.append)

    class Boom:
        calls = 0
        def create(self, **kw):
            Boom.calls += 1
            raise anthropic.APIConnectionError(request=None)
    client._client = type("C", (), {"messages": Boom()})()
    with pytest.raises(LLMError, match="gave up"):
        client.complete([{"role": "user", "content": "x"}], "s")
    assert Boom.calls == CFG.llm.max_api_retries + 1 and len(sleeps) == CFG.llm.max_api_retries
    assert sleeps[1] > sleeps[0] - 1 and sleeps[-1] <= CFG.llm.backoff_max_seconds
