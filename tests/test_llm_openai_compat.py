"""OpenAI-compatible provider (Groq / xAI Grok / generic): request shape, parsing, retries, errors. No network."""
import json

import pytest

from src.wsp.llm_agent import (CachingClient, CostGuard, LLMAgent, LLMError, OpenAICompatibleClient, build_llm_client)
from tests.wsp_helpers import CFG, NO_EDIT, proposal

OK = {"model": "served-model", "choices": [{"message": {"content": '[{"a": 1}]'}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 1200, "completion_tokens": 80}}


class FakeTransport:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, body))
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def client(transport, sleeps=None):
    return OpenAICompatibleClient(CFG.llm, "https://example.test/v1/", "secret-key", "some-model", transport,
                                  sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


def test_request_shape_and_parsing() -> None:
    t = FakeTransport((200, {}, OK))
    resp = client(t).complete([{"role": "user", "content": "hi"}], "SYS")
    url, headers, body = t.calls[0]
    assert url == "https://example.test/v1/chat/completions" and headers["Authorization"] == "Bearer secret-key"
    assert body["messages"][0] == {"role": "system", "content": "SYS"} and body["messages"][1]["content"] == "hi"
    assert body["model"] == "some-model" and body["temperature"] == CFG.llm.temperature and "output_config" not in body
    assert resp.text == '[{"a": 1}]' and resp.usage.input_tokens == 1200 and resp.usage.output_tokens == 80
    assert resp.usage.cost_usd == 0.0 and resp.usage.model == "served-model" and resp.stop_reason == "stop"


def test_think_blocks_stripped_and_length_flagged() -> None:
    data = {"choices": [{"message": {"content": "<think>reasoning...\nmore</think>\n[1]"}, "finish_reason": "length"}], "usage": {}}
    resp = client(FakeTransport((200, {}, data))).complete([{"role": "user", "content": "x"}], "s")
    assert resp.text == "[1]" and resp.stop_reason == "max_tokens"


def test_rate_limit_retries_with_retry_after() -> None:
    sleeps: list[float] = []
    t = FakeTransport((429, {"retry-after": "7"}, {"error": {"message": "slow down"}}), (429, {}, {"error": {}}), (200, {}, OK))
    resp = client(t, sleeps).complete([{"role": "user", "content": "x"}], "s")
    assert len(t.calls) == 3 and 7 <= sleeps[0] <= 8 and sleeps[1] >= CFG.llm.backoff_base_seconds * 2 and resp.text


def test_non_retryable_errors_raise_immediately() -> None:
    for status in (400, 401, 413):
        t = FakeTransport((status, {}, {"error": {"message": "request too large"}}))
        with pytest.raises(LLMError, match=str(status)):
            client(t).complete([{"role": "user", "content": "x"}], "s")
        assert len(t.calls) == 1


def test_gives_up_after_retries() -> None:
    t = FakeTransport((503, {}, {"error": {"message": "down"}}))
    with pytest.raises(LLMError, match="gave up"):
        client(t).complete([{"role": "user", "content": "x"}], "s")
    assert len(t.calls) == CFG.llm.max_api_retries + 1


def test_agent_works_with_this_provider(model) -> None:
    reply = {"choices": [{"message": {"content": json.dumps([proposal("a", NO_EDIT)])}, "finish_reason": "stop"}], "usage": {}}
    agent = LLMAgent(client(FakeTransport((200, {}, reply))), CFG, "pure", tuple(CFG.dsl.allowed_layers), tuple(CFG.dsl.allowed_ops))
    res = agent.propose("{}", model, 1)
    assert len(res.proposals) == 1 and not res.failed


@pytest.mark.parametrize("provider,key_var", [("groq", "GROQ_API_KEY"), ("xai", "XAI_API_KEY"), ("openai_compatible", "LLM_API_KEY")])
def test_build_client_from_env(monkeypatch, tmp_path, provider, key_var) -> None:
    for var in ("GROQ_API_KEY", "XAI_API_KEY", "LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)  # ignore the real .env
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", "m1")
    with pytest.raises(SystemExit, match=key_var):
        build_llm_client(CFG, CostGuard(5, 1.0))
    monkeypatch.setenv(key_var, "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://h.test/v1")
    caching, model = build_llm_client(CFG, CostGuard(5, 1.0))
    assert model == "m1" and isinstance(caching.inner, OpenAICompatibleClient)
    assert caching.inner.base_url == "https://h.test/v1"


def test_unknown_provider_rejected(monkeypatch) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(SystemExit, match="unknown LLM_PROVIDER"):
        build_llm_client(CFG, CostGuard(5, 1.0))


def test_cache_key_includes_provider(tmp_path) -> None:
    fake = FakeTransport((200, {}, OK))
    a = CachingClient(client(fake), tmp_path, CostGuard(5, 1.0), {"provider": "groq"})
    b = CachingClient(client(fake), tmp_path, CostGuard(5, 1.0), {"provider": "xai"})
    m = [{"role": "user", "content": "x"}]
    assert a.key(m, "s") != b.key(m, "s")
