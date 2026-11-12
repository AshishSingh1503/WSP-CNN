"""LLM provider abstraction, request cache, cost guard, and the proposal agent (with optional inspection)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from torch import nn

from src.utils.config import Config, LLMConfig, resolve_path
from src.utils.logger import JsonlLogger
from src.wsp.dsl_doc import render_dsl_doc
from src.wsp.inspection import inspect_slice


class LLMError(RuntimeError):
    """Non-retryable provider failure."""


class LLMRefused(LLMError):
    """The provider declined the request (stop_reason == 'refusal')."""


class BudgetExceeded(RuntimeError):
    """The per-run call or USD cap was reached; the run must stop cleanly."""


@dataclass
class Usage:
    """Token, cost and provenance record for one model call (cached replays have cost_usd 0)."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    request_sha256: str = ""
    prompt_version: str = ""
    prompt_sha256: str = ""


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    stop_reason: str | None = None


def compute_cost(cfg: LLMConfig, usage: Usage) -> float:
    """USD cost from token counts and the configured per-million-token prices."""
    return (usage.input_tokens * cfg.price_input_per_mtok + usage.output_tokens * cfg.price_output_per_mtok
            + usage.cache_read_tokens * cfg.price_cache_read_per_mtok
            + usage.cache_write_tokens * cfg.price_cache_write_per_mtok) / 1e6


class LLMClient(ABC):
    """Provider interface: one call, returns text plus usage."""

    model: str = "unknown"

    @abstractmethod
    def complete(self, messages: list[dict[str, Any]], system: str,
                 response_schema: dict[str, Any] | None = None) -> LLMResponse:
        """Send ``messages`` (user/assistant turns) and return the reply."""


class FakeLLM(LLMClient):
    """Scripted client for tests and dry runs: each entry is a string or ``callable(messages, system)``."""

    model = "fake-llm"

    def __init__(self, script: list[Any], repeat_last: bool = False, cost_per_call: float = 0.0,
                 tokens: tuple[int, int] = (100, 50)) -> None:
        self.script, self.repeat_last, self.cost, self.tokens = list(script), repeat_last, cost_per_call, tokens
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, Any]], system: str,
                 response_schema: dict[str, Any] | None = None) -> LLMResponse:
        i = len(self.calls)
        self.calls.append({"messages": json.loads(json.dumps(messages)), "system": system})
        if i >= len(self.script) and not self.repeat_last:
            raise LLMError("FakeLLM script exhausted")
        item = self.script[min(i, len(self.script) - 1)]
        text = item(messages, system) if callable(item) else item
        return LLMResponse(text, Usage(self.model, self.tokens[0], self.tokens[1], cost_usd=self.cost), "end_turn")


class AnthropicClient(LLMClient):
    """Anthropic Messages API client with exponential backoff on rate limits, overload and 5xx."""

    def __init__(self, cfg: LLMConfig, model: str | None = None, api_key: str | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        import anthropic
        self.cfg, self.model, self._sleep = cfg, model or cfg.model, sleep
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)  # retries are ours, so they are counted

    def _request(self, messages: list[dict[str, Any]], system: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.cfg.max_tokens, "system": system,
                                  "messages": messages, "output_config": {"effort": self.cfg.effort},
                                  "cache_control": {"type": "ephemeral"}}
        if self.cfg.send_temperature:
            kwargs["temperature"] = self.cfg.temperature
        return kwargs

    def complete(self, messages: list[dict[str, Any]], system: str,
                 response_schema: dict[str, Any] | None = None) -> LLMResponse:
        a, last = self._anthropic, None
        for attempt in range(self.cfg.max_api_retries + 1):
            start = time.perf_counter()
            try:
                msg = self._client.messages.create(**self._request(messages, system))
                break
            except (a.RateLimitError, a.APIConnectionError, a.APITimeoutError) as err:
                last = err
            except a.APIStatusError as err:
                if err.status_code < 500:
                    raise LLMError(f"{type(err).__name__}: {err.message}") from err
                last = err
            if attempt < self.cfg.max_api_retries:
                self._sleep(min(self.cfg.backoff_base_seconds * 2 ** attempt + random.uniform(0, 1),
                                self.cfg.backoff_max_seconds))
        else:
            raise LLMError(f"gave up after {self.cfg.max_api_retries + 1} attempts: {last}")
        if msg.stop_reason == "refusal":
            raise LLMRefused(f"model refused: {getattr(msg, 'stop_details', None)}")
        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        usage = Usage(model=msg.model, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                      cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                      cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                      latency_s=time.perf_counter() - start)
        usage.cost_usd = compute_cost(self.cfg, usage)
        return LLMResponse(text, usage, msg.stop_reason)


# OpenAI-compatible chat-completions providers: name -> (base URL, API-key env var)
OPENAI_COMPATIBLE: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openai_compatible": ("", "LLM_API_KEY"),  # base URL from LLM_BASE_URL
}
Transport = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, dict[str, str], dict[str, Any]]]


def _urllib_transport(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
                      ) -> tuple[int, dict[str, str], dict[str, Any]]:
    """POST JSON with the standard library; returns (status, lower-cased headers, parsed JSON body)."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            data = json.loads(raw)
        except ValueError:
            data = {"error": {"message": raw[:300].decode("utf-8", "replace")}}
        return err.code, {k.lower(): v for k, v in err.headers.items()}, data
    except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
        return 599, {}, {"error": {"message": f"connection error: {err}"}}


class OpenAICompatibleClient(LLMClient):
    """Chat-completions client for Groq, xAI (Grok) or any OpenAI-compatible endpoint.

    Cost is reported as 0 (free tiers); the call cap still applies. Honors ``retry-after`` on 429.
    """

    def __init__(self, cfg: LLMConfig, base_url: str, api_key: str, model: str,
                 transport: Transport = _urllib_transport, sleep: Callable[[float], None] = time.sleep) -> None:
        self.cfg, self.model, self.base_url, self._key = cfg, model, base_url.rstrip("/"), api_key
        self._post, self._sleep = transport, sleep

    def _body(self, messages: list[dict[str, Any]], system: str) -> dict[str, Any]:
        return {"model": self.model, "max_tokens": self.cfg.max_tokens, "temperature": self.cfg.temperature,
                "messages": [{"role": "system", "content": system}, *messages]}

    def complete(self, messages: list[dict[str, Any]], system: str,
                 response_schema: dict[str, Any] | None = None) -> LLMResponse:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}",
                   "User-Agent": "wsp-cnn/1.0"}
        url, last = f"{self.base_url}/chat/completions", ""
        for attempt in range(self.cfg.max_api_retries + 1):
            start = time.perf_counter()
            status, resp_headers, data = self._post(url, headers, self._body(messages, system), 300.0)
            if status == 200:
                break
            message = str(data.get("error", {}).get("message", data))[:300]
            last = f"HTTP {status}: {message}"
            if status not in (429, 599) and status < 500:
                raise LLMError(last)  # 4xx other than rate limit (e.g. 413 request too large, 401) is not retryable
            if attempt < self.cfg.max_api_retries:
                wait = float(resp_headers.get("retry-after", 0) or 0) or self.cfg.backoff_base_seconds * 2 ** attempt
                self._sleep(min(wait + random.uniform(0, 1), self.cfg.backoff_max_seconds))
        else:
            raise LLMError(f"gave up after {self.cfg.max_api_retries + 1} attempts: {last}")
        choice = data["choices"][0]
        text = re.sub(r"<think>.*?</think>", "", choice["message"].get("content") or "", flags=re.S).strip()
        u = data.get("usage", {})
        usage = Usage(model=data.get("model", self.model), input_tokens=u.get("prompt_tokens", 0),
                      output_tokens=u.get("completion_tokens", 0), latency_s=time.perf_counter() - start)
        return LLMResponse(text, usage, "max_tokens" if choice.get("finish_reason") == "length" else choice.get("finish_reason"))


class CostGuard:
    """Hard per-run caps on real API calls and USD."""

    def __init__(self, max_calls: int, max_usd: float) -> None:
        self.max_calls, self.max_usd, self.calls, self.usd = max_calls, max_usd, 0, 0.0

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call cap reached ({self.calls}/{self.max_calls})")
        if self.usd >= self.max_usd:
            raise BudgetExceeded(f"cost cap reached (${self.usd:.2f} of ${self.max_usd:.2f})")

    def record(self, usage: Usage) -> None:
        self.calls += 1
        self.usd += usage.cost_usd


class CachingClient(LLMClient):
    """Wraps a client: identical requests are answered from results/llm_cache/{sha256}.json for free."""

    def __init__(self, inner: LLMClient, cache_dir: Path, guard: CostGuard, settings: dict[str, Any]) -> None:
        self.inner, self.cache_dir, self.guard, self.settings = inner, Path(cache_dir), guard, settings
        self.model = inner.model
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key(self, messages: list[dict[str, Any]], system: str) -> str:
        payload = {"model": self.inner.model, "system": system, "messages": messages, **self.settings}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def complete(self, messages: list[dict[str, Any]], system: str,
                 response_schema: dict[str, Any] | None = None) -> LLMResponse:
        key = self.key(messages, system)
        path = self.cache_dir / f"{key}.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            usage = Usage(**{**data["usage"], "cached": True, "cost_usd": 0.0, "request_sha256": key})
            return LLMResponse(data["text"], usage, data.get("stop_reason"))
        self.guard.check()
        response = self.inner.complete(messages, system, response_schema)
        self.guard.record(response.usage)
        response.usage.request_sha256 = key
        path.write_text(json.dumps({"request": {"system": system, "messages": messages, **self.settings},
                                    "text": response.text, "stop_reason": response.stop_reason,
                                    "usage": asdict(response.usage)}, indent=1), encoding="utf-8")
        return response


def build_llm_client(cfg: Config, guard: CostGuard) -> tuple[CachingClient, str]:
    """Client from .env (LLM_PROVIDER, LLM_MODEL, API key) wrapped with the request cache; returns (client, model).

    Providers: ``anthropic`` (ANTHROPIC_API_KEY), ``groq`` (GROQ_API_KEY), ``xai`` (XAI_API_KEY) and
    ``openai_compatible`` (LLM_BASE_URL + LLM_API_KEY).
    """
    from dotenv import load_dotenv
    load_dotenv(resolve_path(".env"))
    provider = os.environ.get("LLM_PROVIDER", cfg.llm.provider).lower()
    model = os.environ.get("LLM_MODEL", cfg.llm.model)
    settings = {"provider": provider, "max_tokens": cfg.llm.max_tokens}
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ERROR: ANTHROPIC_API_KEY is not set (see .env.example, or use --dry-run).")
        inner: LLMClient = AnthropicClient(cfg.llm, model=model)
        settings |= {"effort": cfg.llm.effort, "temperature": cfg.llm.temperature if cfg.llm.send_temperature else None}
    elif provider in OPENAI_COMPATIBLE:
        base_url, key_var = OPENAI_COMPATIBLE[provider]
        base_url = os.environ.get("LLM_BASE_URL", base_url)
        key = os.environ.get(key_var, "")
        if not key or not base_url:
            raise SystemExit(f"ERROR: {key_var} (and LLM_BASE_URL for 'openai_compatible') must be set in .env.")
        inner = OpenAICompatibleClient(cfg.llm, base_url, key, model)
        settings |= {"temperature": cfg.llm.temperature, "base_url": base_url}
    else:
        raise SystemExit(f"ERROR: unknown LLM_PROVIDER {provider!r}; use anthropic, groq, xai or openai_compatible")
    return CachingClient(inner, resolve_path(cfg.llm.cache_dir), guard, settings), model


# --- reply parsing ---

def parse_reply(text: str) -> tuple[Literal["proposals", "inspect", "error"], Any]:
    """Classify a model reply: a proposal list, an inspect request, or an error message."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").removeprefix("json").strip()
    try:
        value = json.loads(stripped)
    except ValueError:
        starts = [i for i in (stripped.find("["), stripped.find("{")) if i >= 0]
        if not starts:
            return "error", "no JSON value found in the reply"
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped[min(starts):])
        except ValueError as err:
            return "error", f"invalid JSON: {err}"
    if isinstance(value, dict) and "inspect" in value:
        return "inspect", value["inspect"]
    if isinstance(value, dict) and isinstance(value.get("proposals"), list):
        value = value["proposals"]
    if isinstance(value, dict) and "operations" in value:
        value = [value]
    if isinstance(value, list) and all(isinstance(v, dict) for v in value) and value:
        return "proposals", value
    return "error", "reply must be a JSON list of proposal objects or one inspect request"


@dataclass
class AgentResult:
    """What one propose() call produced."""

    proposals: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    inspections: list[dict[str, Any]] = field(default_factory=list)
    usages: list[Usage] = field(default_factory=list)
    extra_dropped: int = 0


class LLMAgent:
    """Turns a diagnostic packet into raw proposal dicts, with optional read-only inspection turns."""

    def __init__(self, client: LLMClient, cfg: Config, mode: Literal["pure", "hybrid"],
                 allowed_layers: tuple[str, ...], allowed_ops: tuple[str, ...],
                 log: JsonlLogger | None = None) -> None:
        self.client, self.cfg, self.mode, self.log = client, cfg, mode, log
        self.allowed_layers, self.allowed_ops = allowed_layers, allowed_ops
        path = resolve_path(cfg.llm.prompt_pure if mode == "pure" else cfg.llm.prompt_hybrid)
        raw = path.read_bytes()
        self.prompt_version, self.prompt_sha256 = path.stem, hashlib.sha256(raw).hexdigest()
        self.system = self._render(raw.decode("utf-8"))
        self.all_usages: list[Usage] = []

    def _render(self, template: str) -> str:
        c = self.cfg.llm
        inspect_fmt = (f'- or, to look at real weights first, ONE inspection request '
                       f'{{"inspect": {{"param": "features.10.weight", "slice": [[0,4],[0,8],[0,3],[0,3]]}}}} '
                       f'(one [start, stop) pair per dimension, at most {c.max_inspect_values} values; you may ask up to '
                       f'{c.n_inspect} times per iteration; values come back rounded to {c.inspect_sig_figs} significant '
                       f'figures).') if c.inspect_enabled else "- (weight inspection is disabled in this run)"
        text = (template.replace("{{MAX_CANDIDATES}}", str(self.cfg.budgets.candidates_per_iteration))
                .replace("{{INSPECT_FORMAT}}", inspect_fmt).replace("{{EDIT_DSL}}", render_dsl_doc(self.cfg)))
        return text + (f"\n\n# This experiment\nAllowed layers: {', '.join(self.allowed_layers)}.\n"
                       f"Allowed operations: {', '.join(self.allowed_ops)}.\n")

    def _call(self, messages: list[dict[str, Any]], result: AgentResult, iteration: int) -> str:
        response = self.client.complete(messages, self.system)
        u = response.usage
        u.prompt_version, u.prompt_sha256 = self.prompt_version, self.prompt_sha256
        result.usages.append(u)
        self.all_usages.append(u)
        if self.log:
            self.log.log("llm_call", iteration=iteration, **{k: v for k, v in asdict(u).items()})
        if response.stop_reason == "max_tokens":
            return response.text + "\n[reply truncated by max_tokens]"
        return response.text

    def propose(self, packet_json: str, model: nn.Module, iteration: int) -> AgentResult:
        """One iteration: optional inspection turns, then proposals. Never raises on bad LLM output."""
        c, result = self.cfg.llm, AgentResult()
        messages: list[dict[str, Any]] = [{"role": "user", "content":
            f"Iteration {iteration}. Diagnostic packet:\n{packet_json}\n\nReply with JSON only."}]
        inspections_used, repairs, reminded = 0, 0, False
        while True:
            try:
                text = self._call(messages, result, iteration)
            except LLMRefused as err:
                result.failed.append({"error": f"refused: {err}", "raw": ""})
                return result
            kind, value = parse_reply(text)
            if kind == "inspect":
                if not c.inspect_enabled or inspections_used >= c.n_inspect:
                    if reminded:
                        result.failed.append({"error": "repeated inspect request after the limit", "raw": text[:2000]})
                        return result
                    reminded = True
                    reply = {"ok": False, "error": "inspection is unavailable or its limit is used up; reply with proposals now"}
                else:
                    inspections_used += 1
                    reply = inspect_slice(model, value, self.allowed_layers, c.max_inspect_values, c.inspect_sig_figs)
                    result.inspections.append({"request": value, "ok": reply["ok"],
                                               "n_values": int(math.prod(reply.get("shape", [0])))})
                    if self.log:
                        self.log.log("inspection", iteration=iteration, request=value, ok=reply["ok"])
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": "Inspection result:\n" + json.dumps(reply, separators=(",", ":"))}]
                continue
            if kind == "proposals":
                k = self.cfg.budgets.candidates_per_iteration
                result.extra_dropped = max(0, len(value) - k)
                result.proposals = value[:k]
                return result
            if repairs < c.repair_retries:
                repairs += 1
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"Your reply was invalid: {value}. "
                                                         "Reply again with valid JSON only, in the required format."}]
                continue
            result.failed.append({"error": value, "raw": text[:2000]})
            return result
