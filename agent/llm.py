"""OpenRouter chat-completions client with tool calling, a pinned provider, and per-call accounting.

Uses OpenRouter's OpenAI-compatible API (https://openrouter.ai/api/v1) through the `openai` SDK.
Every call is recorded (model, serving provider, tokens, cost, latency) so runs can be audited and costed.

Configuration (environment variables, or a git-ignored .env at the repo root):
  OPENROUTER_API_KEY    required
  KV_MODEL              model id            (default: anthropic/claude-sonnet-5.5)
  KV_PROVIDER           provider slug(s), comma-separated, tried in order with no fallback (default: anthropic)
  KV_MAX_TOKENS         max completion tokens per call (default: 8192)
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-5.5"
DEFAULT_PROVIDER = "anthropic"


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        env = ROOT / ".env"
        if env.exists():
            try:
                from dotenv import dotenv_values

                key = dotenv_values(env).get("OPENROUTER_API_KEY")
            except ImportError:  # minimal fallback parser
                for line in env.read_text().splitlines():
                    if line.strip().startswith("OPENROUTER_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (export it, or put it in .env; see .env.example)")
    return key


@dataclass
class CallRecord:
    t_start: float
    seconds: float
    model_requested: str
    model_served: str | None
    provider: str | None
    generation_id: str | None
    finish_reason: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    n_tool_calls: int = 0
    error: str | None = None


@dataclass
class Usage:
    calls: list[CallRecord] = field(default_factory=list)

    def totals(self) -> dict:
        c = self.calls
        return {
            "llm_calls": len(c),
            "prompt_tokens": sum(r.prompt_tokens for r in c),
            "completion_tokens": sum(r.completion_tokens for r in c),
            "cached_tokens": sum(r.cached_tokens for r in c),
            "reasoning_tokens": sum(r.reasoning_tokens for r in c),
            "cost_usd": round(sum(r.cost_usd for r in c), 6),
            "llm_seconds": round(sum(r.seconds for r in c), 3),
            "providers": sorted({r.provider for r in c if r.provider}),
            "models_served": sorted({r.model_served for r in c if r.model_served}),
        }


class OpenRouterClient:
    def __init__(self, model: str | None = None, provider: str | list[str] | None = None,
                 max_tokens: int | None = None, reasoning: dict | None = None, timeout: float = 300.0):
        from openai import OpenAI

        self.model = model or os.environ.get("KV_MODEL", DEFAULT_MODEL)
        prov = provider or os.environ.get("KV_PROVIDER", DEFAULT_PROVIDER)
        self.provider_order = prov.split(",") if isinstance(prov, str) else list(prov)
        self.max_tokens = int(max_tokens or os.environ.get("KV_MAX_TOKENS", 8192))
        self.reasoning = reasoning
        self._key = load_api_key()
        self._client = OpenAI(base_url=BASE_URL, api_key=self._key, timeout=timeout, max_retries=2,
                              default_headers={"X-Title": "kernel-verification (CS 329Z)"})
        self.usage = Usage()

    # Routing: pinned provider(s), no silent fallback to another provider, and only providers that support every
    # parameter we send (tools, tool_choice, ...).
    def provider_prefs(self) -> dict:
        return {"order": self.provider_order, "allow_fallbacks": False, "require_parameters": True}

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice=None) -> tuple[dict, CallRecord]:
        body = {"provider": self.provider_prefs()}
        if self.reasoning is not None:
            body["reasoning"] = self.reasoning
        kwargs = dict(model=self.model, messages=messages, max_tokens=self.max_tokens, extra_body=body)
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # recorded, then re-raised for the caller to handle
            rec = CallRecord(t0, time.time() - t0, self.model, None, None, None, None, error=f"{type(e).__name__}: {e}")
            self.usage.calls.append(rec)
            raise
        dt = time.time() - t0
        d = resp.model_dump()
        u = d.get("usage") or {}
        ptd = u.get("prompt_tokens_details") or {}
        ctd = u.get("completion_tokens_details") or {}
        choice = (d.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        rec = CallRecord(
            t_start=t0, seconds=round(dt, 3), model_requested=self.model, model_served=d.get("model"),
            provider=d.get("provider"), generation_id=d.get("id"), finish_reason=choice.get("finish_reason"),
            prompt_tokens=u.get("prompt_tokens") or 0, completion_tokens=u.get("completion_tokens") or 0,
            cached_tokens=ptd.get("cached_tokens") or 0, cache_write_tokens=ptd.get("cache_write_tokens") or 0,
            reasoning_tokens=ctd.get("reasoning_tokens") or 0, cost_usd=float(u.get("cost") or 0.0),
            n_tool_calls=len(msg.get("tool_calls") or []),
        )
        self.usage.calls.append(rec)
        return d, rec

    def generation_stats(self, generation_id: str, retries: int = 12, delay: float = 3.0) -> dict | None:
        """GET /generation?id=... (provider_name, native token counts, total_cost, latency). Stats land a moment
        after the call finishes (observed: >10 s), so retry for up to ~40 s."""
        url = f"{BASE_URL}/generation?" + urllib.parse.urlencode({"id": generation_id})
        last_err = None
        for _ in range(retries):
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self._key}"})
            try:
                with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as r:
                    data = json.loads(r.read()).get("data")
                if data:
                    return data
                last_err = "empty data"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(delay)
        self.last_generation_error = last_err
        return None


def _ssl_context() -> ssl.SSLContext:
    # The python.org macOS build does not read the system keychain, so plain urllib fails certificate verification.
    # truststore (an openai-SDK dependency) uses the OS trust store, as the SDK itself does.
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


def record_to_dict(rec: CallRecord) -> dict:
    return asdict(rec)
