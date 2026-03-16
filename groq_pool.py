"""LLM provider pool: Groq (primary) with automatic Ollama fallback.

Tier 1: Groq cloud — multiple API keys with rotation on rate limits.
Tier 2: Local Ollama — automatic fallback when all Groq keys are exhausted.

All consumers call `create_completion()` instead of managing clients directly.
Model names are mapped automatically when falling back to Ollama.
"""

import os
import time
import threading
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# Model mapping: Groq model → Ollama local fallback
OLLAMA_MODEL_MAP = {
    "openai/gpt-oss-120b": "gemma2:27b",
    "openai/gpt-oss-20b": "gemma2:9b",
    "llama-3.3-70b-versatile": "gemma2:27b",
    "llama-3.1-8b-instant": "gemma2:9b",
}

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def _ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _create_ollama_client():
    """Create an OpenAI-compatible client pointing to local Ollama."""
    try:
        from openai import OpenAI
        return OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    except ImportError:
        return None


class LLMPool:
    """Manages Groq keys + Ollama fallback transparently."""

    def __init__(self):
        raw_keys = os.getenv("GROQ_API_KEYS", "")
        if raw_keys:
            self._keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        else:
            single = os.getenv("GROQ_API_KEY", "")
            self._keys = [single] if single else []

        self._groq_clients = [Groq(api_key=k) for k in self._keys] if self._keys else []
        self._current = 0
        self._lock = threading.Lock()
        self._cooldowns: dict[int, float] = {}

        self._ollama_client = None
        self._ollama_checked = False
        self._using_ollama = False

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    @property
    def provider(self) -> str:
        return "ollama" if self._using_ollama else "groq"

    def _next_available_groq(self) -> int | None:
        now = time.time()
        for offset in range(len(self._keys)):
            idx = (self._current + offset) % len(self._keys)
            if self._cooldowns.get(idx, 0) < now:
                return idx
        return None

    def _get_ollama(self):
        if not self._ollama_checked:
            self._ollama_checked = True
            if _ollama_available():
                self._ollama_client = _create_ollama_client()
                if self._ollama_client:
                    log.info("Ollama fallback available")
        return self._ollama_client

    def get_client(self) -> Groq:
        with self._lock:
            available = self._next_available_groq()
            if available is not None:
                self._current = available
                self._using_ollama = False
                return self._groq_clients[self._current]
            if self._groq_clients:
                return self._groq_clients[self._current]
            raise RuntimeError("No LLM providers available")

    def report_rate_limit(self, cooldown_seconds: float = 60.0) -> bool:
        with self._lock:
            self._cooldowns[self._current] = time.time() + cooldown_seconds
            next_idx = self._next_available_groq()
            if next_idx is not None:
                self._current = next_idx
                return True
            return False

    def status(self) -> list[dict]:
        now = time.time()
        entries = [
            {
                "index": i,
                "provider": "groq",
                "active": (not self._using_ollama) and i == self._current,
                "rate_limited": self._cooldowns.get(i, 0) > now,
                "cooldown_remaining": max(0, self._cooldowns.get(i, 0) - now),
                "key_hint": f"...{self._keys[i][-6:]}",
            }
            for i in range(len(self._keys))
        ]
        if self._ollama_client or _ollama_available():
            entries.append({
                "index": len(self._keys),
                "provider": "ollama",
                "active": self._using_ollama,
                "rate_limited": False,
                "cooldown_remaining": 0,
                "key_hint": "local",
            })
        return entries


pool = LLMPool()


def _is_rate_or_server_error(exc: Exception) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return any(x in msg for x in ("rate", "limit", "quota", "429", "503", "500")) or "RateLimit" in name or "InternalServer" in name


def create_completion(model: str, messages: list[dict], temperature: float = 0.2, **kwargs):
    """Create a chat completion, trying Groq first then falling back to Ollama.

    Handles rate limits, key rotation, and provider fallback transparently.
    Strips tool-calling params when falling back to Ollama.
    """
    # --- Tier 1: Try Groq ---
    last_error = None
    for attempt in range(3):
        try:
            client = pool.get_client()
            pool._using_ollama = False
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, **kwargs,
            )
            return resp
        except Exception as exc:
            last_error = exc
            if _is_rate_or_server_error(exc):
                pool.report_rate_limit()
                time.sleep(1 * (attempt + 1))
                continue
            break

    # --- Tier 2: Ollama fallback ---
    ollama = pool._get_ollama()
    if ollama:
        ollama_model = OLLAMA_MODEL_MAP.get(model, "gemma2:9b")
        fallback_kwargs = {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}
        fallback_kwargs.pop("stream", None)
        try:
            pool._using_ollama = True
            resp = ollama.chat.completions.create(
                model=ollama_model, messages=messages, temperature=temperature, **fallback_kwargs,
            )
            return resp
        except Exception as ollama_exc:
            log.warning(f"Ollama fallback failed: {ollama_exc}")

    raise last_error or RuntimeError("All LLM providers exhausted")


def create_completion_stream(model: str, messages: list[dict], temperature: float = 0.3, **kwargs):
    """Streaming completion — tries Groq, falls back to Ollama (non-streaming yield)."""
    # --- Tier 1: Try Groq streaming ---
    try:
        client = pool.get_client()
        pool._using_ollama = False
        stream = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, stream=True, **kwargs,
        )
        return stream
    except Exception as exc:
        if _is_rate_or_server_error(exc):
            pool.report_rate_limit()

    # --- Tier 2: Ollama fallback (non-streaming, yield as one chunk) ---
    ollama = pool._get_ollama()
    if ollama:
        ollama_model = OLLAMA_MODEL_MAP.get(model, "gemma2:9b")
        try:
            pool._using_ollama = True
            resp = ollama.chat.completions.create(
                model=ollama_model, messages=messages, temperature=temperature,
            )
            return resp
        except Exception:
            pass

    raise RuntimeError("All LLM providers exhausted for streaming")


# Backward-compat exports
def get_client() -> Groq:
    return pool.get_client()

def handle_rate_limit() -> bool:
    return pool.report_rate_limit()
