"""Shared configuration constants and pure utilities for the AI sales agent."""

import json
import re

# ═══════════════════════════════════════════════════════════════════════════
# Model identifiers — single source of truth
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CONVERSATIONAL = "openai/gpt-oss-120b"
MODEL_FAST = "openai/gpt-oss-20b"
MODEL_EVALUATOR = MODEL_CONVERSATIONAL

# ═══════════════════════════════════════════════════════════════════════════
# Pipeline constants
# ═══════════════════════════════════════════════════════════════════════════

MAX_CONTEXT_MESSAGES = 30
MAX_TOOL_ROUNDS = 3
CONFIDENCE_THRESHOLD = 0.7

REQUIRED_FIELDS = ["industry", "company_size", "team", "use_case"]

PLAN_NAMES = {"starter": "Starter", "standard": "Standard", "pro": "Pro"}

CACHE_KEYS = ("cached_board", "cached_email", "cached_eval", "cached_improvements", "_email_cache_key")

# ═══════════════════════════════════════════════════════════════════════════
# Pure utilities (no external deps — safe to import from anywhere)
# ═══════════════════════════════════════════════════════════════════════════


def format_conversation(messages: list[dict]) -> str:
    """Build a role-labeled text transcript from a message list.

    Used by the extractor, scorer, evaluator, planner, and summarizer.
    """
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    )


def find_json_block(text: str) -> str | None:
    """Extract the first plausible JSON object from text (fenced or raw)."""
    if not isinstance(text, str):
        return None
    fence = re.search(r"```(?:json)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1].strip()
    return None


def parse_json(text: str) -> dict | None:
    """Best-effort JSON parsing with fallback to embedded-block extraction."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    candidate = find_json_block(text)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None
