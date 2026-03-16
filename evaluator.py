"""Conversation quality self-evaluation and self-improvement.

After a deal closes:
1. The evaluator LLM (70b) scores the conversation on 5 quality metrics.
2. The self-improvement LLM generates actionable suggestions for future calls.
3. Suggestions are persisted and injected into the next conversation prompt.

This creates a flywheel: each conversation makes the agent smarter.
"""

import json
from pathlib import Path

from prompts import EVALUATOR_PROMPT, SELF_IMPROVE_PROMPT
from groq_pool import create_completion
from config import MODEL_EVALUATOR, MODEL_FAST, format_conversation, parse_json, safe_response_text

IMPROVEMENTS_FILE = Path(__file__).parent / ".learned_improvements.json"
MAX_STORED_IMPROVEMENTS = 10


def _call_and_parse(messages: list[dict], model: str, temp: float = 0.1) -> dict | None:
    try:
        resp = create_completion(model=model, messages=messages, temperature=temp)
        raw = safe_response_text(resp)
        return parse_json(raw) if raw else None
    except Exception:
        return None


def evaluate_conversation(messages: list[dict]) -> dict | None:
    """Score a completed sales conversation on 5 quality metrics."""
    text = format_conversation(messages)
    if not text.strip():
        return None
    return _call_and_parse(
        [{"role": "system", "content": EVALUATOR_PROMPT},
         {"role": "user", "content": text}],
        MODEL_EVALUATOR, 0.1,
    )


def generate_improvements(eval_result: dict) -> list[str]:
    """Generate improvement suggestions from an evaluation and persist them."""
    context = json.dumps(eval_result, indent=2)
    result = _call_and_parse(
        [{"role": "system", "content": SELF_IMPROVE_PROMPT},
         {"role": "user", "content": context}],
        MODEL_FAST, 0.2,
    )
    improvements = result.get("improvements", []) if result else []

    if improvements:
        _store_improvements(improvements)

    return improvements


def _store_improvements(new_items: list[str]):
    """Persist improvements to disk, keeping the most recent ones."""
    existing = load_improvements()
    combined = existing + new_items
    trimmed = combined[-MAX_STORED_IMPROVEMENTS:]
    try:
        IMPROVEMENTS_FILE.write_text(json.dumps(trimmed, indent=2))
    except OSError:
        pass


def load_improvements() -> list[str]:
    """Load previously learned improvements from disk."""
    try:
        if IMPROVEMENTS_FILE.exists():
            data = json.loads(IMPROVEMENTS_FILE.read_text())
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return []
