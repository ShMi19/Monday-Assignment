import os
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_CONTEXT_MESSAGES = 30
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


def ask_llm(messages):
    """Send a chat completion request to the LLM with a trimmed, clean payload."""
    clean_messages = _trim_context(messages)

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=clean_messages,
                temperature=0.2,
            )
            return completion.choices[0].message.content
        except Exception as exc:
            is_rate_limit = "rate" in str(exc).lower() or "RateLimit" in type(exc).__name__
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            return (
                "I'm sorry, I'm having trouble connecting right now. "
                "Please try again in a moment. "
                f"(Error: {type(exc).__name__})"
            )


def _trim_context(messages):
    """Keep the system prompt + the most recent turns to stay within context limits."""
    clean = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]

    if len(clean) <= MAX_CONTEXT_MESSAGES + 1:
        return clean

    system = [clean[0]] if clean[0]["role"] == "system" else []
    recent = clean[-(MAX_CONTEXT_MESSAGES):]
    return system + recent


def _find_json_block(text: str) -> str | None:
    """Locate a JSON object inside an arbitrary LLM response."""
    if not isinstance(text, str):
        return None

    fence_match = re.search(
        r"```(?:json)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start : brace_end + 1].strip()

    return None


def try_extract_json(text):
    """Best-effort JSON extraction from an LLM response. Returns dict | None."""
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    candidate = _find_json_block(text)
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except Exception:
        return None


_BOARD_GEN_PROMPT = """You generate monday.com board definitions as JSON.

Given the prospect's qualification data, create a board with:
- 5 columns appropriate for their use case (first column is always type "name")
- 3 example items that are realistic for their specific industry and use case

Column types must be one of: name, status, person, date, numbers, text, timeline

Respond with ONLY valid JSON, no extra text:
{
  "columns": [
    {"id": "<snake_case>", "title": "<Display Name>", "type": "<column_type>"}
  ],
  "items": [
    {"<col_id>": "<value>", ...}
  ]
}
"""


def generate_board_with_llm(qualification: dict) -> dict | None:
    """Use the LLM to generate board columns and items tailored to the prospect."""
    user_msg = (
        f"Industry: {qualification.get('industry', 'unknown')}\n"
        f"Team: {qualification.get('team_using_monday', 'unknown')}\n"
        f"Use case: {qualification.get('use_case', 'unknown')}\n"
        f"Company size: {qualification.get('company_size', 'unknown')}"
    )

    messages = [
        {"role": "system", "content": _BOARD_GEN_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
            )
            raw = completion.choices[0].message.content
            return try_extract_json(raw)
        except Exception as exc:
            is_rate_limit = "rate" in str(exc).lower() or "RateLimit" in type(exc).__name__
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            return None
