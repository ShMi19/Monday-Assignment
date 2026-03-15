import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_llm(messages):
    """Send a chat completion request to the LLM with a clean message payload."""
    clean_messages = []
    for m in messages:
        clean_messages.append(
            {
                "role": str(m["role"]),
                "content": str(m["content"]),
            }
        )

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=clean_messages,
        temperature=0.2,
    )

    return completion.choices[0].message.content


def _find_json_block(text: str) -> str | None:
    """
    Try to locate a JSON object inside an arbitrary LLM response.

    Handles:
    - Leading / trailing text
    - JSON inside ``` ``` code fences
    """
    if not isinstance(text, str):
        return None

    # If response uses code fences, grab the content inside the first fence.
    fence_match = re.search(r"```(?:json)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    # Fallback: find the first {...} block in the text.
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start : brace_end + 1].strip()

    return None


def try_extract_json(text):
    """
    Best-effort JSON extraction from an LLM response.

    Returns:
        dict | None
    """
    if not text:
        return None

    # Fast path: plain JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Slower path: search for embedded JSON
    candidate = _find_json_block(text)
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except Exception:
        return None