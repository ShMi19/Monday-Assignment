"""Input and output guardrails for the sales agent.

Uses the fast 8b model for low-latency safety checks:
- Input guardrail: Detects prompt injection and off-topic messages before
  the conversation LLM sees them.
- Output guardrail: Catches hallucinated features, unauthorized pricing promises,
  and competitor disparagement before showing the response.
"""

from prompts import INPUT_GUARD_PROMPT, OUTPUT_GUARD_PROMPT
from groq_pool import create_completion
from config import MODEL_FAST, parse_json, safe_response_text


def _call_guard(system_prompt: str, user_content: str) -> dict | None:
    """Run a guardrail check and parse JSON response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = create_completion(model=MODEL_FAST, messages=messages, temperature=0.0)
        text = safe_response_text(resp)
        return parse_json(text) if text else None
    except Exception:
        return None


SHORT_MESSAGE_THRESHOLD = 120


def check_input(user_message: str, recent_context: str = "") -> dict:
    """Check user input for prompt injection or off-topic content.

    Short messages (< 120 chars) that passed the regex pre-filter are
    auto-approved — small models misclassify brief contextual replies
    like "tech industry" or "engineering" when they lack conversation context.

    Returns:
        {"safe": True} or {"safe": False, "classification": "injection"|"off_topic", "reason": "..."}
    """
    if len(user_message) < SHORT_MESSAGE_THRESHOLD:
        return {"safe": True}

    content = user_message
    if recent_context:
        content = f"RECENT CONVERSATION CONTEXT:\n{recent_context}\n\nLATEST MESSAGE:\n{user_message}"

    result = _call_guard(INPUT_GUARD_PROMPT, content)

    if not result:
        return {"safe": True}

    classification = result.get("classification", "safe")
    if classification == "safe":
        return {"safe": True}

    return {
        "safe": False,
        "classification": classification,
        "reason": result.get("reason", "Message flagged by safety system."),
    }


def check_output(agent_response: str) -> dict:
    """Check agent output for hallucination, unauthorized promises, or competitor disparagement.

    Returns:
        {"safe": True} or {"safe": False, "issue": "..."}
    """
    result = _call_guard(OUTPUT_GUARD_PROMPT, agent_response)

    if not result:
        return {"safe": True}

    if result.get("safe", True):
        return {"safe": True}

    return {
        "safe": False,
        "issue": result.get("issue", "Response flagged by safety system."),
    }


def regenerate_safe_response(original_response: str, issue: str, messages: list[dict]) -> str | None:
    """Auto-regenerate a response that failed the output guardrail.

    Feeds the flagged issue back as a correction instruction so the LLM
    produces a clean response without the problematic content.
    """
    correction_prompt = (
        f"Your previous response was flagged by our safety system for: {issue}\n\n"
        f"Original response:\n{original_response}\n\n"
        f"Please rewrite your response to fix this issue. Rules:\n"
        f"- Remove any hallucinated features not in monday.com\n"
        f"- Only quote official pricing: Starter $9, Standard $12, Pro $19 per seat/month\n"
        f"- Focus on monday.com strengths, don't disparage competitors\n"
        f"- Keep the same helpful tone and information where accurate\n"
        f"Respond with ONLY the corrected message, no meta-commentary."
    )
    regen_messages = [
        {"role": "system", "content": "You are a monday.com sales agent. Rewrite the flagged response to be accurate and compliant."},
        {"role": "user", "content": correction_prompt},
    ]
    try:
        resp = create_completion(model=MODEL_FAST, messages=regen_messages, temperature=0.2)
        return safe_response_text(resp)
    except Exception:
        return None
