"""Agentic multi-LLM sales agent with autonomous tool calling and streaming.

Per-turn pipeline (up to 8 LLM calls per message):
- Sentiment Analyzer (8b): Classifies prospect emotional state and intensity.
  \\ These three run in PARALLEL via ThreadPoolExecutor
- Extractor LLM (8b): Silently extracts structured data with confidence scores.  /
- Lead Scorer LLM (8b): Computes real-time 0-100 lead score from buying signals. /
- Strategy Planner (8b): Meta-reasoning — decides response approach before acting.
- Live Coach (8b, every 3 turns): Mid-conversation quality monitor that feeds
  tactical adjustments to the conversation LLM — like a sales manager listening in.
- Conversation LLM (70b, streaming): Token-by-token response with autonomous
  tool calling (lookup_pricing, compare_competitor, calculate_roi, suggest_automations).
- Context Summarizer (8b): Compresses older messages instead of dropping them.

Close sequence:
- Board Gen LLM (70b): Creates tailored monday.com boards.
- Follow-up Email LLM (70b): Generates onboarding email after close.
"""

import json
import re

from prompts import (
    build_conversation_prompt,
    EXTRACTOR_PROMPT,
    BOARD_GEN_PROMPT,
    FOLLOW_UP_EMAIL_PROMPT,
    LEAD_SCORE_PROMPT,
    SENTIMENT_PROMPT,
    STRATEGY_PLANNER_PROMPT,
    CONTEXT_SUMMARIZER_PROMPT,
    LIVE_COACH_PROMPT,
)
from tools import TOOL_SCHEMAS, execute_tool
from security import validate_tool_args
from evaluator import load_improvements
from groq_pool import create_completion, create_completion_stream
from config import (
    MODEL_CONVERSATIONAL, MODEL_FAST,
    MAX_CONTEXT_MESSAGES, MAX_TOOL_ROUNDS,
    CONFIDENCE_THRESHOLD, REQUIRED_FIELDS,
    format_conversation, parse_json, safe_response_text,
)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════


_context_summary_cache: dict = {"hash": "", "summary": ""}


def _trim_context(messages: list[dict], use_summarization: bool = True) -> list[dict]:
    """Keep the system prompt + most recent turns within context limits.

    When the context exceeds MAX_CONTEXT_MESSAGES, older messages are
    compressed into a summary paragraph via the 8b model instead of
    being dropped. The summary is cached to avoid redundant LLM calls.
    """
    clean = []
    for m in messages:
        entry = {"role": str(m["role"]), "content": str(m.get("content", ""))}
        if m.get("tool_calls"):
            entry["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            entry["tool_call_id"] = m["tool_call_id"]
            entry["role"] = "tool"
            entry["name"] = m.get("name", "")
        clean.append(entry)

    if len(clean) <= MAX_CONTEXT_MESSAGES + 1:
        return clean

    system = [clean[0]] if clean[0]["role"] == "system" else []
    non_system = clean[1:] if system else clean

    if not use_summarization or len(non_system) <= MAX_CONTEXT_MESSAGES:
        return system + non_system[-MAX_CONTEXT_MESSAGES:]

    cutoff = len(non_system) - MAX_CONTEXT_MESSAGES + 2
    older = non_system[:cutoff]
    recent = non_system[cutoff:]

    older_hash = str(len(older)) + (older[-1].get("content", "")[:50] if older else "")
    if _context_summary_cache["hash"] == older_hash and _context_summary_cache["summary"]:
        summary_text = _context_summary_cache["summary"]
    else:
        summary_text = summarize_context(older)
        _context_summary_cache["hash"] = older_hash
        _context_summary_cache["summary"] = summary_text

    summary_msg = {"role": "user", "content": f"[CONVERSATION SUMMARY — earlier messages compressed]\n{summary_text}"}
    bridge = {"role": "assistant", "content": "Thanks for the summary. I'll keep everything we discussed in mind as we continue."}

    return system + [summary_msg, bridge] + recent


def _call_llm(messages: list[dict], model: str, temperature: float = 0.2) -> str | None:
    """Synchronous LLM call with Groq → Ollama fallback."""
    clean = _trim_context(messages, use_summarization=False)
    try:
        resp = create_completion(model=model, messages=clean, temperature=temperature)
        return safe_response_text(resp)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Agentic conversation with tool calling
# ═══════════════════════════════════════════════════════════════════════════

def conversation_stream_with_tools(
    messages: list[dict],
    profile: dict,
    sentiment: dict | None = None,
    strategy: dict | None = None,
    live_coaching: str | None = None,
):
    """Stream the final response from the conversation LLM using Groq streaming.

    If the LLM decides to call tools, those are handled non-streaming first.
    The final natural-language response is then streamed token by token.

    Yields: (chunk_type, data) tuples:
        ("tool", {"tool": name, "args": {...}, "result": {...}})
        ("token", "text chunk")
        ("done", {"full_text": str, "tool_log": list})
    """
    summary = build_profile_summary(profile)
    low_fields = get_low_confidence_fields(profile)
    learned = load_improvements()
    system_prompt = build_conversation_prompt(
        summary, low_fields, learned,
        sentiment=sentiment, strategy=strategy,
        live_coaching=live_coaching,
    )

    llm_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m["role"] != "system":
            llm_messages.append({"role": m["role"], "content": m["content"]})

    tool_log = []
    used_tools = False

    conv_temp = 0.5

    for _round in range(MAX_TOOL_ROUNDS):
        try:
            resp = create_completion(
                model=MODEL_CONVERSATIONAL,
                messages=_trim_context(llm_messages),
                temperature=conv_temp,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
        except Exception as exc:
            error_msg = f"I'm sorry, I'm having trouble connecting right now. Please try again in a moment. (Error: {type(exc).__name__})"
            yield ("token", error_msg)
            yield ("done", {"full_text": error_msg, "tool_log": tool_log})
            return

        choice = resp.choices[0]

        has_tool_calls = (
            getattr(choice, "finish_reason", None) == "tool_calls"
            or (getattr(choice.message, "tool_calls", None) and len(choice.message.tool_calls) > 0)
        )

        if has_tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in choice.message.tool_calls
                ],
            }
            llm_messages.append(assistant_msg)

            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                validation = validate_tool_args(tc.function.name, args)
                if validation["valid"]:
                    args = validation["args"]
                    result = execute_tool(tc.function.name, args)
                else:
                    result = json.dumps({"error": validation["reason"]})

                entry = {"tool": tc.function.name, "args": args, "result": parse_json(result) or result}
                tool_log.append(entry)
                yield ("tool", entry)

                llm_messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "name": tc.function.name, "content": result,
                })
            used_tools = True
            continue

        if not used_tools:
            text = choice.message.content or ""
            for i in range(0, len(text), 4):
                yield ("token", text[i:i+4])
            yield ("done", {"full_text": text, "tool_log": tool_log})
            return

        break

    full_text = ""
    try:
        stream = create_completion_stream(
            model=MODEL_CONVERSATIONAL,
            messages=_trim_context(llm_messages),
            temperature=conv_temp,
        )
        if hasattr(stream, '__iter__') and not hasattr(stream, 'choices'):
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_text += delta.content
                    yield ("token", delta.content)
        else:
            full_text = stream.choices[0].message.content or ""
            for i in range(0, len(full_text), 4):
                yield ("token", full_text[i:i+4])
    except Exception:
        if not full_text:
            full_text = "I'm having trouble right now. Please try again."
            yield ("token", full_text)

    yield ("done", {"full_text": full_text, "tool_log": tool_log})


# ═══════════════════════════════════════════════════════════════════════════
# Profile helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_profile_summary(profile: dict) -> str:
    """Build a natural-language summary of everything we know about the prospect.

    Shows all extracted fields (not just a hardcoded list) so the LLM can
    autonomously reason about what's known vs. missing.
    """
    if not profile:
        return "Nothing known yet — this is the very start of the conversation."

    known = []
    unknown = []

    display_fields = {
        "industry": "Industry",
        "company_size": "Company size",
        "team": "Team",
        "use_case": "Use case / problem",
    }

    for field, label in display_fields.items():
        entry = profile.get(field, {})
        val = entry.get("value") if isinstance(entry, dict) else entry
        conf = entry.get("confidence", 0) if isinstance(entry, dict) else 0
        if val and conf >= CONFIDENCE_THRESHOLD:
            known.append(f"- {label}: {val}")
        elif val:
            known.append(f"- {label}: {val} (not fully confirmed)")
        else:
            unknown.append(label)

    plan = profile.get("preferred_plan")
    seats = profile.get("preferred_seats")
    if plan:
        known.append(f"- Preferred plan: {plan}")
    if seats:
        known.append(f"- Seats requested: {seats}")

    parts = []
    if known:
        parts.append("What we know:\n" + "\n".join(known))
    if unknown:
        parts.append("Still unclear: " + ", ".join(unknown))
    if not known and not unknown:
        return "Nothing known yet — this is the very start of the conversation."

    return "\n".join(parts)


def get_low_confidence_fields(profile: dict) -> list[str]:
    low = []
    for field in REQUIRED_FIELDS:
        entry = profile.get(field, {})
        if not isinstance(entry, dict):
            low.append(field)
            continue
        val = entry.get("value")
        conf = entry.get("confidence", 0)
        if not val or conf < CONFIDENCE_THRESHOLD:
            low.append(field)
    return low


def is_fully_qualified(profile: dict) -> bool:
    return len(get_low_confidence_fields(profile)) == 0


_PLAN_RE = re.compile(r"\b(starter|standard|pro)\s+(?:plan|pack|tier|package)\b", re.IGNORECASE)
_SEAT_RE = re.compile(r"\b(\d+)\s*(?:seats?|users?|engineers?|people|ppl|members?|devs?|developers?|employees?|licenses?|accounts?)\b", re.IGNORECASE)


def _fallback_extract_plan(messages: list[dict]) -> str | None:
    """Regex fallback: scan the last 6 messages for the most recent plan mention."""
    recent = [m for m in messages[-6:] if m["role"] in ("user", "assistant")]
    for msg in reversed(recent):
        match = _PLAN_RE.search(msg["content"])
        if match:
            return match.group(1).capitalize()
    return None


def _fallback_extract_seats(messages: list[dict]) -> int | None:
    """Regex fallback: scan the last 6 messages for the most recent seat count."""
    recent = [m for m in messages[-6:] if m["role"] in ("user", "assistant")]
    for msg in reversed(recent):
        match = _SEAT_RE.search(msg["content"])
        if match:
            val = int(match.group(1))
            if 1 <= val <= 10000:
                return val
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Extractor
# ═══════════════════════════════════════════════════════════════════════════

def extract_profile(messages: list[dict]) -> dict:
    """Run the extractor LLM (8b) to extract structured data with confidence."""
    llm_messages = [
        {"role": "system", "content": EXTRACTOR_PROMPT},
        {"role": "user", "content": format_conversation(messages)},
    ]
    raw = _call_llm(llm_messages, MODEL_FAST, temperature=0.0)
    result = parse_json(raw) if raw else None

    if not result:
        return {f: {"value": None, "confidence": 0.0} for f in REQUIRED_FIELDS} | {"phase": "contact"}

    for f in REQUIRED_FIELDS:
        if f not in result:
            result[f] = {"value": None, "confidence": 0.0}
        elif not isinstance(result[f], dict):
            result[f] = {"value": result[f], "confidence": 0.5}

    if "phase" not in result:
        result["phase"] = "contact"

    # Preserve plan/seat preferences (not confidence-scored, just direct values)
    if "preferred_plan" not in result:
        result["preferred_plan"] = None
    if "preferred_seats" not in result:
        result["preferred_seats"] = None

    # Fallback: if LLM didn't extract a plan, scan recent assistant messages for confirmation
    if not result.get("preferred_plan"):
        result["preferred_plan"] = _fallback_extract_plan(messages)
    if not result.get("preferred_seats"):
        result["preferred_seats"] = _fallback_extract_seats(messages)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Lead Scorer
# ═══════════════════════════════════════════════════════════════════════════

def score_lead(messages: list[dict], profile: dict) -> dict:
    """Compute a 0-100 lead score from the conversation and profile."""
    context = f"CONVERSATION:\n{format_conversation(messages)}\n\nEXTRACTED PROFILE:\n{json.dumps(profile, default=str)}"
    llm_messages = [
        {"role": "system", "content": LEAD_SCORE_PROMPT},
        {"role": "user", "content": context},
    ]
    raw = _call_llm(llm_messages, MODEL_FAST, temperature=0.0)
    result = parse_json(raw) if raw else None
    if not result or "score" not in result:
        return {"score": 0, "signals": [], "label": "Unknown"}
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Board Gen & Follow-up Email
# ═══════════════════════════════════════════════════════════════════════════

def generate_board_with_llm(qualification: dict) -> dict | None:
    user_msg = (
        f"Industry: {qualification.get('industry', 'unknown')}\n"
        f"Team: {qualification.get('team', qualification.get('team_using_monday', 'unknown'))}\n"
        f"Use case: {qualification.get('use_case', 'unknown')}\n"
        f"Company size: {qualification.get('company_size', 'unknown')}"
    )
    messages = [
        {"role": "system", "content": BOARD_GEN_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = _call_llm(messages, MODEL_CONVERSATIONAL, temperature=0.3)
    return parse_json(raw) if raw else None


def generate_follow_up_email(qualification: dict, board_name: str, plan: str, price_per_seat: str) -> str | None:
    context = (
        f"Team: {qualification.get('team', qualification.get('team_using_monday', 'unknown'))}\n"
        f"Industry: {qualification.get('industry', 'unknown')}\n"
        f"Use case: {qualification.get('use_case', 'unknown')}\n"
        f"Board name: {board_name}\n"
        f"Plan: {plan}\n"
        f"Price: {price_per_seat}\n"
    )
    messages = [
        {"role": "system", "content": FOLLOW_UP_EMAIL_PROMPT},
        {"role": "user", "content": context},
    ]
    return _call_llm(messages, MODEL_CONVERSATIONAL, temperature=0.4)


# ═══════════════════════════════════════════════════════════════════════════
# Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_sentiment(user_message: str, recent_context: str = "") -> dict:
    """Classify the emotional state of a prospect's message.

    Returns {"sentiment": "positive|neutral|skeptical|frustrated|excited",
             "intensity": 0.0-1.0, "reason": "brief explanation"}
    """
    content = f"RECENT CONTEXT:\n{recent_context}\n\nLATEST MESSAGE:\n{user_message}" if recent_context else user_message
    llm_messages = [
        {"role": "system", "content": SENTIMENT_PROMPT},
        {"role": "user", "content": content},
    ]
    raw = _call_llm(llm_messages, MODEL_FAST, temperature=0.0)
    result = parse_json(raw) if raw else None
    if not result or "sentiment" not in result:
        return {"sentiment": "neutral", "intensity": 0.5, "reason": "Unable to assess"}
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Agent Strategy Planner
# ═══════════════════════════════════════════════════════════════════════════

def plan_strategy(messages: list[dict], profile: dict, sentiment: dict, lead_score: dict) -> dict:
    """Generate a strategic plan before responding to the prospect.

    The planner sees the full state and outputs a brief strategy that guides
    the conversation LLM. Visible in the Agent Reasoning panel.

    Returns {"phase_assessment": str, "prospect_state": str,
             "strategy": str, "tools_to_consider": list, "priority": str}
    """
    recent = messages[-6:] if len(messages) > 6 else messages
    conv_text = format_conversation(recent)
    profile_text = build_profile_summary(profile)

    context = (
        f"CONVERSATION (last few turns):\n{conv_text}\n\n"
        f"PROSPECT PROFILE:\n{profile_text}\n\n"
        f"SENTIMENT: {sentiment.get('sentiment', 'neutral')} (intensity: {sentiment.get('intensity', 0.5)}) — {sentiment.get('reason', '')}\n\n"
        f"LEAD SCORE: {lead_score.get('score', 0)}/100 ({lead_score.get('label', 'Unknown')})\n"
        f"SIGNALS: {', '.join(lead_score.get('signals', []))}"
    )
    llm_messages = [
        {"role": "system", "content": STRATEGY_PLANNER_PROMPT},
        {"role": "user", "content": context},
    ]
    raw = _call_llm(llm_messages, MODEL_FAST, temperature=0.1)
    result = parse_json(raw) if raw else None
    if not result:
        return {"phase_assessment": "unknown", "prospect_state": "unknown",
                "strategy": "Continue the conversation naturally", "tools_to_consider": [], "priority": "rapport"}
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Live Coaching — mid-conversation quality monitor
# ═══════════════════════════════════════════════════════════════════════════

def live_coaching_check(messages: list[dict], profile: dict, turn_count: int) -> str | None:
    """Periodic quality check that acts like a sales manager listening in.

    Fires every 3 user turns (starting at turn 3). Returns a short tactical
    coaching note or None if no adjustment is needed.
    """
    if turn_count < 3 or turn_count % 3 != 0:
        return None

    recent = messages[-8:] if len(messages) > 8 else messages
    conv_text = format_conversation(recent)
    profile_text = build_profile_summary(profile)

    content = (
        f"CONVERSATION (recent):\n{conv_text}\n\n"
        f"PROSPECT PROFILE:\n{profile_text}"
    )
    llm_messages = [
        {"role": "system", "content": LIVE_COACH_PROMPT},
        {"role": "user", "content": content},
    ]
    raw = _call_llm(llm_messages, MODEL_FAST, temperature=0.1)
    result = parse_json(raw) if raw else None
    if result and result.get("coaching"):
        return result["coaching"]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Intelligent Context Compression
# ═══════════════════════════════════════════════════════════════════════════

def summarize_context(messages: list[dict]) -> str:
    """Compress older conversation messages into a summary paragraph.

    Used when context exceeds limits instead of hard truncation.
    Returns a summary of the older messages.
    """
    llm_messages = [
        {"role": "system", "content": CONTEXT_SUMMARIZER_PROMPT},
        {"role": "user", "content": format_conversation(messages)},
    ]
    raw = _call_llm(llm_messages, MODEL_FAST, temperature=0.0)
    return raw or "Previous conversation covered initial discovery and product discussion."
