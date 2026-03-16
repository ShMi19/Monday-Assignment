"""Defense-in-depth security layer for the AI sales agent.

Provides multiple layers of protection against prompt injection, XSS,
abuse, and data exfiltration — beyond the LLM-based guardrails.

Layers:
1. Input sanitization — length limits, control chars, unicode normalization
2. Regex pre-filter — catches known prompt injection patterns BEFORE the LLM
3. Output sanitization — strips dangerous HTML/JS from LLM output
4. Rate limiting — per-session message flood protection
5. Tool argument validation — prevents malicious tool call arguments
"""

import re
import time
import unicodedata

MAX_MESSAGE_LENGTH = 2000
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 15  # messages per window

# ═══════════════════════════════════════════════════════════════════════════
# 1. Input Sanitization
# ═══════════════════════════════════════════════════════════════════════════

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INVISIBLE_CHARS = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064"
    r"\ufeff\u00ad\u034f\u180e\u061c\u2066-\u2069\u202a-\u202e]"
)


def sanitize_input(text: str) -> str:
    """Clean user input before any processing."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = _INVISIBLE_CHARS.sub("", text)
    text = text.strip()
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH]
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 2. Regex Pre-filter (runs BEFORE LLM-based guardrail)
# ═══════════════════════════════════════════════════════════════════════════

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(\w+\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)",
        r"(disregard|forget|override|bypass)\s+(\w+\s+)?(instructions?|prompts?|rules?|programming|constraints?|guidelines?)",
        r"(you\s+are|act\s+as|pretend\s+to\s+be|roleplay\s+as|behave\s+as)\s+(now\s+)?(a\s+)?(?!interested|looking|a\s+(?:small|large|medium|startup))",
        r"(reveal|show|print|output|display|repeat|tell\s+me)\s+(your\s+)?(system\s+prompt|instructions?|rules?|internal|initial\s+prompt|hidden\s+prompt|secret)",
        r"(\bDAN\b|do\s+anything\s+now|jailbreak|evil\s+mode|developer\s+mode|god\s+mode|unrestricted\s+mode)",
        r"(system|assistant)\s*:\s*",
        r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|<<SYS>>|<</SYS>>",
        r"(new\s+instructions?|updated?\s+instructions?)\s*[:=]",
        r"base64[_\s]*decode|eval\s*\(|exec\s*\(|import\s+os|subprocess",
        r"(?:what|repeat|recite|echo)\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instructions?|rules?)",
        r"translate\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:to|into)",
        r"(?:from\s+now\s+on|starting\s+now|henceforth)\s+you\s+(?:will|must|should|are)",
    ]
]

_SAFE_OVERRIDES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"you\s+are\s+(now\s+)?(?:interested|looking|using|considering|thinking)",
        r"act\s+as\s+(?:a\s+)?(?:team|company|business|manager|lead)",
    ]
]


def regex_injection_check(text: str) -> dict:
    """Fast regex-based prompt injection detection. Runs before LLM guardrail.

    Returns {"safe": True} or {"safe": False, "pattern": "...", "reason": "..."}
    """
    for safe_re in _SAFE_OVERRIDES:
        if safe_re.search(text):
            return {"safe": True}

    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return {
                "safe": False,
                "pattern": match.group(0)[:50],
                "reason": "Message matched known prompt injection pattern.",
            }

    delimiter_count = sum(text.count(d) for d in ["```", "---", "===", "###", "***"])
    if delimiter_count >= 5:
        return {
            "safe": False,
            "pattern": "excessive_delimiters",
            "reason": "Message contains suspicious formatting that may attempt prompt boundary manipulation.",
        }

    return {"safe": True}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Output Sanitization (HTML/XSS protection)
# ═══════════════════════════════════════════════════════════════════════════

_DANGEROUS_TAGS = re.compile(
    r"<\s*/?\s*(script|iframe|object|embed|form|input|textarea|button|link|meta|base|applet)\b[^>]*>",
    re.IGNORECASE,
)
_EVENT_HANDLERS = re.compile(r"\bon\w+\s*=", re.IGNORECASE)
_JS_URLS = re.compile(r"(?:href|src|action)\s*=\s*[\"']?\s*javascript:", re.IGNORECASE)
_DATA_URLS = re.compile(r"(?:href|src)\s*=\s*[\"']?\s*data:\s*text/html", re.IGNORECASE)


def sanitize_output(text: str) -> str:
    """Strip dangerous HTML/JS from LLM output before rendering.

    Keeps safe HTML (b, i, em, strong, p, br, div, span, a, ul, ol, li, h1-h6, table, etc.)
    but removes anything that could execute code.
    """
    text = _DANGEROUS_TAGS.sub("", text)
    text = _EVENT_HANDLERS.sub("", text)
    text = _JS_URLS.sub('href="', text)
    text = _DATA_URLS.sub('href="', text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 4. Rate Limiting
# ═══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Simple in-memory sliding-window rate limiter per session."""

    def __init__(self, window: int = RATE_LIMIT_WINDOW, max_messages: int = RATE_LIMIT_MAX):
        self._window = window
        self._max = max_messages
        self._timestamps: list[float] = []

    def check(self) -> dict:
        """Check if the current request is within rate limits.

        Returns {"allowed": True} or {"allowed": False, "retry_after": <seconds>}
        """
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < self._window]

        if len(self._timestamps) >= self._max:
            oldest = self._timestamps[0]
            retry_after = int(self._window - (now - oldest)) + 1
            return {"allowed": False, "retry_after": retry_after}

        self._timestamps.append(now)
        return {"allowed": True}

    def reset(self):
        self._timestamps.clear()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Tool Argument Validation
# ═══════════════════════════════════════════════════════════════════════════

from config import PLAN_NAMES

VALID_PLANS = set(PLAN_NAMES.keys())

# Built from competitive_intel.json keys + known aliases
def _build_valid_competitors() -> set:
    try:
        import json, os
        path = os.path.join(os.path.dirname(__file__), "competitive_intel.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        keys = set(data.get("competitors", {}).keys())
        aliases = {"excel", "google sheets", "sheets", "spreadsheets", "spreadsheet"}
        return keys | aliases
    except Exception:
        return {"jira", "asana", "trello", "clickup", "notion", "smartsheet", "excel_sheets", "excel", "spreadsheets"}

VALID_COMPETITORS = _build_valid_competitors()
VALID_TEAMS = {"sales", "engineering", "marketing", "hr", "operations", "project management"}


def validate_tool_args(tool_name: str, args: dict) -> dict:
    """Validate and sanitize tool arguments before execution.

    Returns {"valid": True, "args": sanitized_args}
    or {"valid": False, "reason": "..."}
    """
    sanitized = {}

    if tool_name == "lookup_pricing":
        plan = str(args.get("plan", "")).strip().capitalize()
        if plan.lower() not in VALID_PLANS:
            return {"valid": False, "reason": f"Invalid plan: {plan}. Must be Starter, Standard, or Pro."}
        sanitized["plan"] = plan

    elif tool_name == "compare_competitor":
        comp = str(args.get("competitor", "")).strip().lower()
        if comp not in VALID_COMPETITORS:
            return {"valid": False, "reason": f"Unknown competitor: {comp}."}
        sanitized["competitor"] = comp

    elif tool_name == "calculate_roi":
        tool = str(args.get("current_tool", ""))[:100]
        cost = args.get("current_monthly_cost", 0)
        size = args.get("team_size", 10)
        replaced = args.get("num_tools_replaced", 2)
        try:
            cost = max(0, min(float(cost), 1_000_000))
            size = max(1, min(int(size), 100_000))
            replaced = max(1, min(int(replaced), 20))
        except (ValueError, TypeError):
            return {"valid": False, "reason": "Invalid numeric arguments for ROI calculation."}
        sanitized = {"current_tool": tool, "current_monthly_cost": cost, "team_size": size, "num_tools_replaced": replaced}

    elif tool_name == "suggest_automations":
        team = str(args.get("team_type", "")).strip().lower()
        if team not in VALID_TEAMS:
            team = "project management"
        sanitized["team_type"] = team

    else:
        return {"valid": False, "reason": f"Unknown tool: {tool_name}"}

    return {"valid": True, "args": sanitized}


