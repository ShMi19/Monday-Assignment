"""monday.com AI Sales Agent — Streamlit application.

Agentic multi-LLM architecture with:
- Autonomous tool calling (the LLM decides when to look up pricing, compare
  competitors, calculate ROI, and suggest automations)
- Streaming responses with visible reasoning panel
- Input/output guardrails
- Confidence-based qualification with deferred close
- Real-time lead scoring
- Conversation self-evaluation + self-improvement loop
- Follow-up email generation
- Auto-demo mode for reviewers
"""

import re
import json
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from config import REQUIRED_FIELDS, CONFIDENCE_THRESHOLD, PLAN_NAMES, CACHE_KEYS
from agent import (
    conversation_stream_with_tools,
    extract_profile,
    build_profile_summary,
    get_low_confidence_fields,
    is_fully_qualified,
    generate_follow_up_email,
    score_lead,
    analyze_sentiment,
    plan_strategy,
    live_coaching_check,
)
from guardrails import check_input, check_output, regenerate_safe_response
from evaluator import evaluate_conversation, generate_improvements, load_improvements
from monday_mock import generate_monday_board
from stripe_mock import create_payment_link
from security import (
    sanitize_input,
    regex_injection_check,
    sanitize_output,
    RateLimiter,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

GREETING = (
    "Hey! I'm monday.com's AI assistant. "
    "I help teams find the right workspace setup.\n\n"
    "What brings you here today?"
)

PHASE_LABELS = ["Contact", "Demo", "Qualification", "Use-Case Setup", "Close & Payment"]

STATUS_COLORS = {
    "Working on it": "#FDAB3D", "Done": "#00CA72", "Stuck": "#DF2F4A",
    "Not started": "#C4C4C4", "In progress": "#FDAB3D", "Pending": "#FDAB3D",
    "High": "#DF2F4A", "Medium": "#FDAB3D", "Low": "#00CA72",
    "Critical": "#DF2F4A", "In Review": "#0086C0",
    "Open": "#0086C0", "To Do": "#C4C4C4", "Review": "#FDAB3D",
    "Resolved": "#00CA72",
}

TOOL_ICONS = {
    "lookup_pricing": "💰",
    "compare_competitor": "⚔️",
    "calculate_roi": "📊",
    "suggest_automations": "⚡",
}

MONDAY_CSS = """
<style>
    .stApp { font-family: 'Figtree', 'Poppins', sans-serif; }
    .phase-bar { display: flex; gap: 4px; margin-bottom: 1rem; }
    .phase-step { flex: 1; text-align: center; padding: 8px 4px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; }
    .phase-done { background: #00CA72; color: white; }
    .phase-active { background: #6161FF; color: white; }
    .phase-pending { background: #F5F6F8; color: #676879; }
    .board-table { width: 100%; border-collapse: separate; border-spacing: 0; border-radius: 8px; overflow: hidden; margin: 0.5rem 0; }
    .board-table th { background: #F5F6F8; color: #323338; font-weight: 600; padding: 10px 14px; text-align: left; font-size: 0.85rem; border-bottom: 2px solid #E6E9EF; }
    .board-table td { padding: 10px 14px; border-bottom: 1px solid #E6E9EF; font-size: 0.85rem; color: #323338; }
    .board-table tr:hover td { background: #F7F8FA; }
    .status-pill { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; color: white; text-align: center; min-width: 80px; }
    .metric-row { display: flex; gap: 12px; margin: 0.5rem 0; flex-wrap: wrap; }
    .metric-card { flex: 1; min-width: 100px; background: #F5F6F8; border-radius: 10px; padding: 16px; text-align: center; }
    .metric-card .label { font-size: 0.75rem; color: #676879; font-weight: 500; }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; color: #323338; margin-top: 4px; }
    .grade-badge { display: inline-block; padding: 6px 20px; border-radius: 8px; font-size: 1.8rem; font-weight: 800; color: white; }
    .grade-A { background: #00CA72; } .grade-B { background: #6161FF; } .grade-C { background: #FDAB3D; } .grade-D, .grade-F { background: #DF2F4A; }
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #6161FF; }
    .conf-bar { height: 6px; border-radius: 3px; background: #E6E9EF; margin-top: 2px; }
    .conf-fill { height: 100%; border-radius: 3px; }
    .lead-score { font-size: 2rem; font-weight: 800; text-align: center; }
    .lead-hot { color: #00CA72; } .lead-warm { color: #FDAB3D; } .lead-cold { color: #C4C4C4; }
    .tool-call { background: #F0EDFF; border-left: 3px solid #6161FF; padding: 8px 12px; border-radius: 0 6px 6px 0; margin: 4px 0; font-size: 0.82rem; }
    .tool-name { font-weight: 700; color: #6161FF; }
    [data-testid="stSidebarNav"] { display: none; }
</style>
"""

PLAN_FEATURES = {
    "Starter": {"color": "#00CA72", "views": ["Table"], "automations": "Basic", "integrations": "Basic"},
    "Standard": {"color": "#6161FF", "views": ["Table", "Timeline", "Gantt", "Calendar"], "automations": "250/month", "integrations": "250/month"},
    "Pro": {"color": "#F04095", "views": ["Table", "Timeline", "Gantt", "Calendar", "Chart", "Workload"], "automations": "25,000/month", "integrations": "25,000/month"},
}

PLAN_CHECKOUT_FEATURES = {
    "Starter": {"color": "#00CA72", "features": ["Up to 10 seats", "Basic boards and 200+ templates", "Unlimited docs", "8 column types", "iOS and Android apps"]},
    "Standard": {"color": "#6161FF", "features": ["Up to 50 seats", "Timeline & Gantt views", "250 automations / month", "250 integrations / month", "Guest access", "Calendar view"]},
    "Pro": {"color": "#F04095", "features": ["Unlimited seats", "Private boards & docs", "Chart view", "Time tracking", "Formula column", "25,000 automations / month", "25,000 integrations / month", "Dependency column"]},
}

# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="monday.com AI Sales Agent", page_icon="🤖", layout="wide")
st.markdown(MONDAY_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_company_size(raw: str) -> int | None:
    if not raw:
        return None
    numbers = re.findall(r"\d+", str(raw))
    if not numbers:
        return None
    nums = [int(n) for n in numbers if int(n) > 0]
    if not nums:
        return None
    if len(nums) == 2 and max(nums) / min(nums) < 5:
        return (nums[0] + nums[1]) // 2
    return max(nums)

_PLAN_SEAT_LIMITS = {
    "Starter": (1, 10),
    "Standard": (3, 50),
    "Pro": (3, 9999),
}


def _plan_for_seats(seats: int) -> str:
    """Return the cheapest plan that fits a given seat count."""
    if seats <= 10:
        return "Starter"
    if seats <= 50:
        return "Standard"
    return "Pro"


def _select_plan(company_size_raw: str, profile: dict = None) -> str:
    """Select plan based on user's explicit preference, falling back to company size.

    If the preferred plan can't support the seat count, auto-upgrade to the
    cheapest valid plan so we never display an impossible combination.
    """
    preferred_plan = None
    if profile:
        preferred = profile.get("preferred_plan")
        if isinstance(preferred, dict):
            preferred = preferred.get("value")
        if preferred and isinstance(preferred, str):
            normalized = PLAN_NAMES.get(preferred.strip().lower())
            if normalized:
                preferred_plan = normalized

    seats = _select_seats(company_size_raw, profile)

    if preferred_plan:
        _min, _max = _PLAN_SEAT_LIMITS.get(preferred_plan, (1, 9999))
        if seats is not None and seats > _max:
            return _plan_for_seats(seats)
        return preferred_plan

    size = _parse_company_size(company_size_raw)
    if size is None:
        return "Standard"
    return _plan_for_seats(size)


def _select_seats(company_size_raw: str, profile: dict = None) -> int | None:
    """Get seat count: user's preferred seats first, then company size."""
    if profile:
        preferred = profile.get("preferred_seats")
        if isinstance(preferred, dict):
            preferred = preferred.get("value")
        if preferred is not None:
            try:
                val = int(float(str(preferred)))
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
    return _parse_company_size(company_size_raw)

def _phase_from_profile(profile: dict, ready: bool, closed: bool) -> int:
    if closed: return 4
    if ready: return 3
    phase_str = profile.get("phase", "contact")
    return {"contact": 0, "demo": 1, "qualification": 2, "ready": 3}.get(phase_str, 0)

def _render_progress(phase: int):
    steps_html = ""
    for i, label in enumerate(PHASE_LABELS):
        if i < phase: cls, icon = "phase-done", "✓ "
        elif i == phase: cls, icon = "phase-active", "→ "
        else: cls, icon = "phase-pending", ""
        steps_html += f'<div class="phase-step {cls}">{icon}{label}</div>'
    st.markdown(f'<div class="phase-bar">{steps_html}</div>', unsafe_allow_html=True)

def _board_col_meta(columns: list) -> tuple[list[str], list[str]]:
    """Extract parallel lists of column IDs and titles from board columns."""
    col_ids = [c.get("id", c.get("title", "").lower().replace(" ", "_")) for c in columns]
    col_titles = [c.get("title", c.get("id", "")) for c in columns]
    return col_ids, col_titles


def _resolve_cell_value(item: dict, cid: str, ctitle: str) -> str:
    """Look up a cell value using all known key variants."""
    value = item.get(cid) or item.get(ctitle) or item.get(ctitle.lower(), "")
    if not value:
        ctitle_norm = ctitle.lower()
        for k, v in item.items():
            if k.lower().replace("_", " ") == ctitle_norm:
                value = v
                break
    if not value and "name" in cid:
        value = item.get("name", "")
    return value or ""


def _render_board_html(columns: list, items: list):
    col_ids, col_titles = _board_col_meta(columns)
    header = "".join(f"<th>{t}</th>" for t in col_titles)
    rows_html = ""
    for item in items:
        cells = ""
        for cid, ctitle in zip(col_ids, col_titles):
            value = _resolve_cell_value(item, cid, ctitle)
            col_type = next((c.get("type") for c in columns if c.get("id") == cid), "text")
            if col_type == "status" and value in STATUS_COLORS:
                color = STATUS_COLORS[value]
                cells += f'<td><span class="status-pill" style="background:{color}">{value}</span></td>'
            else:
                cells += f"<td>{value}</td>"
        rows_html += f"<tr>{cells}</tr>"
    st.markdown(f'<table class="board-table"><thead><tr>{header}</tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)

def _render_metrics_row(metrics: list[tuple[str, str]]):
    cards = ""
    for label, value in metrics:
        cards += f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>'
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)

def _flat_qualification(profile: dict) -> dict:
    flat = {}
    for field in REQUIRED_FIELDS:
        entry = profile.get(field, {})
        flat[field] = entry.get("value", "") if isinstance(entry, dict) else (entry or "")
    flat["team_using_monday"] = flat.get("team", "")
    return flat

def _render_tool_calls(tool_log: list[dict]):
    """Render the agent's autonomous tool calls as a visible reasoning panel."""
    if not tool_log:
        return
    with st.expander("🧠 Agent Reasoning — Tool Calls", expanded=True):
        for call in tool_log:
            icon = TOOL_ICONS.get(call["tool"], "🔧")
            args_str = ", ".join(f"{k}={v}" for k, v in call.get("args", {}).items())
            st.markdown(
                f'<div class="tool-call">{icon} <span class="tool-name">{call["tool"]}</span>({args_str})</div>',
                unsafe_allow_html=True,
            )

def _get_current_payment() -> dict:
    """Compute fresh payment from current profile (no caching)."""
    profile = st.session_state.get("profile", {})
    data = _flat_qualification(profile)
    plan = _select_plan(data.get("company_size", ""), profile)
    seats = _select_seats(data.get("company_size", ""), profile)
    return create_payment_link(plan, seats)


def _render_board_view():
    """Render the full-page monday.com board view with inline styles."""
    board = st.session_state.get("cached_board")
    payment = _get_current_payment()
    plan_name = payment.get("plan", "Standard")

    if not board:
        st.warning("No board data found.")
        return

    plan_info = PLAN_FEATURES.get(plan_name, PLAN_FEATURES["Standard"])
    columns = board.get("columns", [])
    items = board.get("items", [])
    board_name = board.get("board_name", "Workspace")

    col_ids, col_titles = _board_col_meta(columns)

    views_html = ""
    for i, v in enumerate(plan_info["views"]):
        bg = "#6161FF" if i == 0 else "#3C3F5C"
        fg = "white" if i == 0 else "#A1A1B5"
        views_html += f'<span style="padding:6px 16px;border-radius:6px;font-size:0.8rem;font-weight:600;background:{bg};color:{fg}">{v}</span>'

    done_count = sum(1 for item in items for v in item.values() if str(v).lower() == "done")
    in_prog = sum(1 for item in items for v in item.values() if str(v).lower() in ("working on it", "in progress"))

    header_html = "".join(
        f'<th style="background:#3C3F5C;color:#A1A1B5;padding:10px 16px;text-align:left;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">{t}</th>'
        for t in col_titles
    )
    rows_html = ""
    td_style = "padding:12px 16px;border-bottom:1px solid #3C3F5C;color:#E0E0E8;font-size:0.85rem"
    for item in items:
        cells = ""
        for cid, ctitle in zip(col_ids, col_titles):
            value = _resolve_cell_value(item, cid, ctitle)
            col_type = next((c.get("type") for c in columns if c.get("id") == cid), "text")
            if col_type == "status" and value in STATUS_COLORS:
                color = STATUS_COLORS[value]
                cells += f'<td style="{td_style}"><span style="display:inline-block;padding:4px 14px;border-radius:12px;font-size:0.78rem;font-weight:600;color:white;min-width:90px;text-align:center;background:{color}">{value}</span></td>'
            else:
                cells += f'<td style="{td_style}">{value}</td>'
        rows_html += f"<tr>{cells}</tr>"

    stat = lambda n, l: f'<div style="background:#3C3F5C;border-radius:8px;padding:12px 20px;flex:1;text-align:center"><div style="font-size:1.6rem;font-weight:800;color:white">{n}</div><div style="font-size:0.75rem;color:#A1A1B5;margin-top:2px">{l}</div></div>'

    st.markdown(f"""
    <div style="background:#292F4C;border-radius:12px;padding:0;overflow:hidden;font-family:Figtree,Poppins,sans-serif">
        <div style="background:#30324E;padding:20px 30px;border-radius:12px 12px 0 0">
            <h2 style="color:white;margin:0;font-size:1.4rem">📋 {board_name}
                <span style="display:inline-block;padding:4px 12px;border-radius:6px;font-size:0.75rem;font-weight:700;color:white;margin-left:12px;background:{plan_info['color']}">{plan_name} Plan</span>
            </h2>
            <div style="color:#A1A1B5;font-size:0.85rem;margin-top:4px">AI-generated workspace • {len(items)} items • {len(columns)} columns • Automations: {plan_info['automations']}</div>
            <div style="display:flex;gap:8px;margin-top:12px">{views_html}</div>
        </div>
        <div style="display:flex;gap:16px;margin:16px 30px">
            {stat(len(items), "Total Items")}
            {stat(done_count, "Done")}
            {stat(in_prog, "In Progress")}
            {stat(len(columns), "Columns")}
        </div>
        <div style="background:#30324E;border-radius:0 0 12px 12px;padding:0;overflow:hidden">
            <table style="width:100%;border-collapse:collapse">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_checkout_view():
    """Render the Stripe-like checkout view with inline styles."""
    payment = _get_current_payment()

    plan = payment.get("plan", "Standard")
    seats = payment.get("seats", 3)
    details = PLAN_CHECKOUT_FEATURES.get(plan, PLAN_CHECKOUT_FEATURES["Standard"])
    price_per_seat = int(payment.get("price_per_seat", "$12").replace("$", "").split("/")[0])
    subtotal = price_per_seat * seats
    session_id = payment.get("session_id", "cs_demo_unknown")

    features_html = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;font-size:0.85rem;color:#3C4257"><span style="color:#00CA72;font-weight:bold">✓</span> {f}</div>'
        for f in details["features"]
    )

    inp = "width:100%;padding:10px 12px;border:1px solid #E6EBF1;border-radius:6px;font-size:0.9rem;margin-bottom:16px;background:#F9FAFB;color:#8898AA;box-sizing:border-box"
    lbl = "font-size:0.8rem;font-weight:600;color:#3C4257;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px"

    st.markdown(f"""
    <div style="max-width:480px;margin:2rem auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif">
        <div style="text-align:center;margin-bottom:2rem">
            <div style="font-size:1.5rem;font-weight:800;color:#6161FF">monday</div>
            <div style="color:#8898AA;font-size:0.75rem">Powered by <strong>Stripe</strong> (Demo)</div>
        </div>
        <div style="background:white;border-radius:12px;padding:28px;box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:20px">
            <div style="font-size:1.3rem;font-weight:700;color:#1A1F36;display:flex;align-items:center;gap:10px">
                monday.com {plan}
                <span style="padding:3px 10px;border-radius:4px;font-size:0.7rem;font-weight:700;color:white;background:{details['color']}">{plan.upper()}</span>
            </div>
            <div style="margin:16px 0">{features_html}</div>
            <div style="margin:20px 0;border-top:1px solid #E6EBF1;padding-top:16px">
                <div style="display:flex;justify-content:space-between;padding:8px 0;font-size:0.9rem;color:#3C4257">
                    <span>{plan} plan × {seats} seats</span>
                    <span>${price_per_seat} × {seats}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;font-size:0.9rem;color:#3C4257">
                    <span>Billing period</span>
                    <span>Monthly</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-top:2px solid #1A1F36;margin-top:12px;padding-top:12px;font-weight:700;font-size:1.1rem;color:#1A1F36">
                    <span>Total due today</span>
                    <span>${subtotal:,}/month</span>
                </div>
            </div>
        </div>
        <div style="background:white;border-radius:12px;padding:28px;box-shadow:0 2px 12px rgba(0,0,0,0.08)">
            <div style="{lbl}">Email</div>
            <div style="{inp}">prospect@company.com</div>
            <div style="{lbl}">Card information</div>
            <div style="{inp}">4242 4242 4242 4242</div>
            <div style="display:flex;gap:12px">
                <div style="flex:1"><div style="{inp}">12 / 28</div></div>
                <div style="flex:1"><div style="{inp}">123</div></div>
            </div>
            <div style="{lbl}">Name on card</div>
            <div style="{inp}">Demo User</div>
            <div style="display:block;width:100%;padding:14px;background:#6161FF;color:white;border:none;border-radius:8px;font-size:1rem;font-weight:700;text-align:center;margin-top:16px;box-sizing:border-box">Pay ${subtotal:,}/month</div>
            <div style="text-align:center;color:#8898AA;font-size:0.75rem;margin-top:12px">🔒 This is a demo checkout. No real payment will be processed.<br>Session: {session_id}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_results(profile: dict):
    data = _flat_qualification(profile)
    plan = _select_plan(data.get("company_size", ""), profile)
    seats = _select_seats(data.get("company_size", ""), profile)

    st.markdown("### ✓ Qualification Summary")
    _render_metrics_row([
        ("Industry", data.get("industry", "—")),
        ("Company Size", data.get("company_size", "—")),
        ("Team", data.get("team", "—")),
        ("Use Case", (data.get("use_case", "—") or "—")[:35]),
    ])

    st.markdown("### ✓ Your Tailored Workspace")
    if "cached_board" not in st.session_state:
        with st.spinner("AI is generating your personalized workspace…"):
            st.session_state.cached_board = generate_monday_board(data)
    board = st.session_state.cached_board
    st.markdown(f"**{board['board_name']}**")
    if board.get("generated_by") == "ai":
        st.caption("Board columns and items were AI-generated from your specific use case.")
    _render_board_html(board["columns"], board["items"])
    if st.button("📋 Open board in monday.com →", key="btn_board"):
        st.session_state.current_view = "board"
        st.rerun()

    payment = create_payment_link(plan, seats)
    st.markdown("### ✓ Suggested Plan")
    _render_metrics_row([
        ("Plan", plan), ("Per Seat", payment["price_per_seat"]),
        ("Seats", str(payment["seats"])), ("Total", payment["subtotal"]),
    ])
    if st.button("💳 Proceed to payment →", key="btn_checkout"):
        st.session_state.current_view = "checkout"
        st.rerun()
    st.caption(f"Session: `{payment['session_id']}`")

    st.markdown("### ✉ Follow-Up Email")
    email_cache_key = f"{plan}_{seats}"
    if st.session_state.get("_email_cache_key") != email_cache_key:
        st.session_state.pop("cached_email", None)
        st.session_state["_email_cache_key"] = email_cache_key
    if "cached_email" not in st.session_state:
        with st.spinner("Generating personalized onboarding email…"):
            st.session_state.cached_email = generate_follow_up_email(data, board["board_name"], plan, payment["price_per_seat"])
    email_text = st.session_state.cached_email
    if email_text:
        with st.expander("Copy this email to share with your team", expanded=True):
            st.code(email_text, language=None)
    else:
        st.caption("Email generation unavailable — try again later.")

    st.markdown("### 📊 Conversation Analytics")
    if "cached_eval" not in st.session_state:
        with st.spinner("AI is evaluating the sales conversation quality…"):
            st.session_state.cached_eval = evaluate_conversation(st.session_state.messages)
    eval_result = st.session_state.cached_eval
    if eval_result:
        st.session_state.eval_result = eval_result
        grade = eval_result.get("overall_grade", "?")
        grade_cls = f"grade-{grade}" if grade in "ABCDF" else "grade-C"
        col_grade, col_summary = st.columns([1, 3])
        with col_grade:
            st.markdown(f'<div class="grade-badge {grade_cls}">{grade}</div>', unsafe_allow_html=True)
        with col_summary:
            st.markdown(f"*{eval_result.get('summary', '')}*")
        _render_metrics_row([
            ("Turns to Qualify", str(eval_result.get("turns_to_qualify", "—"))),
            ("Personalization", f"{eval_result.get('personalization_score', '—')}/10"),
            ("Assumption Violations", str(eval_result.get("assumption_violations", "—"))),
            ("Demo Quality", f"{eval_result.get('demo_quality', '—')}/10"),
        ])

        # Self-improvement: generate and store suggestions for future conversations
        if "cached_improvements" not in st.session_state:
            with st.spinner("Generating self-improvement suggestions…"):
                st.session_state.cached_improvements = generate_improvements(eval_result)
        improvements = st.session_state.cached_improvements
        if improvements:
            with st.expander("🧬 Self-Improvement — Learned for Next Conversation"):
                for imp in improvements:
                    st.markdown(f"- {imp}")
    else:
        st.caption("Analytics unavailable — try again later.")

    st.divider()
    team = data.get("team", "your team")
    st.markdown(
        f"You're all set! I've created a **{plan}** workspace for "
        f"**{team}** to get started right away. "
        f"Click the payment link above to activate your account. "
        f"Welcome to monday.com! 🎉"
    )

# ═══════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════

def _reset():
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.profile = {}
    st.session_state.ready_to_close = False
    st.session_state.closed = False
    st.session_state.turn_count = 0
    st.session_state.eval_result = None
    st.session_state.guardrail_log = []
    st.session_state.tool_log = []
    st.session_state.lead_score = None
    st.session_state.auto_demo_running = False
    st.session_state.current_view = "chat"
    st.session_state.sentiment_history = []
    st.session_state.strategy_log = []
    st.session_state.rate_limiter = RateLimiter()
    for key in CACHE_KEYS:
        st.session_state.pop(key, None)

if "messages" not in st.session_state:
    _reset()

# ═══════════════════════════════════════════════════════════════════════════
# Auto-demo mode
# ═══════════════════════════════════════════════════════════════════════════

AUTO_DEMO_MESSAGES = [
    "We're a fintech company with about 50 people. Our engineering team needs help with sprint planning and bug tracking.",
    "That sounds great! We're currently using Jira and paying about $500 a month. Can you tell me more about how monday.com compares?",
    "Interesting! What about automations? How would those help our engineering workflow specifically?",
    "This sounds promising. I think we're ready to try it out!",
]

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://dapulse-res.cloudinary.com/image/upload/f_auto,q_auto/remote_mondaycom_static/img/monday-logo-x2.png", width=160)
    st.markdown("---")

    col_reset, col_demo = st.columns(2)
    with col_reset:
        if st.button("🔄 New Prospect"):
            _reset()
            st.rerun()
    with col_demo:
        if st.button("🎬 Auto Demo"):
            _reset()
            st.session_state.auto_demo_step = 0
            st.session_state.auto_demo_running = True
            st.rerun()

    # Lead Score
    lead_data = st.session_state.get("lead_score")
    if lead_data and isinstance(lead_data, dict):
        score = lead_data.get("score", 0)
        label = lead_data.get("label", "Unknown")
        cls = "lead-hot" if label == "Hot" else ("lead-warm" if label == "Warm" else "lead-cold")
        st.markdown("### Lead Score")
        st.markdown(f'<div class="lead-score {cls}">{score}/100</div>', unsafe_allow_html=True)
        st.caption(f"**{label}** lead")
        signals = lead_data.get("signals", [])
        if signals:
            for sig in signals:
                st.markdown(f"- {sig}")
        st.markdown("---")

    # Sentiment tracking
    sentiment_history = st.session_state.get("sentiment_history", [])
    if sentiment_history:
        latest = sentiment_history[-1]
        sent = latest.get("sentiment", "neutral")
        intensity = latest.get("intensity", 0.5)
        sent_colors = {"excited": "#00CA72", "positive": "#6161FF", "neutral": "#C4C4C4", "skeptical": "#FDAB3D", "frustrated": "#DF2F4A", "objecting": "#DF2F4A"}
        sent_icons = {"excited": "🔥", "positive": "😊", "neutral": "😐", "skeptical": "🤔", "frustrated": "😤", "objecting": "🛑"}
        color = sent_colors.get(sent, "#C4C4C4")
        icon = sent_icons.get(sent, "😐")
        st.markdown("### Prospect Mood")
        st.markdown(f'<div style="text-align:center;font-size:2rem">{icon}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;font-weight:700;color:{color}">{sent.title()} ({intensity:.0%})</div>', unsafe_allow_html=True)
        st.caption(latest.get("reason", ""))
        if len(sentiment_history) > 1:
            trend = " → ".join(sent_icons.get(s.get("sentiment", "neutral"), "😐") for s in sentiment_history[-5:])
            st.caption(f"Trend: {trend}")
        st.markdown("---")

    # Strategy log
    strategy_log = st.session_state.get("strategy_log", [])
    if strategy_log:
        latest_strategy = strategy_log[-1]
        with st.expander("🧠 Agent Strategy (latest)", expanded=False):
            st.markdown(f"**Priority:** {latest_strategy.get('priority', '?')}")
            what_known = latest_strategy.get("what_i_know", latest_strategy.get("phase_assessment", "?"))
            what_needed = latest_strategy.get("what_i_still_need", "")
            next_move = latest_strategy.get("next_move", latest_strategy.get("strategy", "?"))
            st.markdown(f"**What I know:** {what_known}")
            if what_needed:
                st.markdown(f"**What I still need:** {what_needed}")
            st.markdown(f"**Prospect state:** {latest_strategy.get('prospect_state', '?')}")
            st.markdown(f"**Next move:** {next_move}")
            tools = latest_strategy.get("tools_to_consider", [])
            if tools:
                st.markdown(f"**Tools to consider:** {', '.join(tools)}")
        st.markdown("---")

    coaching_log = st.session_state.get("coaching_log", [])
    if coaching_log:
        with st.expander("🎙️ Live Coach (latest)", expanded=False):
            st.markdown(f"_{coaching_log[-1]}_")
            if len(coaching_log) > 1:
                st.caption(f"{len(coaching_log)} coaching notes this session")
        st.markdown("---")

    # Prospect profile
    st.markdown("### Prospect Profile")
    profile = st.session_state.get("profile", {})
    if profile:
        for field in REQUIRED_FIELDS:
            entry = profile.get(field, {})
            val = entry.get("value") if isinstance(entry, dict) else entry
            conf = entry.get("confidence", 0) if isinstance(entry, dict) else 0
            if val:
                conf_pct = int(conf * 100)
                bar_color = "#00CA72" if conf >= CONFIDENCE_THRESHOLD else ("#FDAB3D" if conf >= 0.4 else "#C4C4C4")
                st.markdown(f"**{field.replace('_', ' ').title()}:** {val}")
                st.markdown(f'<div class="conf-bar"><div class="conf-fill" style="width:{conf_pct}%;background:{bar_color}"></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f"**{field.replace('_', ' ').title()}:** *unknown*")
        if st.session_state.closed: st.success("✓ Deal closed")
        elif st.session_state.ready_to_close: st.info("Ready — waiting for prospect")
    else:
        st.caption("No data collected yet.")
    st.markdown("---")

    # Analytics in sidebar
    eval_result = st.session_state.get("eval_result")
    if eval_result:
        st.markdown("### Conversation Analytics")
        st.markdown(f"**Grade:** {eval_result.get('overall_grade', '?')}")
        st.markdown(f"**Personalization:** {eval_result.get('personalization_score', '—')}/10")
        st.markdown(f"**Demo quality:** {eval_result.get('demo_quality', '—')}/10")
        st.markdown("---")

    # Tool call log
    if st.session_state.get("tool_log"):
        with st.expander("🔧 Tool Calls Log"):
            for entry in st.session_state.tool_log:
                icon = TOOL_ICONS.get(entry["tool"], "🔧")
                st.markdown(f"{icon} **{entry['tool']}**({', '.join(f'{k}={v}' for k, v in entry.get('args', {}).items())})")

    # Security & Guardrail log
    guard_log = st.session_state.get("guardrail_log", [])
    if guard_log:
        blocked = sum(1 for e in guard_log if not e.get("safe"))
        total = len(guard_log)
        with st.expander(f"🛡️ Security Log ({blocked} blocked / {total} checks)"):
            for entry in reversed(guard_log[-20:]):
                icon = "✅" if entry.get("safe") else "🚫"
                layer = entry["type"].replace("_", " ").title()
                st.markdown(f"{icon} **{layer}**: {entry.get('detail', 'OK')}")

    # Learned improvements
    learned = load_improvements()
    if learned:
        with st.expander("🧬 Learned Improvements"):
            for imp in learned:
                st.markdown(f"- {imp}")

    st.markdown("### GTM Flow")
    st.markdown("1. **Contact** — Lead arrives\n2. **Demo** — Feature walkthrough\n3. **Qualification** — Collect details\n4. **Use-Case Setup** — AI board gen\n5. **Close & Payment** — Plan + checkout")

    with st.expander("Debug"):
        st.write("Turns:", st.session_state.turn_count)
        st.write("Ready:", st.session_state.ready_to_close)
        st.write("Closed:", st.session_state.closed)
        prof = st.session_state.get("profile", {})
        st.write("**Extracted plan:**", prof.get("preferred_plan", "—"))
        st.write("**Extracted seats:**", prof.get("preferred_seats", "—"))
        st.json(prof)

    # LLM provider pool status
    from groq_pool import pool as _llm_pool
    pool_status = _llm_pool.status()
    if pool_status:
        provider_label = f"🏠 Ollama" if _llm_pool.provider == "ollama" else f"☁️ Groq"
        with st.expander(f"🔑 LLM Pool — {provider_label}"):
            for ks in pool_status:
                prov = ks.get("provider", "groq")
                if prov == "ollama":
                    icon = "🟢" if ks["active"] else "⚪"
                    st.markdown(f"{icon} **Ollama** (local fallback) — {'active' if ks['active'] else 'standby'}")
                else:
                    icon = "🟢" if ks["active"] else ("🔴" if ks["rate_limited"] else "⚪")
                    label = "active" if ks["active"] else ("cooldown" if ks["rate_limited"] else "standby")
                    cd = f" ({ks['cooldown_remaining']:.0f}s)" if ks["rate_limited"] else ""
                    st.markdown(f"{icon} Groq `{ks['key_hint']}` — {label}{cd}")

# ═══════════════════════════════════════════════════════════════════════════
# View routing — board and checkout views replace the main content
# ═══════════════════════════════════════════════════════════════════════════

if "current_view" not in st.session_state:
    st.session_state.current_view = "chat"

_VIEW_RENDERERS = {"board": _render_board_view, "checkout": _render_checkout_view}

if st.session_state.current_view in _VIEW_RENDERERS:
    view = st.session_state.current_view
    if st.button("← Back to Sales Agent", key=f"back_from_{view}"):
        st.session_state.current_view = "chat"
        st.rerun()
    _VIEW_RENDERERS[view]()
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Sales Agent", key=f"back_from_{view}_bottom"):
        st.session_state.current_view = "chat"
        st.rerun()
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# Title & progress bar
# ═══════════════════════════════════════════════════════════════════════════

st.title("monday.com AI Sales Agent")
st.caption("Autonomous AI sales agent with tool calling — from discovery to payment, no rep needed.")
phase = _phase_from_profile(st.session_state.get("profile", {}), st.session_state.get("ready_to_close", False), st.session_state.get("closed", False))
_render_progress(phase)

# ═══════════════════════════════════════════════════════════════════════════
# Chat display
# ═══════════════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Close button
if st.session_state.ready_to_close and not st.session_state.closed:
    col_l, col_btn, col_r = st.columns([1, 2, 1])
    with col_btn:
        if st.button("🚀 Set up my workspace", use_container_width=True, type="primary"):
            # Re-extract profile to capture any plan/seat changes from negotiation
            fresh_profile = extract_profile(st.session_state.messages)
            st.session_state.profile = fresh_profile
            # Clear stale caches so results reflect the latest agreement
            for key in CACHE_KEYS:
                st.session_state.pop(key, None)
            st.session_state.closed = True
            close_msg = "Great, let's do it! I'm setting up your tailored workspace now…"
            st.session_state.messages.append({"role": "assistant", "content": close_msg})
            st.rerun()

if st.session_state.closed and st.session_state.get("profile"):
    _render_results(st.session_state.profile)

# ═══════════════════════════════════════════════════════════════════════════
# Process a single user turn (used by both manual input and auto-demo)
# ═══════════════════════════════════════════════════════════════════════════

def _process_turn(user_text: str):
    """Run the full agentic pipeline for one user message.

    Pipeline: Sanitize → Rate Limit → Regex Pre-filter → LLM Input Guard → Sentiment
    → Extractor → Lead Score → Strategy Planner → Conversation LLM (streaming + tools)
    → Output Sanitize → Output Guard (with auto-regen) → Qualification
    """
    # ── 0a. Input sanitization ──────────────────────────────────────────
    user_text = sanitize_input(user_text)
    if not user_text:
        return

    with st.chat_message("user"):
        st.markdown(user_text)

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.turn_count += 1

    # ── 0b. Rate limiting ───────────────────────────────────────────────
    rate_check = st.session_state.rate_limiter.check()
    if not rate_check["allowed"]:
        retry = rate_check["retry_after"]
        block_msg = f"You're sending messages too quickly. Please wait {retry} seconds and try again."
        with st.chat_message("assistant"):
            st.markdown(block_msg)
        st.session_state.messages.append({"role": "assistant", "content": block_msg})
        return

    # ── 0c. Regex pre-filter (fast, before LLM) ────────────────────────
    regex_check = regex_injection_check(user_text)
    if not regex_check["safe"]:
        block_msg = "I'm here to help you explore monday.com for your team. Let's focus on finding the right workspace setup for you!"
        with st.chat_message("assistant"):
            st.markdown(block_msg)
        st.session_state.messages.append({"role": "assistant", "content": block_msg})
        st.session_state.guardrail_log.append({"type": "regex_prefilter", "safe": False, "detail": regex_check.get("reason", "Pattern match")})
        return

    # ── 1. LLM input guardrail ──────────────────────────────────────────
    recent_msgs = st.session_state.messages[-6:]
    guard_context = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in recent_msgs if m.get("role") in ("user", "assistant")
    )
    input_check = check_input(user_text, guard_context)
    st.session_state.guardrail_log.append({"type": "input", "safe": input_check["safe"], "detail": input_check.get("reason", "OK")})

    if not input_check["safe"]:
        classification = input_check.get("classification", "off_topic")
        if classification == "injection":
            block_msg = "I'm here to help you explore monday.com for your team. Let's focus on finding the right workspace setup for you!"
        else:
            block_msg = "That's a bit outside what I can help with — I'm focused on finding the right monday.com setup for your team. What can I help you with?"
        with st.chat_message("assistant"):
            st.markdown(block_msg)
        st.session_state.messages.append({"role": "assistant", "content": block_msg})
        return

    # ── 2-4. Parallel: sentiment + extractor + lead scorer ─────────────
    #    These three are independent — run them concurrently to cut latency.
    #    Strategy planner depends on all three, so it runs after they complete.
    recent_ctx = "\n".join(m["content"] for m in st.session_state.messages[-4:] if m["role"] in ("user", "assistant"))
    msgs_snapshot = list(st.session_state.messages)
    old_plan = st.session_state.get("profile", {}).get("preferred_plan")
    old_seats = st.session_state.get("profile", {}).get("preferred_seats")
    old_profile = st.session_state.get("profile", {})

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_sentiment = pool.submit(analyze_sentiment, user_text, recent_ctx)
        fut_profile = pool.submit(extract_profile, msgs_snapshot)
        fut_lead = pool.submit(score_lead, msgs_snapshot, old_profile)

    sentiment = fut_sentiment.result()
    profile = fut_profile.result()
    lead_result = fut_lead.result()

    st.session_state.sentiment_history.append(sentiment)
    st.session_state.profile = profile
    st.session_state.lead_score = lead_result

    new_plan = profile.get("preferred_plan")
    new_seats = profile.get("preferred_seats")
    if st.session_state.closed and (new_plan != old_plan or new_seats != old_seats):
        st.session_state.closed = False
        st.session_state.ready_to_close = True
        for key in CACHE_KEYS:
            st.session_state.pop(key, None)

    # Re-score lead with fresh profile (lead scorer ran with old profile)
    if profile != old_profile:
        lead_result = score_lead(msgs_snapshot, profile)
        st.session_state.lead_score = lead_result

    # ── 5. Strategy planner (depends on sentiment + profile + lead) ───
    strategy = plan_strategy(st.session_state.messages, profile, sentiment, lead_result)
    st.session_state.strategy_log.append(strategy)

    # ── 5b. Live coaching (fires every 3 turns) ──────────────────────
    coaching = live_coaching_check(
        st.session_state.messages, profile, st.session_state.turn_count
    )
    if coaching:
        st.session_state.setdefault("coaching_log", []).append(coaching)

    # ── 6. Conversation LLM with streaming + tool calling ───────────────
    tool_calls = []
    full_response = ""

    with st.chat_message("assistant"):
        with st.expander("🧠 Agent Reasoning", expanded=True):
            next_move = strategy.get("next_move", strategy.get("strategy", "Continue naturally"))
            st.markdown(
                f'<div class="tool-call">🎯 <span class="tool-name">Strategy</span>: '
                f'{next_move} '
                f'<em style="color:#676879">(priority: {strategy.get("priority", "?")})</em></div>',
                unsafe_allow_html=True,
            )
            if coaching:
                st.markdown(
                    f'<div class="tool-call">🎙️ <span class="tool-name">Live Coach</span>: '
                    f'{coaching}</div>',
                    unsafe_allow_html=True,
                )
            tool_placeholder = st.empty()

        response_placeholder = st.empty()

        stream = conversation_stream_with_tools(
            st.session_state.messages, profile,
            sentiment=sentiment, strategy=strategy,
            live_coaching=coaching,
        )
        streamed_text = ""

        for chunk_type, data in stream:
            if chunk_type == "tool":
                tool_calls.append(data)
                icon = TOOL_ICONS.get(data["tool"], "🔧")
                args_str = ", ".join(f"{k}={v}" for k, v in data.get("args", {}).items())
                tool_placeholder.markdown(
                    f'<div class="tool-call">{icon} <span class="tool-name">{data["tool"]}</span>({args_str})</div>',
                    unsafe_allow_html=True,
                )
            elif chunk_type == "token":
                streamed_text += data
                response_placeholder.markdown(streamed_text + "▌")
            elif chunk_type == "done":
                full_response = data.get("full_text", streamed_text)
                tool_calls = data.get("tool_log", tool_calls)

        response_placeholder.markdown(full_response)

    if not full_response:
        full_response = "I'm sorry, I'm having trouble right now. Please try again."

    # ── 6b. Output sanitization (strip dangerous HTML/JS) ────────────────
    full_response = sanitize_output(full_response)

    # Store tool calls
    if tool_calls:
        st.session_state.tool_log.extend(tool_calls)

    # ── 7. Output guardrail with auto-regeneration ──────────────────────
    output_check = check_output(full_response)
    st.session_state.guardrail_log.append({"type": "output", "safe": output_check["safe"], "detail": output_check.get("issue", "OK")})

    if not output_check["safe"]:
        issue = output_check.get("issue", "")
        corrected = regenerate_safe_response(full_response, issue, st.session_state.messages)
        if corrected and corrected != full_response:
            st.session_state.guardrail_log.append({"type": "auto-correction", "safe": True, "detail": f"Regenerated: {issue}"})
            full_response = corrected
            with st.chat_message("assistant"):
                st.caption("🛡️ *Response auto-corrected by safety guardrail*")
                st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # ── 8. Check qualification ──────────────────────────────────────────
    if not st.session_state.ready_to_close and is_fully_qualified(profile):
        st.session_state.ready_to_close = True

# ═══════════════════════════════════════════════════════════════════════════
# Auto-demo mode execution
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.get("auto_demo_running", False):
    step = st.session_state.get("auto_demo_step", 0)
    if step < len(AUTO_DEMO_MESSAGES):
        st.session_state.rate_limiter.reset()
        _process_turn(AUTO_DEMO_MESSAGES[step])
        st.session_state.auto_demo_step = step + 1
        if st.session_state.auto_demo_step < len(AUTO_DEMO_MESSAGES):
            st.rerun()
        else:
            st.session_state.auto_demo_running = False
            if st.session_state.ready_to_close and not st.session_state.closed:
                st.session_state.closed = True
                close_msg = "Great, let's do it! I'm setting up your tailored workspace now…"
                st.session_state.messages.append({"role": "assistant", "content": close_msg})
            st.rerun()
    else:
        st.session_state.auto_demo_running = False

# ═══════════════════════════════════════════════════════════════════════════
# Manual user input
# ═══════════════════════════════════════════════════════════════════════════

if not st.session_state.get("auto_demo_running", False):
    if st.session_state.closed:
        placeholder = "Still have questions? Ask away…"
    elif st.session_state.ready_to_close:
        placeholder = "Have questions before we set things up? Or click the button above…"
    else:
        placeholder = "Tell me about your company…"

    user_input = st.chat_input(placeholder)

    if user_input and isinstance(user_input, str):
        _process_turn(user_input)
        st.rerun()
