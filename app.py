import re
import pandas as pd
import streamlit as st
from agent import ask_llm, try_extract_json
from monday_mock import generate_monday_board
from stripe_mock import create_payment_link
from prompts import QUALIFICATION_PROMPT

GREETING = (
    "Hi there! I'm monday.com's AI sales assistant. "
    "I'd love to learn about your team and help set up a workspace tailored to your needs.\n\n"
    "To get started — what industry is your company in, and which team would be using monday.com?"
)

PHASE_LABELS = [
    "Contact",
    "Demo",
    "Qualification",
    "Use-Case Setup",
    "Close & Payment",
]

st.set_page_config(page_title="monday.com AI Sales Agent", page_icon="🤖")
st.title("monday.com AI Sales Agent")
st.caption(
    "Autonomous AI sales agent — from discovery to payment, no rep needed."
)


# ── helpers ──────────────────────────────────────────────────────────────

def _parse_company_size(raw: str) -> int | None:
    if not raw:
        return None
    numbers = re.findall(r"\d+", raw)
    if not numbers:
        return None
    nums = [int(n) for n in numbers]
    if len(nums) >= 2:
        return (nums[0] + nums[1]) // 2
    return nums[0]


def _select_plan(company_size_raw: str) -> str:
    size = _parse_company_size(company_size_raw)
    if size is None:
        return "Standard"
    if size <= 10:
        return "Starter"
    if size <= 50:
        return "Standard"
    return "Pro"


def _render_board_table(columns: list, items: list):
    """Render board columns and items as a visual table."""
    col_ids = [c.get("id", c.get("title", "").lower().replace(" ", "_")) for c in columns]
    col_titles = [c.get("title", c.get("id", "")) for c in columns]

    rows = []
    for item in items:
        row = {}
        for cid, ctitle in zip(col_ids, col_titles):
            value = item.get(cid) or item.get(ctitle) or item.get(ctitle.lower(), "")
            if not value:
                for k, v in item.items():
                    if k.lower().replace("_", " ") == ctitle.lower():
                        value = v
                        break
            if not value and "name" in cid:
                value = item.get("name", "")
            row[ctitle] = value or ""
        rows.append(row)

    if not rows:
        st.info("No example items generated.")
        return

    df = pd.DataFrame(rows, columns=col_titles)
    st.dataframe(df, width="stretch", hide_index=True)


def _current_phase() -> int:
    """Determine which GTM phase we're in based on conversation state."""
    if st.session_state.qualified_data:
        return 4
    turns = st.session_state.turn_count
    if turns == 0:
        return 0
    if turns <= 2:
        return 1
    return 2


def _render_progress(phase: int):
    """Show a visual step indicator for the 5-phase GTM flow."""
    cols = st.columns(len(PHASE_LABELS))
    for i, (col, label) in enumerate(zip(cols, PHASE_LABELS)):
        if i < phase:
            col.markdown(f"~~**:green[✓ {label}]**~~")
        elif i == phase:
            col.markdown(f"**:blue[→ {label}]**")
        else:
            col.markdown(f":gray[{label}]")


def _extract_partial_profile(messages: list) -> dict:
    """Try to extract partial prospect info from conversation for the sidebar."""
    profile = {}
    text = " ".join(
        m["content"] for m in messages
        if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)
    ).lower()

    if st.session_state.qualified_data:
        return st.session_state.qualified_data

    for m in messages:
        if m["role"] != "assistant":
            continue
        data = try_extract_json(m.get("content", ""))
        if data and isinstance(data, dict):
            for key in ("industry", "company_size", "team_using_monday", "use_case"):
                if key in data and data[key]:
                    profile[key] = data[key]

    return profile


def _render_results(data: dict):
    plan = _select_plan(data.get("company_size", ""))
    seats = _parse_company_size(data.get("company_size", ""))

    with st.spinner("Generating your tailored workspace with AI…"):
        board = generate_monday_board(data)

    payment = create_payment_link(plan, seats)

    st.write("### ✓ Qualification Summary")
    col_ind, col_size, col_team, col_uc = st.columns(4)
    col_ind.metric("Industry", data.get("industry", "—"))
    col_size.metric("Company Size", data.get("company_size", "—"))
    col_team.metric("Team", data.get("team_using_monday", "—"))
    uc = data.get("use_case", "—")
    col_uc.metric("Use Case", uc[:30] + "…" if len(uc) > 30 else uc)

    st.write("### ✓ Your Tailored Workspace")
    st.write(f"**{board['board_name']}**")
    if board.get("generated_by") == "ai":
        st.caption("Board columns and items were AI-generated from your use case.")

    _render_board_table(board["columns"], board["items"])
    st.write(f"[Open board in monday.com →]({board['link']})")

    st.write("### ✓ Suggested Plan")
    st.write(f"**{plan}** — {payment['price_per_seat']}")

    st.write("### ✓ Checkout")
    col1, col2 = st.columns(2)
    col1.metric("Seats", payment["seats"])
    col2.metric("Total", payment["subtotal"])
    st.write(f"[Proceed to payment →]({payment['checkout_url']})")
    st.caption(f"Session: `{payment['session_id']}`")

    st.divider()
    team = data.get("team_using_monday", "your team")
    st.write(
        f"You're all set! I've created a **{plan}** workspace for "
        f"**{team}** to get started right away. "
        f"Click the payment link above to activate your account. "
        f"Welcome to monday.com! 🎉"
    )


# ── session state ────────────────────────────────────────────────────────

def _reset_conversation():
    st.session_state.messages = [
        {"role": "system", "content": QUALIFICATION_PROMPT},
        {"role": "assistant", "content": GREETING},
    ]
    st.session_state.qualified_data = None
    st.session_state.turn_count = 0


if "messages" not in st.session_state:
    _reset_conversation()
if "qualified_data" not in st.session_state:
    st.session_state.qualified_data = None
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0


# ── progress indicator ───────────────────────────────────────────────────

phase = _current_phase()
_render_progress(phase)
st.divider()


# ── sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Session Controls")
    if st.button("Start new prospect"):
        _reset_conversation()
        st.rerun()

    st.markdown("---")

    st.markdown("### Prospect Profile")
    profile = _extract_partial_profile(st.session_state.messages)
    if profile:
        if profile.get("industry"):
            st.write(f"**Industry:** {profile['industry']}")
        if profile.get("team_using_monday"):
            st.write(f"**Team:** {profile['team_using_monday']}")
        if profile.get("company_size"):
            st.write(f"**Company Size:** {profile['company_size']}")
        if profile.get("use_case"):
            st.write(f"**Use Case:** {profile['use_case']}")
        if st.session_state.qualified_data:
            st.success("Qualification complete")
    else:
        st.caption("No data collected yet.")

    st.markdown("---")

    st.markdown("### monday.com GTM Flow")
    st.markdown(
        "1. **Contact** — Lead arrives, agent greets.\n"
        "2. **Demo** — Personalized feature walkthrough.\n"
        "3. **Qualification** — Collect team size & use case.\n"
        "4. **Use-Case Setup** — Auto-generate a board.\n"
        "5. **Close & Payment** — Plan + payment link."
    )

    with st.expander("Debug (for reviewers)"):
        st.write("Current phase:", PHASE_LABELS[phase])
        st.write("Conversation turns:", st.session_state.turn_count)
        last_assistant = None
        for m in reversed(st.session_state.messages):
            if m["role"] == "assistant":
                last_assistant = m["content"]
                break
        if last_assistant:
            st.text_area("Last LLM response", last_assistant, height=150)


# ── chat display ─────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ── user input ───────────────────────────────────────────────────────────

prompt = st.chat_input("Tell me about your company…")

if prompt and isinstance(prompt, str):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.turn_count += 1

    response = ask_llm(st.session_state.messages)
    data = try_extract_json(response)

    if data:
        required_keys = {"industry", "company_size", "team_using_monday", "use_case"}
        missing = required_keys - set(data.keys())

        if missing:
            with st.chat_message("assistant"):
                st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            friendly = (
                "Thanks for sharing all of that! I now have everything I need. "
                "Let me set up a tailored workspace for you."
            )
            with st.chat_message("assistant"):
                st.write(friendly)
            st.session_state.messages.append({"role": "assistant", "content": friendly})

            st.session_state.qualified_data = data
            st.success("Qualification complete — workspace ready!")
            _render_results(data)
            st.rerun()
    else:
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

elif st.session_state.get("qualified_data"):
    _render_results(st.session_state["qualified_data"])
