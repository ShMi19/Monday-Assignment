import streamlit as st
from agent import ask_llm, try_extract_json
from monday_mock import generate_monday_board
from stripe_mock import create_payment_link
from prompts import QUALIFICATION_PROMPT


st.set_page_config(page_title="AI Sales Agent Demo", page_icon="🤖")
st.title("AI Sales Agent Demo")
st.caption(
    "Prototype of an autonomous AI sales agent guiding a prospect from discovery to payment."
)


def reset_conversation():
    st.session_state.messages = [
        {"role": "system", "content": QUALIFICATION_PROMPT},
    ]
    st.session_state.qualified_data = None
    st.session_state.json_failures = 0


# initialize conversation
if "messages" not in st.session_state:
    reset_conversation()

if "qualified_data" not in st.session_state:
    st.session_state.qualified_data = None

if "json_failures" not in st.session_state:
    st.session_state.json_failures = 0

with st.sidebar:
    st.subheader("Session Controls")
    if st.button("Start new prospect"):
        reset_conversation()
        st.success("Conversation reset. You can start a new prospect.")

    st.markdown("### What this demo shows")
    st.markdown(
        "- **Qualification**: conversational capture of key GTM fields.\n"
        "- **Board generation**: mock monday board (name, columns, example items).\n"
        "- **Payment**: mock Stripe-style payment link based on plan."
    )

    with st.expander("Flow overview"):
        st.markdown(
            "1. Chat with the prospect to collect industry, company size, team, and use case.\n"
            "2. When confident, the LLM emits **JSON only** with those fields.\n"
            "3. The app parses that JSON, validates required keys, and builds a monday-style board.\n"
            "4. A plan is selected from the company size and a payment link is generated."
        )

    with st.expander("Debug (for reviewers)"):
        st.write("JSON parse failures in this session:", st.session_state.json_failures)
        last_msg = (
            st.session_state.messages[-1]["content"]
            if len(st.session_state.messages) > 1
            else None
        )
        if last_msg:
            st.text_area("Last LLM response", last_msg, height=150)

# show previous messages (excluding system prompt)
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# user input
prompt = st.chat_input("Tell me about your company")

if prompt and isinstance(prompt, str):
    # display user message
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # call LLM
    response = ask_llm(st.session_state.messages)

    # display AI message
    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # Try to extract structured qualification data from the latest response
    data = try_extract_json(response)

    if data:
        required_keys = {"industry", "company_size", "team_using_monday", "use_case"}
        missing = required_keys - set(data.keys())

        if missing:
            st.session_state.json_failures += 1
            st.warning(
                "I tried to structure the data, but some fields are still missing "
                f"({', '.join(sorted(missing))}). I'll keep asking follow-up questions."
            )
        else:
            st.session_state.qualified_data = data
            st.success("Qualification complete")

            # Simple plan selection based on company size (mock logic)
            size = (data.get("company_size") or "").lower()
            if any(s in size for s in ["1-10", "1-20", "solo", "1-5"]):
                plan = "Starter"
            elif any(s in size for s in ["11-50", "20-50", "50-100"]):
                plan = "Standard"
            else:
                plan = "Pro"

            board = generate_monday_board(data)
            payment = create_payment_link(plan)

            st.write("### Qualification Summary")
            st.json(data)

            st.write("### Demo Board")
            st.write(board["board_name"])
            st.write(board["link"])
            st.write("Columns:")
            st.json(board["columns"])
            st.write("Example items:")
            st.json(board["items"])

            st.write("### Suggested Plan")
            st.write(plan)

            st.write("### Payment Link")
            st.write(payment)

    else:
        # No parseable JSON in this turn – track it for observability
        st.session_state.json_failures += 1
        if st.session_state.json_failures >= 3:
            st.info(
                "I'm still gathering information in natural language. "
                "Once I have everything I need, I'll summarize it in structured form."
            )

elif st.session_state.get("qualified_data"):
    # If the user reloads after qualification, keep showing the results.
    data = st.session_state["qualified_data"]

    st.write("### Qualification Summary")
    st.json(data)

    board = generate_monday_board(data)

    size = (data.get("company_size") or "").lower()
    if any(s in size for s in ["1-10", "1-20", "solo", "1-5"]):
        plan = "Starter"
    elif any(s in size for s in ["11-50", "20-50", "50-100"]):
        plan = "Standard"
    else:
        plan = "Pro"

    payment = create_payment_link(plan)

    st.write("### Demo Board")
    st.write(board["board_name"])
    st.write(board["link"])
    st.write("Columns:")
    st.json(board["columns"])
    st.write("Example items:")
    st.json(board["items"])

    st.write("### Suggested Plan")
    st.write(plan)

    st.write("### Payment Link")
    st.write(payment)