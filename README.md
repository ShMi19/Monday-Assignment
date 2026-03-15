## AI Sales Agent Demo

This repository contains a small, working prototype of an AI-powered sales agent for monday.com.
It demonstrates how an autonomous agent can guide a prospect from discovery to a mock payment flow.

### 1. Hands-On Prototype

- **Stack**: Python, Streamlit, Groq (LLM API), `python-dotenv`.
- **Entry point**: `app.py` (Streamlit UI).
- **Core logic**: `agent.py` (LLM client + JSON extraction), `prompts.py` (qualification prompt).
- **Mock integrations**:
  - `monday_mock.py` – simulates auto-generating a monday board from the use case.
  - `stripe_mock.py` – simulates creating a payment link based on a selected plan.

#### Running the demo

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set the Groq API key in a `.env` file:

```bash
echo GROQ_API_KEY=your_key_here > .env
```

4. Launch the Streamlit app:

```bash
streamlit run app.py
```

5. Open the URL shown in the terminal (usually `http://localhost:8501`) and start chatting with the agent.

### 2. Flow Overview

**Customer journey through the agent:**

1. **Discovery** – A prospect lands on the page and starts a chat in the Streamlit UI.
2. **Qualification conversation** – The LLM, guided by `QUALIFICATION_PROMPT`, asks targeted questions to capture:
   - `industry`
   - `company_size`
   - `team_using_monday`
   - `use_case`
3. **Structured handoff** – When the LLM is confident it has all four fields, it emits **JSON only**.
   - `agent.try_extract_json` robustly parses the JSON even if it's wrapped in extra text or code fences.
4. **Initial use-case setup** – The app:
   - Calls `generate_monday_board(use_case)` to create a mock board and demo link.
   - Selects a plan based on `company_size` (Starter / Standard / Pro) to mimic packaging logic.
5. **Close & payment** – The app calls `create_payment_link(plan)` to simulate generating a Stripe-style payment URL.

At any time, the user can reset the conversation using the sidebar and start a new prospect journey.

### 3. AI Prompting & Logic

The main prompt (`QUALIFICATION_PROMPT`) enforces:

- Explicit collection of four key GTM fields.
- Asking 1–2 clear questions at a time.
- Avoiding repeated questions by reusing and refining previous answers.
- Emitting **pure JSON** with a fixed schema once qualification is complete.

On the parsing side:

- `ask_llm(messages)` keeps the message history and sends it to Groq's `llama-3.3-70b-versatile` with low temperature for stability.
- `try_extract_json(text)`:
  - First tries to parse the response as raw JSON.
  - If that fails, searches for JSON inside code fences or the first `{ ... }` block.
  - Returns a Python dict only when valid JSON is found.

The app:

- Validates that all required keys are present before marking qualification as complete.
- Stores the structured data in `st.session_state.qualified_data` so the summary and links persist across reloads.

### 4. Edge Cases & Confusion Handling

- **Non-JSON responses** – The agent continues the conversation; the UI does not break and simply waits for the JSON completion.
- **JSON wrapped in explanations or ```json fences** – Handled by the JSON extraction helper.
- **Missing keys** – The UI shows a warning and the agent keeps asking follow-up questions and tracks JSON parse failures.
- **Multiple prospects / restart** – Sidebar button resets the conversation and state.
- **Observability for reviewers** – Sidebar “Debug” expander shows the last LLM response and a counter of JSON parsing failures.

### 5. Design Decisions & Assumptions

- **Mock integrations over live APIs** – All external systems (monday, Stripe, LLM provider choice) are intentionally mocked so the prototype is easy and free to run locally.
- **Single primary journey** – The prototype optimizes for the “happy path” of a single prospect from discovery to payment, but the structure (persistent state + structured JSON) makes it straightforward to support parallel sessions in a real web app.
- **Qualification-first mindset** – The system will not create a board or payment link until qualification JSON passes strict validation, mirroring a B2B GTM where you avoid misaligned demos or pricing.
- **Static pricing tiers** – Plan selection is mocked with simple rules on `company_size`, but the flow mirrors how a pricing engine could plug in later without changing the conversation layer.

### 6. Possible Extensions (Not Implemented)

All of the following are intentionally mocked to keep the prototype free to run locally:

- **Real monday.com integration**:
  - Use monday's GraphQL API to create a real board, groups, and items based on `use_case` and `team_using_monday`.
  - Store the created board ID in the qualification JSON for downstream workflows.
- **Real Stripe integration**:
  - Use Stripe Checkout Sessions or Payment Links to create live checkout URLs tied to the selected plan.
  - Capture webhook events to mark the prospect as converted in a CRM or monday board.
- **Hosted LLM provider (paid)**:
  - For production, you might prefer a managed, enterprise-grade provider (e.g., OpenAI, Anthropic, or similar),
    with features like dedicated capacity, observability, vector search, and better SLAs.
  - The current `agent.py` structure is designed so swapping the underlying client (Groq vs OpenAI vs Anthropic) only changes one module.

These would add cost, so they are described here as design choices rather than required to run this assignment.

