# monday.com AI Sales Agent

An autonomous AI sales agent that replaces the traditional 5-step GTM process — from first contact to payment — using a multi-agent LLM pipeline with agentic tool calling, real-time strategy planning, and defense-in-depth security.

## Quick Start

```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_your_key_here" > .env
python -m streamlit run app.py
```

Optional: provide multiple Groq keys for auto-rotation on rate limits: `GROQ_API_KEYS=key1,key2,key3`

**For reviewers:** Click **"Auto Demo"** in the sidebar to watch the full 5-phase GTM flow run live in ~30 seconds — no typing needed. The real system executes on your machine: strategy planning, tool calls, streaming, extraction, scoring, board generation, payment, email, and self-improvement — all visible in real time.

---

## 1. Architecture

Every user message passes through up to **8 LLM calls** (7 per turn + 1 periodic coach) plus 3 non-LLM security layers. Three of those LLM calls run **in parallel**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Message                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │  Pre-LLM Security Layers     │
              │  • Input sanitization         │
              │  • Rate limiting (15 msg/min) │
              │  • Regex injection pre-filter │
              └──────────────┬───────────────┘
                             │
                    ┌────────▼────────┐
                    │  Input Guardrail │  ← Context-aware: sees last 6 messages
                    │  (20b)           │
                    └───┬─────────┬───┘
              blocked   │         │ safe
                        │         │
     ╔══════════════════╧═════════╧══════════════════╗
     ║     PARALLEL EXECUTION (ThreadPoolExecutor)    ║
     ║                                                ║
     ║  ┌──────────────┐ ┌────────────┐ ┌──────────┐ ║
     ║  │  Sentiment    │ │  Extractor │ │   Lead   │ ║
     ║  │  Analyzer     │ │  (20b)     │ │  Scorer  │ ║
     ║  │  (20b)        │ │  confidence│ │  (20b)   │ ║
     ║  │  emotion +    │ │  -scored   │ │  0-100 + │ ║
     ║  │  intensity    │ │  fields    │ │  signals │ ║
     ║  └──────┬───────┘ └─────┬──────┘ └────┬─────┘ ║
     ╚════════╤════════════════╤══════════════╤═══════╝
              └────────────────┼──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Strategy Planner    │  ← Autonomous meta-reasoning:
                    │  (20b)               │    "fastest path to signed deal"
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Live Coach (20b)    │  ← Fires every 3 turns
                    │  "Sales manager      │    e.g. "Stop asking questions,
                    │   listening in"      │     show value next"
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  Conversation LLM (120b)         │
                    │  WITH AUTONOMOUS TOOL CALLING     │
                    │  + TOKEN STREAMING                │
                    │                                   │
                    │  Tools (LLM decides when):        │
                    │  • lookup_pricing                 │
                    │  • compare_competitor             │
                    │  • calculate_roi                  │
                    │  • suggest_automations            │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Output Guardrail    │  ← If flagged → auto-regenerate
                    │  (20b)               │
                    └──────────┬──────────┘
                               │
                    All fields ≥ 0.7 confidence?
                         │              │
                        no             yes → "Set up workspace" button appears
                         │                         │
                    Continue                 User clicks button
                    talking                        │
                                   ┌───────────────▼──────────────┐
                                   │     Close Sequence             │
                                   │  • Board Gen (120b)            │
                                   │  • Plan + Payment (mock Stripe)│
                                   │  • Follow-up Email (120b)      │
                                   │  • Evaluator (120b)            │
                                   │  • Self-Improvement (20b)      │
                                   │    → persisted to disk          │
                                   └────────────────────────────────┘
```

### Why 14 Agents Instead of 1

Each agent is separated for a specific reason — not just to add complexity:

| Agent | Model | Why it can't be merged into the Conversation LLM |
|-------|-------|---------------------------------------------------|
| **Input Guardrail** | 20b | Security: the main LLM must not judge its own inputs (separate attack surface) |
| **Sentiment Analyzer** | 20b | Needs to run in parallel with extraction; biases extraction if combined |
| **Extractor** | 20b | Requires temperature 0.0 and strict JSON output; mixing with conversation causes hallucinated fields |
| **Lead Scorer** | 20b | Quantitative scoring (0–100) inflates when combined with the politeness of conversation |
| **Strategy Planner** | 20b | Thinks about WHAT to say; the conversation LLM thinks about HOW to say it |
| **Live Coach** | 20b | Periodic quality monitor (every 3 turns); feeds tactical corrections like a sales manager |
| **Conversation LLM** | 120b | Main agent — needs the largest model for tool-calling decisions and objection handling |
| **Output Guardrail** | 20b | The conversation LLM can't reliably detect its own hallucinations |
| **Board Generator** | 120b | Generating realistic board structures (columns, items, workflows) is a distinct reasoning task |
| **Email Generator** | 120b | Professional email writing has different tone and format requirements |
| **Evaluator** | 120b | The model that ran the conversation shouldn't grade itself |
| **Self-Improvement** | 20b | Converts evaluation results into reusable prompt improvements; persists to disk |
| **Context Summarizer** | 20b | Compresses old messages instead of dropping them; runs only when context exceeds limits |

**Model choice:** `openai/gpt-oss-120b` for quality-critical tasks (conversation, board gen, evaluation); `openai/gpt-oss-20b` for speed-critical tasks run in parallel. API key pool auto-rotates on rate limits with Ollama local fallback.

---

## 2. Customer Journey (5-Phase GTM Flow)

### Phase 1 — Contact
Agent greets and asks about industry and team. The Extractor silently begins collecting structured data. Lead score initializes.

### Phase 2 — Demo
Personalized demo based on extracted context. If the prospect mentions a competitor, the agent autonomously calls `compare_competitor` (backed by a knowledge base covering 7 competitors). If they ask about efficiency, it calls `suggest_automations`.

### Phase 3 — Qualification
Agent naturally collects remaining details. Each field has a confidence score (0.0–1.0); low-confidence fields are prioritized. Lead score updates every turn.

### Phase 4 — Use-Case Setup (Deferred Close)
When all fields reach ≥0.7 confidence, a "Set up workspace" button appears. The agent does **not** auto-close — the prospect can keep chatting, object to pricing, compare tools, or negotiate. The agent handles objections with tool calls (ROI calculation, competitor comparison). Close happens only when the prospect explicitly opts in.

### Phase 5 — Close & Payment
When the prospect clicks the button, the system generates:
- AI-tailored monday.com board (custom columns and example items for their use case)
- Auto-selected plan with per-seat pricing (validated against seat limits)
- Mock Stripe checkout page
- Personalized follow-up email for their team
- Conversation quality scorecard (5 metrics)
- Self-improvement suggestions saved to disk for future sessions

### How Input, Decisions, and Transitions Work
- **Input**: Every message passes through sanitization → regex injection filter → context-aware LLM guardrail before the pipeline sees it
- **Decisions**: The Strategy Planner autonomously decides the approach (discover / educate / close / handle_objection). The Conversation LLM autonomously decides which tools to call. No hardcoded if/else chains.
- **Transitions**: Phase progression is driven by field confidence scores, not hardcoded rules. The agent moves forward naturally as it learns more.

---

## 3. AI Prompting & Logic

### How the Conversation Prompt Works

The main conversation prompt is **dynamically assembled** each turn. The same template produces completely different behavior depending on conversation state:

```
┌─────────────────────────────────────────────────────────┐
│  SYSTEM PROMPT (built fresh every turn)                  │
│                                                          │
│  [Security rules]           ← always present             │
│  [Role + voice guidelines]  ← always present             │
│  [Pricing tiers + limits]   ← always present             │
│                                                          │
│  [Profile summary]          ← "Tech company, 50 people,  │
│                                engineering team.          │
│                                Missing: use case."        │
│                                                          │
│  [Sentiment]                ← "Prospect is skeptical     │
│                                (intensity: 0.7)"         │
│                                                          │
│  [Strategy]                 ← "Priority: handle_objection│
│                                Next move: address pricing │
│                                concern with ROI data"     │
│                                                          │
│  [Live coaching]            ← "You've asked 3 questions   │
│                                without showing value —    │
│                                demonstrate a feature next" │
│                                                          │
│  [Learned improvements]     ← "Always mention a specific  │
│                                automation recipe when      │
│                                discussing efficiency"      │
│                                (from previous sessions)    │
└─────────────────────────────────────────────────────────┘
```

This means a skeptical prospect with pricing concerns gets a completely different agent response than an enthusiastic prospect ready to buy — driven by real-time analysis, not scripted branches.

### Agentic Tool Calling

The Conversation LLM has 4 tools via Groq's native function calling API. It decides **autonomously** which to call and when. All tool calls are visible in the UI's "Agent Reasoning" panel.

| Tool | What it does | When the LLM calls it |
|------|-------------|----------------------|
| `lookup_pricing` | Returns plan details, features, seat limits | Prospect asks about cost |
| `compare_competitor` | Structured comparison from knowledge base | Prospect mentions Jira, Asana, Trello, etc. |
| `calculate_roi` | ROI analysis: current spend vs monday.com | Prospect mentions what they're paying |
| `suggest_automations` | Team-specific automation recipes | Prospect asks about efficiency |

The competitor knowledge base (`competitive_intel.json`) covers 7 competitors with strengths, weaknesses, win strategies, and specific objection handlers.

### Branching

Branching happens at six levels with no hardcoded if/else chains:

1. **Strategy**: Planner reasons autonomously about the fastest path to close
2. **Coaching**: Live coach overrides or refines strategy based on conversation quality
3. **Tools**: LLM decides which tools to call based on conversation context
4. **Confidence**: Low-confidence fields are nudged in the prompt
5. **Sentiment**: Agent adapts tone — empathy for frustration, data for skepticism
6. **Safety**: Input guardrail catches injections; output guardrail auto-regenerates bad responses

### Data Capture (Structured Extraction)

The Extractor LLM runs every turn, outputting JSON with per-field confidence scores:

```json
{
  "industry": {"value": "fintech", "confidence": 0.9},
  "company_size": {"value": "50", "confidence": 0.8},
  "team": {"value": "engineering", "confidence": 0.85},
  "use_case": {"value": "sprint planning", "confidence": 0.7},
  "preferred_plan": {"value": "Standard", "confidence": 0.6},
  "preferred_seats": 20
}
```

Fields below 0.7 trigger follow-up questions. Regex fallback catches plan/seat mentions the LLM may miss.

---

## 4. Security (Defense-in-Depth)

| Layer | What It Does |
|-------|-------------|
| **Input Sanitization** | Unicode normalization, control char stripping, 2000-char limit |
| **Regex Pre-filter** | 12+ patterns catch known injection attacks before the LLM sees the message |
| **LLM Input Guardrail** | Context-aware classification (safe / injection / off-topic) |
| **System Prompt Armor** | Anti-extraction instructions at highest priority in every prompt |
| **Output Sanitization** | Strips script tags, event handlers, javascript: URLs |
| **LLM Output Guardrail** | Catches hallucinations and unauthorized pricing — auto-regenerates flagged responses |

Additional: rate limiting (15 msg/60s), tool argument validation, pricing locked to official tiers ($9/$12/$19).

---

## 5. Assumptions & Shortcuts

| Area | What's Mocked | Production Path |
|------|---------------|-----------------|
| **monday.com API** | `monday_mock.py` generates board JSON via LLM | monday.com GraphQL API with OAuth2 |
| **Stripe Payments** | `stripe_mock.py` returns mock checkout URL + session ID | Stripe Checkout Sessions API |
| **User Auth** | No auth — Streamlit session state per browser tab | OAuth2 / SSO + database sessions |
| **Persistence** | State in `st.session_state` (lost on refresh); self-improvement persists to disk | Redis or database |
| **LLM Provider** | Groq free tier with key rotation + Ollama fallback | Production API with proper rate limits |
| **Competitive Intel** | Static JSON file, 7 competitors | Live data source or CRM sync |
| **Email Delivery** | Generated and displayed, not sent | SendGrid / Mailgun integration |
| **CRM** | Lead scoring displayed, not synced | Push to monday.com CRM via API |

---

## File Structure

```
├── app.py                  # Streamlit UI: parallel execution, reasoning panel, auto-demo
├── agent.py                # 14-agent LLM pipeline: tools, streaming, coaching, compression
├── prompts.py              # 13 specialized prompts with dynamic enrichment
├── groq_pool.py            # API key pool with auto-rotation + Ollama fallback
├── tools.py                # Tool definitions + competitive_intel.json loader
├── competitive_intel.json  # Structured competitor KB (7 competitors, win strategies)
├── security.py             # Defense-in-depth: sanitization, regex, rate limit, validation
├── guardrails.py           # LLM-based input/output safety + auto-regeneration
├── evaluator.py            # Quality scoring + self-improvement + disk persistence
├── config.py               # Shared constants, pricing (single source of truth), utilities
├── monday_mock.py          # Mock board generation
├── stripe_mock.py          # Mock payment links
├── requirements.txt        # Dependencies
└── .env                    # API keys (not committed)
```
