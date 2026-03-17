# monday.com AI Sales Agent

An autonomous AI sales agent that replaces the traditional 5-step GTM process — from first contact to payment — using a 14-agent LLM pipeline with agentic tool calling, real-time strategy planning, and defense-in-depth security.

## Quick Start

```bash
pip install -r requirements.txt
# Single key:
echo "GROQ_API_KEY=gsk_your_key_here" > .env
# Multiple keys (recommended — auto-rotates on rate limits):
echo "GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3" >> .env
python -m streamlit run app.py
```

**For reviewers:** Click **"Auto Demo"** in the sidebar to watch the full 5-phase GTM flow run automatically in ~30 seconds — no typing needed. This is better than a pre-recorded video because the real system executes live on your machine: strategy planning, tool calls, streaming, extraction, scoring, board generation, payment, email, and self-improvement — all visible in real time.

---

## Architecture

### Per-Turn Pipeline (14 Specialized LLM Agents)

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
     ║  │  Analyzer     │ │  LLM (20b) │ │  Scorer  │ ║
     ║  │  (20b)        │ │  confidence│ │  (20b)   │ ║
     ║  │  emotion +    │ │  -scored   │ │  0-100 + │ ║
     ║  │  intensity    │ │  fields    │ │  signals │ ║
     ║  └──────┬───────┘ └─────┬──────┘ └────┬─────┘ ║
     ╚════════╤════════════════╤══════════════╤═══════╝
              └────────────────┼──────────────┘
                               │ all three complete (~1s vs ~3s sequential)
                               │
                    ┌──────────▼──────────┐
                    │  Strategy Planner    │  ← Autonomous meta-reasoning:
                    │  (20b)               │    "fastest path to signed deal"
                    │  Sees: profile +     │    visible in Agent Reasoning panel
                    │  sentiment + score   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Live Coach (20b)    │  ← Fires every 3 turns
                    │  "Sales manager      │    "You're asking too many
                    │   listening in"      │     questions — show value next"
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  Conversation LLM (120b)         │
                    │  WITH AUTONOMOUS TOOL CALLING     │
                    │  + TOKEN STREAMING                │
                    │                                   │
                    │  Receives: strategy + coaching +  │
                    │  sentiment + profile + history     │
                    │                                   │
                    │  Tools (LLM decides when):        │
                    │  • lookup_pricing                 │
                    │  • compare_competitor             │
                    │  • calculate_roi                  │
                    │  • suggest_automations            │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Output Guardrail    │  ← Auto-regeneration loop:
                    │  (20b)               │    if flagged → regen + show
                    └──────────┬──────────┘
                               │
                    All fields ≥ 0.7 confidence?
                         │              │
                        no             yes → Deferred Close (button appears)
                         │                         │
                    Continue                 User clicks button
                    talking                        │
                                   ┌───────────────▼──────────────┐
                                   │     Close Sequence             │
                                   │  • Board Gen (120b)            │
                                   │  • Plan + Payment              │
                                   │  • Follow-up Email (120b)      │
                                   │  • Evaluator (120b)            │
                                   │  • Self-Improvement (20b)      │
                                   │    → persisted to disk          │
                                   │  • Context Summarizer (20b)    │
                                   └────────────────────────────────┘
```

Up to **8 LLM calls per user message** (7 always + 1 periodic coach) plus 3 non-LLM security layers. Three of those calls run in parallel via `ThreadPoolExecutor`, cutting ~2s off every turn.

### Why Each Agent Exists

| # | Agent | Model | Why It's Separate |
|---|-------|-------|-------------------|
| 1 | **Input Guardrail** | 20b | Security must be isolated — can't let the main LLM judge its own inputs |
| 2 | **Sentiment Analyzer** | 20b | Runs independently so the strategy planner gets mood data without biasing extraction |
| 3 | **Extractor** | 20b | Structured JSON extraction needs temperature 0.0; mixing with conversation causes hallucinated fields |
| 4 | **Lead Scorer** | 20b | Quantitative scoring (0–100) requires a different reasoning frame; separating prevents score inflation |
| 5 | **Strategy Planner** | 20b | Meta-reasoning: thinks about WHAT to say before the conversation LLM thinks about HOW |
| 6 | **Live Coach** | 20b | Mid-conversation quality monitor — like a sales manager listening in, feeds tactical corrections |
| 7 | **Conversation LLM** | 120b | Main agent — needs the largest model for natural language, tool-calling, and objection handling |
| 8 | **Output Guardrail** | 20b | Post-hoc safety check; auto-regenerates flagged responses rather than just logging |
| 9 | **Board Generator** | 120b | Generating realistic, industry-specific board structures requires strong reasoning |
| 10 | **Email Generator** | 120b | Professional email writing is a distinct skill from sales conversation |
| 11 | **Evaluator** | 120b | Post-conversation analysis needs objectivity — the conversation model shouldn't grade itself |
| 12 | **Self-Improvement** | 20b | Turns evaluation scores into actionable prompt improvements; persists to disk across sessions |
| 13 | **Context Summarizer** | 20b | Compresses old messages instead of dropping them; runs only when context exceeds limits |

### Multi-Model Strategy

| Model | Roles | Why |
|-------|-------|-----|
| `openai/gpt-oss-120b` | Conversation + Tools, Board Gen, Evaluator, Email | 120B params — stronger reasoning for tool-calling decisions and nuanced conversation |
| `openai/gpt-oss-20b` | Extractor, Guardrails, Lead Scorer, Sentiment, Strategy, Live Coach, Self-Improvement, Summarizer | 20B params, ~2x throughput — fast enough for parallel execution without sacrificing accuracy |

Multiple Groq API keys can be provided via `GROQ_API_KEYS` (comma-separated) — the system auto-rotates on rate limits. If all keys are exhausted, it falls back to local Ollama models via OpenAI-compatible API.

---

## Customer Journey (5-Phase GTM Flow)

### 1. Contact
Agent greets and asks about industry and team. The Extractor silently begins collecting structured data. Lead score initializes.

### 2. Demo
Personalized demo based on extracted context. If the prospect mentions a competitor, the agent autonomously calls `compare_competitor` with data from a structured knowledge base covering 7 competitors. If they ask about efficiency, it calls `suggest_automations` with their team type.

### 3. Qualification
Agent naturally collects remaining details through conversation. Each extracted field has a confidence score (0.0–1.0); low-confidence fields are prioritized in follow-up questions. Lead score updates after every turn.

### 4. Use-Case Setup (Deferred Close)
When all fields reach ≥0.7 confidence, a "Set up workspace" button appears — but the agent does NOT auto-close. The prospect can keep chatting, object to pricing, compare tools, or negotiate. The agent handles objections using tool calls (ROI calculation, pricing lookup, competitor comparison). Close only happens when the prospect explicitly opts in.

### 5. Close & Payment
When the prospect clicks the button:
- **Board**: AI-generated monday.com board with custom columns and example items tailored to their use case
- **Plan**: Auto-selected plan with per-seat pricing (validated against seat limits)
- **Payment**: Mock Stripe checkout page with session ID
- **Email**: AI-generated follow-up email for the prospect's team
- **Analytics**: Conversation quality scorecard grading 5 metrics
- **Learning**: Self-improvement suggestions persisted to disk for future sessions

### How Input, Decisions, and Transitions Work

- **Input handling**: Every message passes through input sanitization → regex injection filter → context-aware LLM guardrail before reaching the pipeline
- **Decisions**: The Strategy Planner autonomously decides the approach (discover / educate / close / handle_objection) based on profile completeness, sentiment, and lead score. The Conversation LLM autonomously decides which tools to call. No hardcoded if/else chains drive the conversation.
- **Transitions**: Phase progression is driven by field confidence scores, not hardcoded rules. The agent naturally moves forward as it learns more about the prospect.

---

## AI Prompting & Logic

### Prompt Structure

13 specialized prompts power the 14 agent roles. The main conversation prompt is **dynamically assembled** each turn, injecting:

- Current prospect profile summary (what's known and what's missing)
- Sentiment analysis result (emotional state and intensity)
- Strategy planner output (next move and priority)
- Live coaching notes (tactical adjustments, fires every 3 turns)
- Learned improvements from previous sessions (persisted to disk)

This means the same prompt template produces different agent behavior depending on conversation state — no scripted flows.

| # | Prompt | Model | Purpose |
|---|--------|-------|---------|
| 1 | `build_conversation_prompt()` | 120b | Dynamic — combines strategy + coaching + sentiment + profile + learned improvements |
| 2 | `EXTRACTOR_PROMPT` | 20b | Structured extraction with per-field confidence scores |
| 3 | `INPUT_GUARD_PROMPT` | 20b | Context-aware input safety classification |
| 4 | `OUTPUT_GUARD_PROMPT` | 20b | Output quality check; triggers auto-regeneration if flagged |
| 5 | `EVALUATOR_PROMPT` | 120b | Scores conversation on 5 quality metrics |
| 6 | `LEAD_SCORE_PROMPT` | 20b | Conservative 0–100 lead scoring with signal detection |
| 7 | `SELF_IMPROVE_PROMPT` | 20b | Generates actionable improvements from evaluation results |
| 8 | `FOLLOW_UP_EMAIL_PROMPT` | 120b | Generates personalized onboarding email |
| 9 | `BOARD_GEN_PROMPT` | 120b | Creates tailored board definition (columns + items) |
| 10 | `SENTIMENT_PROMPT` | 20b | Classifies prospect emotional state per message |
| 11 | `STRATEGY_PLANNER_PROMPT` | 20b | Autonomous meta-reasoning: fastest path to signed deal |
| 12 | `LIVE_COACH_PROMPT` | 20b | Mid-conversation quality monitor (sales manager listening in) |
| 13 | `CONTEXT_SUMMARIZER_PROMPT` | 20b | Compresses older messages to preserve information within context limits |

### Agentic Tool Calling

The Conversation LLM has 4 tools via Groq's native function calling API. It decides autonomously which to call and when — the code provides tools, the LLM chooses when to use them. All tool calls are visible in the "Agent Reasoning" panel.

| Tool | Purpose | Typical trigger |
|------|---------|-----------------|
| `lookup_pricing` | Plan details, features, seat limits | User asks about cost |
| `compare_competitor` | Structured comparison from knowledge base | User mentions Jira, Asana, etc. |
| `calculate_roi` | ROI analysis vs current tools | User mentions what they're paying |
| `suggest_automations` | Team-specific automation recipes | User asks about efficiency |

The competitor knowledge base (`competitive_intel.json`) covers 7 competitors with strengths, weaknesses, win strategies, killer points, and objection handlers.

### Branching

Branching happens at six levels — no hardcoded if/else chains:

1. **Strategy**: Autonomous planner reasons about the fastest path to close
2. **Coaching**: Live coach overrides or refines strategy based on conversation quality
3. **Tools**: LLM autonomously decides which tools to call based on context
4. **Confidence**: Low-confidence fields are nudged in the conversation prompt
5. **Sentiment**: Agent adapts tone — empathy for frustration, data for skepticism
6. **Safety**: Input guardrail routes injections; output guardrail auto-regenerates bad responses

### Data Capture

The Extractor LLM runs every turn and outputs structured JSON with confidence scores:

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

Fields below 0.7 confidence are flagged, and the conversation naturally steers toward filling them. Regex fallback extraction catches plan/seat mentions the LLM may miss.

---

## Security (Defense-in-Depth)

Six-layer architecture with no single point of failure:

| Layer | What It Does |
|-------|-------------|
| **Input Sanitization** | Unicode normalization, control char stripping, 2000-char limit, invisible char removal |
| **Regex Pre-filter** | 12+ patterns catch known injection attacks (role hijacking, prompt extraction, delimiter attacks) before the LLM sees the message |
| **LLM Input Guardrail** | 20B model classifies messages as safe / injection / off-topic with conversation context |
| **System Prompt Armor** | Anti-extraction instructions at highest priority in every system prompt; user input treated as untrusted |
| **Output Sanitization** | Strips script tags, event handlers, javascript: URLs from all LLM output |
| **LLM Output Guardrail** | Catches hallucinated features and unauthorized pricing — auto-regenerates flagged responses |

Additional: rate limiting (15 msg/60s per session), tool argument validation with range clamping, pricing locked to official tiers ($9/$12/$19).

---

## Assumptions & Shortcuts

| Area | What's Mocked | Production Path |
|------|---------------|-----------------|
| **monday.com API** | `monday_mock.py` generates board JSON via LLM, doesn't call the real API | monday.com REST/GraphQL API with OAuth2 |
| **Stripe Payments** | `stripe_mock.py` returns a mock checkout URL and session ID | Stripe Checkout Sessions API in test mode |
| **User Auth** | No auth — each browser session is independent via Streamlit session state | OAuth2 / SSO with database-backed sessions |
| **Persistence** | Conversation state in `st.session_state` (lost on refresh); self-improvement persists to `.learned_improvements.json` | Redis or database for session state |
| **LLM Provider** | Groq free tier with key rotation + Ollama local fallback | Production API account with proper rate limits |
| **Competitive Intel** | Static JSON file with 7 competitors, updated manually | Live competitive intelligence source or CRM sync |
| **Email Delivery** | Follow-up email generated and displayed, not sent | SendGrid, Mailgun, or monday.com email automation |
| **CRM Integration** | Lead scoring displayed but not synced to a CRM | Push qualified leads + scores to monday.com CRM via API |

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

---

## Evaluation Criteria

### 🎯 Execution
Working prototype: streaming chat, autonomous tool calling, 5 GTM phases, AI-generated boards, mock Stripe checkout, follow-up email, auto-demo mode. Runs locally with one command.

### 🤖 AI Craftsmanship
- Agentic tool calling — the LLM decides what to do, not hardcoded logic
- 13 specialized prompts across 2 model sizes powering 14 agent roles
- Up to 8 LLM calls per turn, 3 running in parallel
- Autonomous meta-reasoning — strategy planner thinks like a sales rep, not a script
- Live coaching loop — mid-conversation quality monitor feeds real-time adjustments
- Sentiment-aware responses that adapt tone to the prospect's emotional state
- Guardrail-in-the-loop — auto-regenerates bad responses, not just logs them
- Self-improvement loop that persists across sessions
- Competitor intelligence from structured knowledge base with win strategies and objection handlers

### 🧩 System Design
- Parallel LLM execution via ThreadPoolExecutor
- Multi-model strategy: 120b for quality-critical tasks, 20b for speed-critical tasks
- Tool-calling architecture scales to new tools without code changes
- Defense-in-depth security — 6 layers, no single point of failure
- API key pool with Ollama fallback for production resilience
- Deferred close mirrors real sales dynamics
- Context compression prevents information loss at scale
- Centralized configuration (single source of truth for pricing, plans, constants)

### 💬 Clarity
- This README documents the architecture, decisions, and trade-offs
- Agent Reasoning panel shows strategy and tool calls in real time
- Sentiment trend visible in sidebar
- Auto-demo mode lets reviewers see everything in 30 seconds
