# monday.com AI Sales Agent

> An autonomous, agentic AI sales agent that guides prospects from first contact to payment — making its own decisions about when to look up pricing, compare competitors, calculate ROI, and suggest automations.

## Quick Start

```bash
pip install -r requirements.txt
# Single key (basic):
echo "GROQ_API_KEY=gsk_your_key_here" > .env
# Multiple keys (recommended — auto-rotates on rate limits):
echo "GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3" >> .env
python -m streamlit run app.py
```

**For reviewers:** Click **"Auto Demo"** in the sidebar to watch the entire 5-phase GTM flow run live in ~30 seconds — no typing needed. See the [Walkthrough](#walkthrough) section for details.

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
                    │  Strategy Planner    │  ← AUTONOMOUS META-REASONING
                    │  (20b)               │    "fastest path to signed deal"
                    │  Sees: profile +     │    visible in Agent Reasoning panel
                    │  sentiment + score   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Live Coach (20b)    │  ← NEW: fires every 3 turns
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

### Why Each Agent Exists

| # | Agent | Model | Why It's a Separate Agent |
|---|-------|-------|--------------------------|
| 1 | **Input Guardrail** | 20b | Security must be isolated — can't let the main LLM judge its own inputs. Separate model = separate attack surface. |
| 2 | **Sentiment Analyzer** | 20b | Emotional intelligence runs independently so the strategy planner gets mood data without biasing extraction. |
| 3 | **Extractor** | 20b | Structured data extraction needs different temperature (0.0) and format (JSON) than conversation. Mixing them causes hallucinated fields. |
| 4 | **Lead Scorer** | 20b | Quantitative scoring (0-100) requires a different reasoning frame than qualitative conversation. Separating it prevents score inflation from politeness. |
| 5 | **Strategy Planner** | 20b | Meta-reasoning layer — thinks about WHAT to say before the conversation LLM thinks about HOW to say it. Like a chess player planning moves vs. executing them. |
| 6 | **Live Coach** | 20b | Mid-conversation quality monitor. A real sales team has a manager listening in. This agent reviews every 3 turns and feeds tactical corrections. |
| 7 | **Conversation LLM** | 120b | The main agent — needs the biggest model for natural language, tool-calling decisions, and nuanced objection handling. |
| 8 | **Output Guardrail** | 20b | Post-hoc safety check catches hallucinations the conversation LLM might produce. Auto-regenerates rather than just logging. |
| 9 | **Board Generator** | 120b | Generating realistic, industry-specific board structures requires strong reasoning about column types, example items, and workflows. |
| 10 | **Email Generator** | 120b | Professional email writing is a distinct skill from sales conversation. Separate prompt = better formatting and tone. |
| 11 | **Evaluator** | 120b | Post-conversation analysis needs objectivity — the model that ran the conversation shouldn't grade itself. |
| 12 | **Self-Improvement** | 20b | Turns evaluation scores into actionable prompt improvements. Persists to disk, creating a quality flywheel across sessions. |
| 13 | **Context Summarizer** | 20b | Compresses old messages instead of dropping them. Runs lazily (only when context exceeds limits) to preserve information. |

### Multi-Model Strategy

| Model | Role | Why |
|-------|------|-----|
| `openai/gpt-oss-120b` | Conversation + Tools, Board Gen, Evaluator, Email | 120B params, 500 t/s — smarter tool decisions and faster streaming than 70B alternatives |
| `openai/gpt-oss-20b` | Extractor, Guardrails, Lead Scorer, Sentiment, Strategy, Live Coach, Self-Improvement, Summarizer | 20B params, 1000 t/s — 2.5x more capable than 8B at near-double speed |

### Parallel Execution — Why It Matters

Sentiment, Extraction, and Lead Scoring are **independent** — none needs the other's output. Running them sequentially wastes ~2 seconds per turn. Using `ThreadPoolExecutor`, all three fire simultaneously:

```
BEFORE (sequential):                 AFTER (parallel):
  sentiment  ~1s                       sentiment ─┐
  extractor  ~1s                       extractor ─┼─ ~1s total ──► strategy ~1s
  lead_score ~1s                       lead_score─┘
  strategy   ~1s                     
  ─────────────                        ─────────────
  Total: ~4s                           Total: ~2s
```

This is the kind of optimization that shows you think about production latency, not just correctness.

---

## What Makes This Different

### 1. Agentic Tool Calling (not just a chatbot)

Most candidates will build a chatbot that follows a script. This agent **makes its own decisions**. Using Groq's native function calling API, the conversation LLM has 4 tools:

| Tool | What it does | When the agent calls it |
|------|-------------|----------------------|
| `lookup_pricing` | Returns plan details and features | User asks about cost |
| `compare_competitor` | Structured comparison from knowledge base | User mentions Jira, Asana, Trello, etc. |
| `calculate_roi` | ROI analysis comparing current tools vs monday.com | User mentions what they're paying |
| `suggest_automations` | Team-specific automation recipes | User asks about efficiency |

The agent decides **autonomously** which tool to call and when. All tool calls are visible in the "Agent Reasoning" panel so reviewers can see the decision-making.

**Why this matters:** This is the difference between a chatbot and an agent. The LLM isn't just responding — it's *acting*.

### 2. Competitor Intelligence + ROI Calculator

The agent has a structured knowledge base (`competitive_intel.json`) covering 7 competitors — Jira, Asana, Trello, ClickUp, Notion, Smartsheet, and Excel/Google Sheets — with:
- Strengths, weaknesses, and typical pricing for each
- Win strategies: key messages, killer points, and specific objection handlers
- Automatic alias resolution (e.g., "spreadsheets" → Excel/Google Sheets)

When a prospect says "We pay $1000/month for Jira", the agent autonomously calls `calculate_roi` and presents a personalized analysis: current costs, monday.com costs, consolidation savings, and productivity gains.

### 3. Real-Time Lead Scoring

A dedicated 8b model computes a 0-100 lead score after every turn, detecting:
- **Buying signals**: mentions budget, timeline, asks about pricing, compares competitors
- **Negative signals**: "just looking", budget concerns, low engagement
- **Label**: Hot (70-100), Warm (40-69), Cold (0-39)

Displayed in the sidebar in real-time. In production, this would feed into CRM prioritization.

### 4. Self-Improving Agent

After each conversation evaluation, a self-improvement LLM generates actionable suggestions:
- "Ask about current tools within the first 2 messages"
- "When mentioning automations, always give a specific recipe"

These are **persisted to disk** and automatically **injected into the next conversation's prompt**. Over multiple sessions, the agent measurably improves. This creates a flywheel: each call makes the next one better.

### 5. Deferred Close (Mirrors Real Sales)

When all fields reach ≥70% confidence, the system does NOT auto-close. Instead:
- The agent naturally offers to set up the workspace
- A "Set up my workspace" button appears
- The prospect can keep chatting — object to pricing, compare tools, ask questions
- The agent handles objections using tool calls (ROI calculation, competitor comparison)
- Close only happens when the prospect explicitly opts in

### 6. Real Token Streaming

Responses stream token-by-token using Groq's streaming API. When the agent calls tools first, tool execution happens non-streaming (visible in real-time in the reasoning panel), then the final response streams live. This is how production AI apps work.

### 7. Sentiment Analysis + Emotional Intelligence

A dedicated 8b model classifies each prospect message's emotional state (excited / positive / neutral / skeptical / frustrated / objecting) with intensity scoring. The sidebar shows the current mood with emoji trend visualization. The strategy planner uses this to adapt the agent's approach — empathize with frustration, match excitement, handle skepticism with data.

### 8. Agent Strategy Planner (Autonomous Meta-Reasoning)

Before each response, a planning LLM acts as the "strategic brain" of a top sales rep. It reasons through: "Can I recommend a solution yet? What stage are we in? Am I being efficient?" Its output drives the conversation LLM's behavior — not a script, genuine autonomous reasoning about the fastest path to a signed deal. Visible in the **Agent Reasoning** panel.

### 9. Live Coaching Agent (Sales Manager Listening In)

Every 3 turns, a lightweight LLM reviews the recent exchanges like a sales manager listening to a live call. It produces one tactical coaching note — "You're asking too many questions without showing value" or "They're ready, stop discovering and close" — that gets injected into the conversation LLM's next prompt. This creates a **live feedback loop**, not just post-mortem evaluation.

### 10. Guardrail-in-the-Loop (Auto-Correction)

When the output guardrail detects hallucinated features or unauthorized pricing, the system doesn't just log it — it **auto-regenerates** the response with correction instructions. The corrected response replaces the original, and the correction is noted in the guardrail log. Production-grade AI safety.

### 11. Intelligent Context Compression

Instead of brute-force truncation when context exceeds limits, the 20b model **summarizes** older messages into a paragraph preserving all key facts (industry, team, pain points, pricing discussed, objections handled). Recent messages stay in full. This means the agent never "forgets" earlier conversation — it compresses intelligently.

### 12. Parallel LLM Execution

Sentiment, Extraction, and Lead Scoring run **concurrently** via `ThreadPoolExecutor` — cutting ~2 seconds off every turn. This is a production-grade optimization that shows awareness of real-world latency constraints in multi-LLM systems.

### 13. API Key Pool with Auto-Rotation + Ollama Fallback

Supports multiple Groq API keys via `GROQ_API_KEYS` (comma-separated). When a key hits a rate limit, the system automatically rotates to the next available key with zero downtime. If ALL Groq keys are exhausted, it seamlessly falls back to **local Ollama** models via OpenAI-compatible API. Key pool status is visible in the sidebar debug panel.

### 14. Auto-Demo Mode

Click "Auto Demo" and watch the full flow run automatically: 4 pre-scripted prospect messages play through the entire pipeline — strategy planning, tool calls, streaming, extraction, scoring, sentiment tracking, qualification, close, board gen, email, analytics, and self-improvement — all visible in 30 seconds.

---

## The 5-Phase GTM Journey

### 1. Contact
Agent greets, asks about industry and team. Extractor starts collecting data silently. Lead score initializes.

### 2. Demo
Agent delivers a personalized demo. If the prospect mentions a competitor, the agent autonomously calls `compare_competitor`. If they mention automations, it calls `suggest_automations` with their team type.

### 3. Qualification
Agent naturally collects remaining details. Low-confidence fields are nudged. Lead score updates in real-time.

### 4. Use-Case Setup (Deferred Close)
When all fields hit ≥70% confidence, the "Set up workspace" button appears. The prospect can keep chatting — the agent handles objections with tool calls (ROI calculation, pricing lookup, competitor comparison).

### 5. Close & Payment
When the prospect clicks the button:
- AI-generated board with custom columns and items
- Auto-selected plan with per-seat pricing
- Mock Stripe checkout with session ID
- AI-generated follow-up email
- Conversation quality scorecard (5 metrics)
- Self-improvement suggestions stored for next session

---

## AI Architecture Deep Dive

### Dual-LLM with Tool Calling

**The evolution:** Single LLM → Dual LLM → Agentic Tool Calling.

1. **Single LLM** (what most candidates build): One model talks AND decides AND extracts JSON. Fragile.
2. **Dual LLM** (our previous version): Conversation and extraction separated. Better, but the code still decides what happens.
3. **Agentic Tool Calling** (what we built): The LLM itself decides when to call tools. The code provides tools; the LLM chooses when to use them. This is true autonomy.

### 13 Specialized Prompts

| # | Prompt | Model | Purpose |
|---|--------|-------|---------|
| 1 | `build_conversation_prompt()` | 120b | Dynamic agent with tool-calling + strategy + coaching + learned improvements |
| 2 | `EXTRACTOR_PROMPT` | 20b | Structured extraction with confidence scores |
| 3 | `INPUT_GUARD_PROMPT` | 20b | Context-aware input safety classification |
| 4 | `OUTPUT_GUARD_PROMPT` | 20b | Check output quality, auto-regenerate if flagged |
| 5 | `EVALUATOR_PROMPT` | 120b | Score conversation on 5 metrics |
| 6 | `LEAD_SCORE_PROMPT` | 20b | Conservative lead scoring with signal detection |
| 7 | `SELF_IMPROVE_PROMPT` | 20b | Generate improvement suggestions from evaluation |
| 8 | `FOLLOW_UP_EMAIL_PROMPT` | 120b | Generate onboarding email |
| 9 | `BOARD_GEN_PROMPT` | 120b | Create tailored board definition |
| 10 | `SENTIMENT_PROMPT` | 20b | Classify prospect emotional state per message |
| 11 | `STRATEGY_PLANNER_PROMPT` | 20b | Autonomous meta-reasoning: fastest path to signed deal |
| 12 | `LIVE_COACH_PROMPT` | 20b | Mid-conversation quality monitor (sales manager) |
| 13 | `CONTEXT_SUMMARIZER_PROMPT` | 20b | Compress older messages instead of truncating |

### Confidence Scoring

Each extracted field has a confidence score (0.0-1.0). Qualification triggers at ≥0.7 across all 4 fields. This handles uncertainty explicitly — no guessing.

### Security & Guardrails (Defense-in-Depth)

Six-layer security architecture — no single point of failure:

| Layer | Type | What It Does |
|-------|------|-------------|
| **Input Sanitization** | Pre-processing | Unicode normalization, control char stripping, 2000-char limit, invisible char removal |
| **Regex Pre-filter** | Pattern matching | 12+ regex patterns catch known injection attacks (role hijacking, prompt extraction, delimiter attacks, jailbreaks) BEFORE the LLM sees the message |
| **LLM Input Guardrail** | AI classification | 20B model classifies messages as safe/injection/off-topic with social engineering awareness |
| **System Prompt Armor** | Prompt hardening | Anti-extraction instructions at highest priority in every system prompt, user input treated as untrusted |
| **Output Sanitization** | HTML/XSS protection | Strips script tags, event handlers, javascript: URLs, and dangerous HTML from all LLM output before rendering |
| **LLM Output Guardrail** | AI quality check | Catches hallucinated features, unauthorized pricing, competitor disparagement — auto-regenerates flagged responses |

Additional protections:
- **Rate limiting**: 15 messages per 60-second window per session prevents API abuse
- **Tool argument validation**: All tool call arguments are validated, sanitized, and range-clamped before execution
- **Pricing rules**: Agent can ONLY quote official tiers ($9/$12/$19). Cannot invent discounts.

### Branching Logic

Branching happens at six levels:
1. **Strategy-level**: Autonomous meta-reasoning planner decides the approach (discover / educate / close / handle_objection / reassure)
2. **Coaching-level**: Live coach periodically overrides or refines the strategy based on conversation quality
3. **Tool-level**: The LLM autonomously decides which tools to call based on conversation context
4. **Confidence-level**: Low-confidence fields are nudged in the conversation prompt
5. **Sentiment-level**: Agent adapts tone based on detected emotional state (empathy for frustration, energy for excitement)
6. **Safety-level**: Input guardrail routes injection vs off-topic; output guardrail auto-regenerates bad responses

No hardcoded if/else chains. The agent adapts to any conversation path.

### Per-Turn Pipeline (what happens on every message)

```
User message
    │
    ├── Input Sanitization           → clean, normalize, length limit
    ├── Rate Limiter                 → flood protection (15 msg/min)
    ├── Regex Pre-filter             → fast pattern injection check
    ├── LLM Input Guardrail (20b)    → context-aware injection/off-topic detection
    │
    ├── ┌─ Sentiment Analysis (20b)  ─┐
    ├── │  Extractor (20b)            ├── PARALLEL (ThreadPoolExecutor)
    ├── └─ Lead Scorer (20b)         ─┘
    │
    ├── Strategy Planner (20b)       → autonomous meta-reasoning
    ├── Live Coach (20b)             → tactical adjustment (every 3 turns)
    │
    ├── Conversation LLM (120b)      → tool calls + streamed response
    │     ├── [optional] Tool arg validation + execution
    │     └── Token-by-token streaming
    │
    ├── Output Sanitization          → strip XSS/dangerous HTML
    └── LLM Output Guardrail (20b)   → safe / auto-regenerate
```

That's **up to 8 LLM calls per user message** (7 always + 1 periodic coach) plus 3 non-LLM security layers. Three of those calls run in parallel. This is not a chatbot — it's a hardened, latency-optimized AI pipeline.

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
├── config.py               # Shared constants and utility functions
├── monday_mock.py          # Mock board generation
├── stripe_mock.py          # Mock payment links
├── requirements.txt        # Dependencies
└── .env                    # API keys (not committed)
```

---

## Walkthrough

Instead of a pre-recorded Loom video, this project includes an **Auto Demo mode** — click "Auto Demo" in the sidebar and watch the full 5-phase GTM flow run live in ~30 seconds. This is better than a recording because you see the real system executing in real-time: strategy planning, tool calls, streaming, extraction, scoring, sentiment tracking, qualification, board generation, payment, follow-up email, analytics, and self-improvement — all happening on your machine.

---

## Assumptions & Shortcuts

| Area | What's Mocked / Simplified | Production Path |
|------|-----------------------------|-----------------|
| **monday.com API** | `monday_mock.py` generates board JSON with an LLM (columns + items), but doesn't call the real monday.com API. | Replace with monday.com REST/GraphQL API using OAuth2 |
| **Stripe Payments** | `stripe_mock.py` returns a mock checkout URL and session ID. No real payment processing. | Replace with Stripe Checkout Sessions API in test mode |
| **User Authentication** | No user auth — each browser session is independent via Streamlit session state. | Add OAuth2 / SSO, persist sessions to a database |
| **Persistence** | Conversation state lives in `st.session_state` (lost on page refresh). Self-improvement suggestions persist to `.learned_improvements.json`. | Use Redis or a database for session state |
| **LLM Provider** | Uses Groq's free API tier with auto-rotation across multiple keys. Falls back to local Ollama if Groq is unavailable. | Production Groq/OpenAI/Anthropic account with proper rate limits |
| **Competitive Intelligence** | Static JSON file (`competitive_intel.json`) with 7 competitors. Updated manually. | Sync with a live competitive intelligence source or CRM data |
| **Email Delivery** | Follow-up email is generated and displayed, but not actually sent. | Integrate with SendGrid, Mailgun, or monday.com's email automation |
| **CRM Integration** | Lead scoring and profile data are displayed but not synced to a CRM. | Push qualified leads + scores to monday.com CRM via API |

---

## Evaluation Criteria — How This Addresses Each

### 🎯 Execution
Working prototype with streaming chat, autonomous tool calling, 5 GTM phases, AI-generated boards, mock Stripe, follow-up email, and auto-demo mode. Runs locally with one command.

### 🤖 AI Craftsmanship
- **Agentic tool calling** — the LLM decides what to do, not hardcoded logic
- **13 specialized prompts** across 2 model sizes powering **14 distinct agent roles**
- **Up to 8 LLM calls per turn** — 3 running in parallel
- **Autonomous meta-reasoning** — strategy planner thinks like a top sales rep, not a script
- **Live coaching loop** — mid-conversation quality monitor feeds real-time adjustments
- **Sentiment awareness** — detects emotional state and adapts tone
- **Real token streaming** — production-grade streaming via Groq API
- **Guardrail-in-the-loop** — auto-regenerates bad responses, not just logs them
- **Intelligent context compression** — summarizes old messages instead of truncating
- **Self-improvement loop** that persists across sessions
- **Competitor intelligence** from structured JSON — 7 competitors with win strategies and objection handlers
- **ROI calculator** for data-driven objection handling

### 🧩 System Design
- **Parallel LLM execution** — 3 independent agents run concurrently via ThreadPoolExecutor
- Multi-model strategy (120b for quality, 20b for speed) — each call optimized for latency vs quality
- Tool-calling architecture scales to new tools without code changes
- Strategy planner as autonomous reasoner — not a scripted flow, genuine meta-reasoning
- Live coaching creates a feedback loop during conversation, not just post-mortem
- Self-improvement creates a quality flywheel persisted to disk
- Context summarization prevents information loss at scale
- Deferred close mirrors real sales dynamics
- Defense-in-depth security — 6 layers, no single point of failure
- Groq key pool with Ollama fallback — production resilience pattern
- Lead scoring ready for CRM integration

### 💬 Clarity
- This README documents every architectural decision and why
- Agent Reasoning panel shows strategy + tool calls in real-time
- Sentiment trend visible in sidebar with emoji visualization
- Guardrail and tool call logs provide full transparency
- Auto-demo mode lets reviewers see everything in 30 seconds
