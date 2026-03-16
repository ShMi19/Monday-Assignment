"""Agentic tool definitions and execution layer.

This module defines the tools the AI agent can autonomously decide to call
during conversation, plus a competitor knowledge base and ROI calculator.
The LLM sees these as function schemas and chooses when to invoke them.
"""

import json
import os

from config import PLAN_PRICING, plan_for_seats

# ═══════════════════════════════════════════════════════════════════════════
# Competitor Intelligence Knowledge Base — loaded from competitive_intel.json
# ═══════════════════════════════════════════════════════════════════════════

_INTEL_PATH = os.path.join(os.path.dirname(__file__), "competitive_intel.json")

def _load_intel() -> dict:
    try:
        with open(_INTEL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"competitors": {}, "monday_com": {}}

_INTEL = _load_intel()
COMPETITOR_DATA = _INTEL.get("competitors", {})
MONDAY_INFO = _INTEL.get("monday_com", {})

MONDAY_PRICING = PLAN_PRICING

AUTOMATION_TEMPLATES = {
    "sales": [
        {"trigger": "When a deal moves to 'Won'", "action": "Notify the account manager and create an onboarding task", "benefit": "Zero hand-off delays"},
        {"trigger": "When a lead is inactive for 3 days", "action": "Send an automated follow-up email", "benefit": "No leads fall through the cracks"},
        {"trigger": "When deal value exceeds $10K", "action": "Assign to senior rep and notify VP of Sales", "benefit": "Enterprise deals get premium attention"},
    ],
    "engineering": [
        {"trigger": "When a bug status changes to 'Critical'", "action": "Notify the on-call engineer via Slack and PagerDuty", "benefit": "Instant incident response"},
        {"trigger": "When all sprint items are 'Done'", "action": "Move sprint to completed and notify PM", "benefit": "Automated sprint closure"},
        {"trigger": "When a PR is merged in GitHub", "action": "Update the linked item status to 'In Review'", "benefit": "Code and project board always in sync"},
    ],
    "marketing": [
        {"trigger": "When a campaign launch date arrives", "action": "Notify the content team and change status to 'Live'", "benefit": "Never miss a launch"},
        {"trigger": "When an asset is approved", "action": "Move to 'Ready to Publish' and notify social media team", "benefit": "Streamlined approval workflow"},
        {"trigger": "When budget exceeds 80% of allocation", "action": "Alert the marketing director", "benefit": "Proactive budget management"},
    ],
    "hr": [
        {"trigger": "When a new hire start date is 7 days away", "action": "Create onboarding checklist and assign IT setup tasks", "benefit": "Every new hire gets a smooth start"},
        {"trigger": "When a PTO request is submitted", "action": "Notify the manager and update the team calendar", "benefit": "Transparent time-off management"},
        {"trigger": "When probation period ends", "action": "Create a performance review task for the manager", "benefit": "Never miss a review deadline"},
    ],
    "operations": [
        {"trigger": "When a vendor contract is expiring in 30 days", "action": "Notify procurement and create a renewal task", "benefit": "No surprise contract lapses"},
        {"trigger": "When inventory drops below threshold", "action": "Create a reorder request and notify the supplier", "benefit": "Automated inventory management"},
        {"trigger": "When a compliance task is overdue", "action": "Escalate to department head and flag as critical", "benefit": "Stay audit-ready"},
    ],
    "project management": [
        {"trigger": "When a task is overdue", "action": "Notify the assignee and their manager", "benefit": "Proactive deadline management"},
        {"trigger": "When a dependency is completed", "action": "Unblock the next task and notify the assignee", "benefit": "No bottleneck delays"},
        {"trigger": "When project reaches 90% completion", "action": "Notify stakeholders and schedule a review meeting", "benefit": "Smooth project closure"},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Tool Schemas (for Groq function calling)
# ═══════════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_pricing",
            "description": "Look up monday.com pricing details for a specific plan tier. Call this when the user asks about pricing or you need to quote a specific plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "enum": ["Starter", "Standard", "Pro"],
                        "description": "The plan tier to look up",
                    }
                },
                "required": ["plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_competitor",
            "description": "Get a detailed comparison between monday.com and a competitor, including win strategy and objection handlers. Call this when the user mentions they're using or considering any project management tool, spreadsheets, or competitor product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competitor": {
                        "type": "string",
                        "description": "The competitor name (jira, asana, trello, clickup, notion, smartsheet, excel, google sheets, spreadsheets)",
                    }
                },
                "required": ["competitor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_roi",
            "description": "Calculate the ROI of switching from a competitor to monday.com. Call this when the user mentions what they're currently paying for another tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_tool": {"type": "string", "description": "The tool they're currently using"},
                    "current_monthly_cost": {"type": "number", "description": "Their current monthly cost in USD"},
                    "team_size": {"type": "integer", "description": "Number of users/seats"},
                    "num_tools_replaced": {
                        "type": "integer",
                        "description": "Estimated number of separate tools monday.com would replace (default 2)",
                    },
                },
                "required": ["current_tool", "current_monthly_cost", "team_size"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_automations",
            "description": "Get automation recipe suggestions tailored to a specific team type. Call this when discussing how monday.com can help a team be more efficient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_type": {
                        "type": "string",
                        "description": "The team type (sales, engineering, marketing, hr, operations, project management)",
                    }
                },
                "required": ["team_type"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Tool Execution
# ═══════════════════════════════════════════════════════════════════════════

def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool call and return the result as a string for the LLM."""
    if name == "lookup_pricing":
        return _exec_lookup_pricing(arguments)
    elif name == "compare_competitor":
        return _exec_compare_competitor(arguments)
    elif name == "calculate_roi":
        return _exec_calculate_roi(arguments)
    elif name == "suggest_automations":
        return _exec_suggest_automations(arguments)
    return json.dumps({"error": f"Unknown tool: {name}"})


def _exec_lookup_pricing(args: dict) -> str:
    plan_name = args.get("plan", "Standard")
    plan = MONDAY_PRICING.get(plan_name, MONDAY_PRICING["Standard"])
    return json.dumps({
        "plan": plan_name,
        "price_per_seat": f"${plan['price']}/seat/month",
        "min_seats": plan["min_seats"],
        "max_seats": plan["max_seats"],
        "seat_limit_note": f"This plan supports {plan['min_seats']}–{plan['max_seats']} seats. If the prospect needs more than {plan['max_seats']}, they MUST use a higher tier.",
        "features": plan["features"],
        "all_plans_summary": "Starter $9 (1–10 seats) | Standard $12 (3–50 seats) | Pro $19 (3+ unlimited)",
        "note": "These are the only official pricing tiers. No custom discounts available.",
    })


_COMPETITOR_ALIASES = {
    "google sheets": "excel_sheets",
    "sheets": "excel_sheets",
    "excel": "excel_sheets",
    "spreadsheets": "excel_sheets",
    "spreadsheet": "excel_sheets",
}


def _exec_compare_competitor(args: dict) -> str:
    raw_key = args.get("competitor", "").lower().strip()
    key = _COMPETITOR_ALIASES.get(raw_key, raw_key)
    data = COMPETITOR_DATA.get(key)
    if not data:
        return json.dumps({"error": f"No data available for '{raw_key}'. Known competitors: {', '.join(COMPETITOR_DATA.keys())}"})

    win = data.get("monday_win_strategy", {})
    result = {
        "competitor": data["name"],
        "category": data.get("category", ""),
        "their_price": data.get("typical_price", "unknown"),
        "their_strengths": data.get("strengths", []),
        "their_weaknesses": data.get("weaknesses", []),
        "why_they_switch": data.get("why_prospects_consider_switching", []),
        "win_strategy": win.get("key_message", ""),
        "killer_points": win.get("killer_points", []),
        "objection_handlers": win.get("objection_handlers", {}),
        "instruction": (
            f"Use the win strategy and killer points to naturally show why monday.com "
            f"is a better fit. If the prospect objects, use the objection handlers. "
            f"Never trash-talk {data['name']} — acknowledge its strengths, then pivot to "
            f"what monday.com does better for THIS prospect's situation."
        ),
    }
    return json.dumps(result)


def _exec_calculate_roi(args: dict) -> str:
    current_tool = args.get("current_tool", "current tool")
    current_cost = args.get("current_monthly_cost", 0)
    team_size = args.get("team_size", 10)
    tools_replaced = args.get("num_tools_replaced", 2)

    best_plan = plan_for_seats(team_size)
    monday_per_seat = MONDAY_PRICING[best_plan]["price"]
    monday_total = monday_per_seat * team_size

    estimated_other_tools_cost = current_cost * 0.6 * (tools_replaced - 1)
    total_current = current_cost + estimated_other_tools_cost
    net_change = monday_total - total_current
    consolidation_savings = estimated_other_tools_cost

    return json.dumps({
        "current_setup": {
            "tool": current_tool,
            "monthly_cost": f"${current_cost:,.0f}",
            "estimated_other_tools": f"${estimated_other_tools_cost:,.0f}" if tools_replaced > 1 else "$0",
            "total_estimated": f"${total_current:,.0f}/month",
        },
        "monday_proposal": {
            "plan": best_plan,
            "per_seat": f"${monday_per_seat}/seat/month",
            "seats": team_size,
            "total": f"${monday_total:,.0f}/month",
        },
        "roi_analysis": {
            "consolidation_savings": f"${consolidation_savings:,.0f}/month by replacing {tools_replaced} tools with one",
            "net_monthly_change": f"{'saves' if net_change < 0 else 'costs'} ${abs(net_change):,.0f}/month vs current total",
            "annual_impact": f"${abs(net_change * 12):,.0f}/year",
            "value_adds": [
                "Consolidate multiple tools into one platform",
                "200+ automations save ~5 hours/person/month",
                "Better cross-team visibility reduces meetings by ~20%",
                f"Estimated productivity gain: ${team_size * 200:,.0f}/month at $200/person",
            ],
        },
    })


def _exec_suggest_automations(args: dict) -> str:
    team = args.get("team_type", "project management").lower().strip()
    templates = AUTOMATION_TEMPLATES.get(team, AUTOMATION_TEMPLATES["project management"])
    return json.dumps({
        "team": team,
        "automations": [
            {"recipe": f"{t['trigger']} → {t['action']}", "benefit": t["benefit"]}
            for t in templates
        ],
        "note": f"monday.com has 200+ automation templates. These are the top 3 most impactful for {team} teams.",
    })
