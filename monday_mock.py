"""Mock monday.com board generation service.

Attempts AI-generated board content first (via the 70b LLM), then falls
back to generic templates if the LLM call fails.
"""

from agent import generate_board_with_llm

_FALLBACK_COLUMNS = [
    {"id": "item_name", "title": "Item", "type": "name"},
    {"id": "status", "title": "Status", "type": "status"},
    {"id": "owner", "title": "Owner", "type": "person"},
    {"id": "priority", "title": "Priority", "type": "status"},
    {"id": "due", "title": "Due Date", "type": "date"},
]

_FALLBACK_ITEMS = [
    {"item_name": "Example task 1", "status": "Working on it", "owner": "Team lead", "priority": "High", "due": "2026-04-10"},
    {"item_name": "Example task 2", "status": "Not started", "owner": "Team member", "priority": "Medium", "due": "2026-04-20"},
    {"item_name": "Example task 3", "status": "Done", "owner": "Team lead", "priority": "Low", "due": "2026-03-30"},
]


def generate_monday_board(qualification: dict) -> dict:
    """Generate a monday.com board tailored to the prospect's input."""
    use_case = qualification.get("use_case") or "Customer Workflow"
    team = qualification.get("team") or qualification.get("team_using_monday") or "Team"
    industry = qualification.get("industry") or "Business"

    board_name = f"{team} – {use_case} ({industry})"

    llm_board = generate_board_with_llm(qualification)

    if llm_board and "columns" in llm_board and "items" in llm_board:
        columns = llm_board["columns"]
        items = llm_board["items"]
    else:
        columns = _FALLBACK_COLUMNS
        items = _FALLBACK_ITEMS

    return {
        "board_name": board_name,
        "link": "https://monday-demo-board.com/abc123",
        "columns": columns,
        "items": items,
        "generated_by": "ai" if llm_board else "template",
    }
