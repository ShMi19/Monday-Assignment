def generate_monday_board(qualification):
    """
    Create a mock monday.com board definition from structured qualification data.

    Args:
        qualification (dict): Parsed qualification JSON with at least:
            - industry
            - company_size
            - team_using_monday
            - use_case

    Returns:
        dict: Board metadata and example structure.
    """

    use_case = qualification.get("use_case") or "Customer Workflow"
    team = qualification.get("team_using_monday") or "Team"
    industry = qualification.get("industry") or "Business"

    board_name = f"{team} – {use_case} ({industry})"

    # Simple, readable structure that looks like a monday board:
    columns = [
        {"id": "item_name", "title": "Item", "type": "name"},
        {"id": "status", "title": "Status", "type": "status"},
        {"id": "owner", "title": "Owner", "type": "person"},
        {"id": "priority", "title": "Priority", "type": "status"},
    ]

    example_items = [
        {
            "name": f"Onboard new {industry.lower()} client",
            "status": "Working on it",
            "owner": "Account manager",
            "priority": "High",
        },
        {
            "name": f"Follow up on {team.lower()} request",
            "status": "Stuck",
            "owner": "Rep",
            "priority": "Medium",
        },
    ]

    return {
        "board_name": board_name,
        "link": "https://monday-demo-board.com/abc123",
        "columns": columns,
        "items": example_items,
    }