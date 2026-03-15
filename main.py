import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# --------------------
# Agent State
# --------------------

class LeadData(BaseModel):
    industry: Optional[str] = None
    team_size: Optional[int] = None
    use_case: Optional[str] = None
    company_stage: Optional[str] = None
    budget: Optional[str] = None


class AgentState:
    def __init__(self):
        self.state = "DISCOVERY"
        self.data = LeadData()

agent = AgentState()

# --------------------
# LLM CALL (FREE)
# --------------------

def call_llm(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

# --------------------
# Extraction
# --------------------

def extract_lead_data(message):

    prompt = f"""
Extract structured fields from this message.

Fields:
industry
team_size
use_case
company_stage
budget

Return JSON only.

Message:
{message}
"""

    response = call_llm(prompt)

    import json

    try:
        parsed = json.loads(response)
    except:
        parsed = {}

    return parsed

# --------------------
# Board generator
# --------------------

def generate_board(use_case):

    templates = {
        "marketing": [
            "Campaign Name",
            "Owner",
            "Status",
            "Budget",
            "Due Date"
        ],
        "software": [
            "Feature",
            "Owner",
            "Priority",
            "Sprint",
            "Status"
        ]
    }

    return templates.get(use_case.lower(), [
        "Task",
        "Owner",
        "Status",
        "Due Date"
    ])

# --------------------
# Pricing
# --------------------

def calculate_price(team_size):

    seat_price = 10

    if team_size is None:
        return None

    return team_size * seat_price

# --------------------
# Payment mock
# --------------------

def create_payment_link(amount):

    return f"https://checkout-demo.local/pay?amount={amount}"

# --------------------
# Chat endpoint
# --------------------

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):

    global agent

    message = req.message

    # DISCOVERY
    if agent.state == "DISCOVERY":

        data = extract_lead_data(message)

        for k,v in data.items():
            if v:
                setattr(agent.data, k, v)

        agent.state = "QUALIFICATION"

        return {
            "reply": "Great! How many people will use the platform?",
            "state": agent.state
        }

    # QUALIFICATION
    if agent.state == "QUALIFICATION":

        data = extract_lead_data(message)

        for k,v in data.items():
            if v:
                setattr(agent.data, k, v)

        if agent.data.team_size is None:

            return {
                "reply": "Could you tell me how big your team is?",
                "state": agent.state
            }

        agent.state = "DEMO"

        return {
            "reply": f"I can generate a workspace for your {agent.data.use_case} workflow.",
            "state": agent.state
        }

    # DEMO
    if agent.state == "DEMO":

        board = generate_board(agent.data.use_case or "")

        agent.state = "USE_CASE"

        return {
            "reply": f"Sample board columns: {', '.join(board)}",
            "state": agent.state
        }

    # USE CASE
    if agent.state == "USE_CASE":

        price = calculate_price(agent.data.team_size)

        payment = create_payment_link(price)

        agent.state = "PAYMENT"

        return {
            "reply": f"Estimated price: ${price}/month. Checkout here: {payment}",
            "state": agent.state
        }

    return {"reply": "Your account is ready!", "state": "DONE"}