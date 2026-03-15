import hashlib
import time

PRICING = {
    "Starter": {"seat_price": 9, "billing": "monthly"},
    "Standard": {"seat_price": 12, "billing": "monthly"},
    "Pro": {"seat_price": 19, "billing": "monthly"},
}


def create_payment_link(plan: str, seats: int | None = None) -> dict:
    """Generate a mock Stripe-like checkout session based on plan and team size."""
    tier = PRICING.get(plan, PRICING["Standard"])
    seat_count = seats or 3
    seat_price = tier["seat_price"]
    subtotal = seat_price * seat_count
    billing = tier["billing"]

    session_seed = f"{plan}-{seat_count}-{int(time.time()) // 3600}"
    session_id = "cs_demo_" + hashlib.md5(session_seed.encode()).hexdigest()[:24]

    return {
        "checkout_url": f"https://checkout.stripe.com/c/pay/{session_id}",
        "plan": plan,
        "seats": seat_count,
        "price_per_seat": f"${seat_price}/seat/{billing}",
        "subtotal": f"${subtotal}/{billing}",
        "session_id": session_id,
    }
