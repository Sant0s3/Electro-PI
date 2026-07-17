from livekit.agents import function_tool, RunContext
import logging
# For error handling and callbacks to prevent call from dropping
logger = logging.getLogger("food-delivery-tools")

_ORDERS = {
    "123": "out for delivery, ETA 12 minutes",
    "456": "being prepared in the kitchen",
    "789": "delivered",
}

@function_tool
async def get_order_status(context: RunContext, order_id: str) -> str:
    """Look up the current status of a delivery order by its ID."""
    if not order_id or not order_id.strip():
        return "I need a valid order ID to look that up. Could you provide it?"

    status = _ORDERS.get(order_id.strip())
    if status is None:
        logger.warning(f"Order lookup failed: unknown order_id = {order_id}")
        return f"I couldn't find order {order_id}. Please double-check the ID."

    return f"Order {order_id} is {status}."
# if call drop it has last card to contact support team, so that the user can get help from human agent
@function_tool
async def contact_support_team(context: RunContext, reason: str) -> str:
    """Escalate to a human support agent when the issue can't be resolved automatically."""
    if not reason or not reason.strip():
        return "Could you briefly describe the issue so I can pass it to our team?"

    logger.info(f"Escalation requested: {reason}")
    return (
        "I've logged this for our support team "
        "you can also reach them directly "
        "at support@fooddelivery.com or +201001234567."
    )