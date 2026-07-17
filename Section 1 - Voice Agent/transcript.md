# Section 1 — Voice Agent Transcript & Tool-Call Evidence

**Run mode:** `python agent.py console`
**Pipeline:** Deepgram (STT) → Groq `openai/gpt-oss-120b` (LLM) → Deepgram (TTS), Silero VAD
**Date:** 2026-07-17

---

## Conversation

**Assistant:** Hello!  How can I assist you with your order or any other question today?

**User:** Hi. I need to check the status of my orders.

**Assistant:** Sure thing! Could you please provide the order ID you'd like to check?

**User:** I have three different orders with you. Can you check the first order? The order number is one two three.

>  **TOOL CALL** `get_order_status(order_id="123")` → executed in 6ms

**Assistant:** Your first order (ID **123**) is currently **out for delivery** and should arrive in about **12 minutes**. Let me know if you'd like updates on your other orders or need anything else!

**User:** Okay. Thanks. What about my second order? Its ID is four five six.

>  **TOOL CALL** `get_order_status(order_id="456")` → executed in 5ms

**Assistant:** Your second order (ID **456**) is currently **being prepared** in the kitchen. Feel free to let me know when you'd like an update on the third order...

>  **INTERRUPTION DETECTED** — user began speaking while agent was mid-reply (barge-in handled automatically by VAD + adaptive interruption detector, probability 0.83)

**User:**  And can you also check my third order? Its ID is seven eight nine.

**Assistant:** Absolutely! Here's a quick snapshot of the two orders you placed at the same time:

| Order ID | Current Status | Estimated Time |
|----------|----------------|----------------|
| 123 | Out for delivery | ~12 minutes |
| 456 | Being prepared | Ready soon |

>  **TOOL CALL** `get_order_status(order_id="789")` → executed in 10ms

**Assistant:** Your third order (ID **789**) has already been **delivered**. If you've received it, great! If there's anything missing or you need further assistance, just let me know.

**User:** Wait — it doesn't make sense if order 789 is already delivered. Where was it delivered? I never received that.

> 🔧 **TOOL CALL** `contact_support_team(reason="Customer reports order 789 marked as delivered but they have not received it")` → executed in 7ms
> Log: `Escalation requested: Customer reports order 789 marked as delivered but they have not received it`

**Assistant:** I'm sorry you haven't received your order 789 even though it shows as delivered. I've escalated this issue to our support...

**User:** Can you double check that, please?

>  **TOOL CALL** `get_order_status(order_id="789")` → executed again (re-verification on user request)

**User:** This is really frustrating. I've been waiting a long time today, and I need some answers.

**Assistant:** I understand how frustrating that must be. The system still shows order 789 as delivered, but since you haven't received it, we'll need to investigate further. Here's what we can do...

**User:** Can you connect me with your support team, please?

>  **TOOL CALL** `contact_support_team(reason="User requests direct connection to support team for unresolved delivery issues")` → executed in 5ms
> Log: `Escalation requested: User requests direct connection to support team for unresolved delivery issues`

**Assistant:** I've escalated your case to our support team, and they'll reach out shortly. In the meantime, you can also contact them directly at **support@fooddelivery.com** or **+201 001 234 567**. Sorry for the inconvenience, and thank you for your patience.

**User:** Thank you.

**Assistant:** You're very welcome! If anything else comes up, just let me know. Have a great day!

**User:** Have a great day.

**Assistant:** Take care! 

---

## Evidence Summary

| # | Tool | Arguments | Result |
|---|------|-----------|--------|
| 1 | `get_order_status` | `order_id="123"` |  success |
| 2 | `get_order_status` | `order_id="456"` |  success |
| 3 | `get_order_status` | `order_id="789"` |  success |
| 4 | `contact_support_team` | `reason="...not received it"` |  success (escalation logged) |
| 5 | `get_order_status` | `order_id="789"` (re-check) |  success |
| 6 | `contact_support_team` | `reason="...direct connection to support team"` |  success (escalation logged) |

**Barge-in observed:** one genuine interruption event was captured mid-run (`interruption detected`, probability 0.83), confirming the session's VAD + adaptive interruption detector stopped agent speech when the user began talking — this is native `AgentSession` behavior, not custom code.

**Note:** raw terminal logs also show the adaptive (ML-based) interruption detector falling back to VAD-only mode after a timeout mid-session — worth mentioning in the write-up as an example of the framework's built-in resilience/fallback behavior.
