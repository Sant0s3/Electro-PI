# Section 1 Notes

## Barge-in & Interruption Handling
In the LiveKit Agents SDK, barge-in is handled natively by the `AgentSession` pipeline using Silero VAD (Voice Activity Detection). When the VAD detects user speech while the agent is generating or playing audio, it immediately triggers an interruption, halts the TTS playback, and clears the current generation queue so the agent can listen to the new input.

To make this robust in production:
- **Tuning Interruption Sensitivity**: VAD sensitivity needs to be balanced. If it's too sensitive, background noise or breathing interrupts the agent. If it's too slow, the user feels ignored. We can tune the `pre_speech_pad_frames` and VAD threshold parameters to filter out short breaths or clicks.
- **Agent Prompting**: To reduce natural interruptions, the agent's system prompt should instruct it to give concise, turn-based responses rather than long paragraphs of text.
- **Handling Delays**: If the LLM has high latency, the user might speak before the agent starts playing TTS. Implementing a "haste" check or streaming tokens immediately to the TTS engine prevents the user from double-talking.

## Adding a Second Tool Safely
To add a new tool (like `get_delivery_estimate(order_id: str)`) safely without breaking the conversational flow, we follow these practices:

1. **Clear Schema Definition**: The LiveKit SDK auto-generates the tool schema for the LLM based on Python function signatures and docstrings. We write precise Google-style docstrings and use strict type hints:
   ```python
   @function_tool
   async def get_delivery_estimate(order_id: str) -> str:
       """Get the estimated delivery time for a specific order.
       
       Args:
           order_id: The unique alphanumeric ID of the order.
       """
   ```
2. **Defensive Parameter Validation**: Inside the function, we validate all inputs before performing any logic. If a parameter is missing or malformed, we return a helpful validation message rather than throwing an exception:
   ```python
   if not order_id or len(order_id.strip()) == 0:
       return "I couldn't read the order ID. Can you please repeat the order number?"
   ```
3. **Graceful Failure & Error Boundaries**: We wrap external API calls or database lookups in try-except blocks. If the external service fails, the tool should log the error for debugging and return a polite fallback response to the agent, keeping the conversation alive:
   ```python
   try:
       # Simulated API call
       eta = await call_delivery_api(order_id)
       return f"The estimated delivery time is {eta}."
   except Exception as e:
       logger.error(f"Failed to fetch delivery estimate for {order_id}: {e}")
       return "I'm having trouble connecting to our tracking system right now. Let me escalate this to our support team."
   ```
4. **Escalation Fallback**: If the tool fails multiple times or cannot resolve the user's issue, we expose a tool to log an escalation ticket or route them to a human agent (`contact_support_team`).