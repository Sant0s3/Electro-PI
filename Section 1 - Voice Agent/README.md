# Section 1: LiveKit Voice Agent

A minimal real-time voice agent built with the LiveKit Agents SDK.

## Overview
This agent acts as a support assistant for a food delivery app. It handles natural speech conversation and exposes two tools:
- `get_order_status`: Looks up simulated delivery status for an order ID.
- `contact_support_team`: Escalates the conversation to human support if needed.

## Setup & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**:
   Copy `.env.example` to `.env` and fill in your keys:
   ```env
   LIVEKIT_URL=your-livekit-url
   LIVEKIT_API_KEY=your-api-key
   LIVEKIT_API_SECRET=your-api-secret
   DEEPGRAM_API_KEY=your-deepgram-key
   GROQ_API_KEY=your-groq-key
   ```

3. **Start the agent**:
   ```bash
   python agent.py dev
   ```

## Swapping Pipeline Components (Section 1.2 Bonus)
In `Section 1.2 - Bonus`, we swapped the Deepgram STT engine with a local **Faster-Whisper** model running on the CPU.
To run the bonus agent:
```bash
cd "Section 1.2 - Bonus"
pip install -r requirements.txt
python agent.py dev
```
See `NOTES.md` for a complete write-up on barge-in, safe tool calling, and details on component swapping.
