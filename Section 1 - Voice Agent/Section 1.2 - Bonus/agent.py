from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import deepgram,groq, silero
from custom_stt import FasterWhisperSTT
from tools import get_order_status, contact_support_team

load_dotenv()


class Assistant(Agent):

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a support assistant for a food delivery app. "
                "You are expected to answer queries related to the tracking of orders and general customer service. "
                "You are encouraged to use the available tools to fetch the details of the orders whenever there is an order ID. "
                "If not, ask for it and do not create any details yourself. "
                "If you can't help a client with a particular query, let them know politely."
            ),
            tools=[get_order_status, contact_support_team,],
        )


async def joining_room(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=FasterWhisperSTT(model_size="base.en"),
        llm=groq.LLM(model="openai/gpt-oss-120b"),
        tts=deepgram.TTS(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
    )

    await session.generate_reply(
        instructions="""Hi! Welcome to our food delivery support. 
        I'm here to help with your orders, deliveries, or any questions you may have. 
        How can I help you today?"""
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=joining_room,
            agent_name="food-delivery-agent",
        )
    )
