import os
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (openai, noise_cancellation, silero)
from livekit.plugins import sarvam, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    session = AgentSession(
        stt=sarvam.STT(model="saarika:v2.5", language="en-IN"),
        llm=google.LLM(model="gemini-2.5-flash-lite", api_key=google_api_key),
        tts=sarvam.TTS(target_language_code="en-IN", model="bulbul:v2", speaker="anushka"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))