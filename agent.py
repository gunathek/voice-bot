from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (noise_cancellation, silero)
from livekit.plugins import sarvam
from sarvam_llm import create_sarvam_llm
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    llm = create_sarvam_llm()
    sarvam_tts = sarvam.TTS(target_language_code="hi-IN", model="bulbul:v2", speaker="anushka", enable_preprocessing=True)

    session = AgentSession(
        stt=sarvam.STT(model="saarika:v2.5", language="unknown"),
        llm=llm,
        tts=sarvam_tts,
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        try:
            if not getattr(ev, "is_final", False):
                from dataclasses import replace
            lang = getattr(ev, "language", None) or "hi-IN"
            sarvam_tts._opts = replace(sarvam_tts._opts, target_language_code=lang)
        except Exception:
            pass

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))