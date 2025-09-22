import os
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, JobContext
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import (noise_cancellation, silero)
from livekit.plugins import sarvam
from sarvam_llm import create_sarvam_llm
from tracing_langfuse import setup_langfuse
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:

        base_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(base_dir, "prompts", "system_prompt.md")
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        super().__init__(instructions=system_prompt)


async def entrypoint(ctx: JobContext):

    trace_provider = setup_langfuse(metadata={"langfuse.session.id": ctx.room.name})

    async def flush_trace():
        trace_provider.force_flush()
    ctx.add_shutdown_callback(flush_trace)

    llm = create_sarvam_llm()
    sarvam_tts = sarvam.TTS(target_language_code="hi-IN", model="bulbul:v2", speaker="anushka", enable_preprocessing=True)

    session = AgentSession(
        stt=sarvam.STT(model="saarika:v2.5", language="unknown"),
        llm=llm,
        tts=sarvam_tts,
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        from livekit.agents import metrics
        metrics.log_metrics(ev.metrics)

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
