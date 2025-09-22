import os
from typing import Optional

from livekit.plugins import openai as lk_openai


def create_sarvam_llm(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Create an OpenAI-compatible LLM client configured for Sarvam.
    """

    api_key = (
        api_key
        or os.getenv("SARVAM_LLM_API_KEY")
        or os.getenv("SARVAM_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Missing Sarvam API key. Set `SARVAM_LLM_API_KEY` or `SARVAM_API_KEY`."
        )

    base_url = base_url or os.getenv("SARVAM_LLM_BASE_URL") or "https://api.sarvam.ai/v1"
    model = model or os.getenv("SARVAM_LLM_MODEL") or "sarvam-m"

    return lk_openai.LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
