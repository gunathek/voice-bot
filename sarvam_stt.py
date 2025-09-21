from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr

SARVAM_STT_BASE_URL = "https://api.sarvam.ai/speech-to-text"


@dataclass
class SarvamSTTOptions:
    language: str
    model: str = "saarika:v2.5"
    api_key: Optional[str] = None
    base_url: str = SARVAM_STT_BASE_URL
    extra_params: dict[str, Any] | None = None


class SarvamCustomSTT(stt.STT):
    """Custom Sarvam STT with flexible parameter pass-through.

    This implementation mirrors the official LiveKit Sarvam plugin but allows
    supplying additional Sarvam API parameters not exposed by the plugin.

    It remains non-streaming and interoperates with VAD/turn detection via
    LiveKit's StreamAdapter inside AgentSession.
    """

    def __init__(
        self,
        *,
        language: str,
        model: str = "saarika:v2.5",
        api_key: Optional[str] = None,
        base_url: str = SARVAM_STT_BASE_URL,
        extra_params: Optional[dict[str, Any]] = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))

        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Sarvam API key is required. Set SARVAM_API_KEY or provide api_key."
            )

        self._opts = SarvamSTTOptions(
            language=language,
            model=model,
            api_key=self._api_key,
            base_url=base_url,
            extra_params=extra_params or {},
        )

        # lazily created session via utils.http_context to share connection pool
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    @staticmethod
    def _add_form_field(form: aiohttp.FormData, key: str, val: Any) -> None:
        if val is None:
            return
        # Convert Python types to form field-friendly values
        if isinstance(val, (bool, int, float)):
            form.add_field(key, json.dumps(val))  # true/false or numbers
        elif isinstance(val, (dict, list)):
            form.add_field(key, json.dumps(val), content_type="application/json")
        else:
            form.add_field(key, str(val))

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        # Resolve language, falling back to constructor value
        resolved_lang = self._opts.language if isinstance(language, type(NOT_GIVEN)) else language

        # Prepare audio as WAV bytes (matches LiveKit plugin behaviour)
        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")

        # Standard Sarvam fields
        self._add_form_field(form, "language_code", resolved_lang)
        self._add_form_field(form, "model", self._opts.model)

        # Pass-through any extra params
        for k, v in (self._opts.extra_params or {}).items():
            # Avoid overriding core fields if user repeats them
            if k in {"file", "language_code", "model"}:
                continue
            self._add_form_field(form, k, v)

        headers = {"api-subscription-key": self._opts.api_key or ""}

        try:
            async with self._ensure_session().post(
                url=self._opts.base_url,
                data=form,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=conn_options.timeout,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    raise APIStatusError(
                        message=f"Sarvam API Error: {error_text}", status_code=res.status
                    )

                payload = await res.json()

                # Common fields in Sarvam responses
                transcript = payload.get("transcript", "")
                request_id = payload.get("request_id", "")
                detected_lang = payload.get("language_code") or resolved_lang or ""

                start_time = 0.0
                end_time = 0.0
                ts = payload.get("timestamps")
                if isinstance(ts, dict):
                    starts = ts.get("start_time_seconds")
                    ends = ts.get("end_time_seconds")
                    if isinstance(starts, list) and starts:
                        try:
                            start_time = float(starts[0])
                        except Exception:
                            start_time = 0.0
                    if isinstance(ends, list) and ends:
                        try:
                            end_time = float(ends[-1])
                        except Exception:
                            end_time = 0.0

                # Estimate end_time if not provided
                if start_time == 0.0 and end_time == 0.0:
                    try:
                        if isinstance(buffer, list) and buffer:
                            total_samples = sum(f.samples_per_channel for f in buffer)
                            sr = buffer[0].sample_rate
                            if total_samples and sr:
                                end_time = total_samples / sr
                        elif hasattr(buffer, "samples_per_channel") and hasattr(
                            buffer, "sample_rate"
                        ):
                            end_time = buffer.samples_per_channel / buffer.sample_rate
                    except Exception:
                        end_time = 0.0

                alternatives = [
                    stt.SpeechData(
                        language=str(detected_lang),
                        text=str(transcript),
                        start_time=float(start_time),
                        end_time=float(end_time),
                        confidence=1.0,
                    )
                ]

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=str(request_id),
                    alternatives=alternatives,
                )

        except asyncio.TimeoutError as e:
            raise APITimeoutError("Sarvam STT request timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Sarvam STT connection error: {e}") from e
        except Exception as e:
            raise APIConnectionError(f"Unexpected error in Sarvam STT: {e}") from e


def _load_extra_params_from_env() -> dict[str, Any]:
    raw = os.getenv("SARVAM_STT_EXTRA_PARAMS_JSON", "").strip()
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass
    return {}


def create_sarvam_stt(
    *,
    language: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    extra_params: Optional[dict[str, Any]] = None,
) -> SarvamCustomSTT:
    """Factory to create the custom Sarvam STT with env fallbacks.

    Env vars used if arguments are not provided:
    - SARVAM_API_KEY (shared)
    - SARVAM_STT_LANGUAGE
    - SARVAM_STT_MODEL (default: saarika:v2.5)
    - SARVAM_STT_BASE_URL (default: https://api.sarvam.ai/speech-to-text)
    - SARVAM_STT_EXTRA_PARAMS_JSON (JSON object string)
    """

    return SarvamCustomSTT(
        language=language or os.getenv("SARVAM_STT_LANGUAGE", "en-IN"),
        model=model or os.getenv("SARVAM_STT_MODEL", "saarika:v2.5"),
        api_key=api_key or os.getenv("SARVAM_API_KEY"),
        base_url=base_url or os.getenv("SARVAM_STT_BASE_URL", SARVAM_STT_BASE_URL),
        extra_params=extra_params or _load_extra_params_from_env(),
    )

