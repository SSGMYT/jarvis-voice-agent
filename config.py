from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Config:
    # Audio
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 20
    device: str | int | None = None

    # VAD
    vad_mode: int = 2
    max_segment_s: float = 12.0
    end_silence_ms: int = 700
    pre_roll_ms: int = 200

    # STT (Vosk)
    vosk_model_path: str = r"models/vosk-model-small-en-us-0.15"

    # LLM (Groq) optional
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"

    # Agent
    system_prompt: str = "You are a concise, helpful voice assistant. Keep responses short and speakable."
    memory_turns: int = 8

    # Wake word
    wake_word_enabled: bool = True
    wake_word: str = "jarvis"
    wake_active_seconds: float = 12.0

    # Push-to-talk
    push_to_talk_enabled: bool = False
    push_to_talk_key: str = "t"

    # TTS
    tts_rate: int = 185
    tts_volume: float = 1.0
    tts_voice_contains: str | None = None


def load_config() -> Config:
    """
    Load configuration from environment variables.
    Must be called after dotenv is loaded so .env values are visible.
    """
    return Config(
        sample_rate=_env_int("VOICE_AGENT_SAMPLE_RATE", 16000),
        block_ms=_env_int("VOICE_AGENT_BLOCK_MS", 20),
        device=os.getenv("VOICE_AGENT_DEVICE") or None,
        vad_mode=_env_int("VOICE_AGENT_VAD_MODE", 2),
        max_segment_s=_env_float("VOICE_AGENT_MAX_SEGMENT_S", 12.0),
        end_silence_ms=_env_int("VOICE_AGENT_END_SILENCE_MS", 700),
        pre_roll_ms=_env_int("VOICE_AGENT_PRE_ROLL_MS", 200),
        vosk_model_path=os.getenv("VOSK_MODEL_PATH", r"models/vosk-model-small-en-us-0.15"),
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        system_prompt=os.getenv(
            "VOICE_AGENT_SYSTEM_PROMPT",
            "You are a professional voice assistant.\n"
            "Be calm, polite, and confident.\n"
            "Keep responses short and easy to speak (usually 1–3 sentences).\n"
            "If a user asks for an action you can take (open app, web search, safe file ops), "
            "confirm briefly and proceed.\n"
            "Avoid filler words and avoid reading URLs unless the user asks.\n",
        ),
        memory_turns=_env_int("VOICE_AGENT_MEMORY_TURNS", 8),
        wake_word_enabled=_env_bool("VOICE_AGENT_WAKE_WORD_ENABLED", True),
        wake_word=(os.getenv("VOICE_AGENT_WAKE_WORD", "jarvis") or "jarvis").strip(),
        wake_active_seconds=_env_float("VOICE_AGENT_WAKE_ACTIVE_SECONDS", 12.0),
        push_to_talk_enabled=_env_bool("VOICE_AGENT_PTT_ENABLED", False),
        push_to_talk_key=(os.getenv("VOICE_AGENT_PTT_KEY", "t") or "t").strip()[:1],
        tts_rate=_env_int("VOICE_AGENT_TTS_RATE", 185),
        tts_volume=_env_float("VOICE_AGENT_TTS_VOLUME", 1.0),
        tts_voice_contains=os.getenv("VOICE_AGENT_TTS_VOICE_CONTAINS") or None,
    )
