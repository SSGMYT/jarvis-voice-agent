from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from .audio import LiveListener, VoskTranscriber, list_devices
from .config import load_config
from .llm import FallbackResponder, GroqResponder, Responder
from .tts import Speaker
from .ptt import PushToTalk
from .tools import execute_tool, parse_tool_call


def _pick_responder(cfg) -> Responder:
    if cfg.groq_api_key:
        return GroqResponder(api_key=cfg.groq_api_key, model=cfg.groq_model, system_prompt=cfg.system_prompt)
    return FallbackResponder()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / ".env")
    cfg = load_config()

    print("Voice Agent starting.")
    print("Audio devices (set VOICE_AGENT_DEVICE to an index if needed):")
    print(json.dumps(list_devices(), indent=2))

    model_path = Path(cfg.vosk_model_path).expanduser()
    if not model_path.is_absolute():
        model_path = project_root / model_path
    transcriber = VoskTranscriber(str(model_path), cfg.sample_rate)
    speaker = Speaker(rate=cfg.tts_rate, volume=cfg.tts_volume, voice_contains=cfg.tts_voice_contains)
    responder = _pick_responder(cfg)
    history: list[dict[str, str]] = []

    listener = LiveListener(
        sample_rate=cfg.sample_rate,
        block_ms=cfg.block_ms,
        device=cfg.device,
        vad_mode=cfg.vad_mode,
        end_silence_ms=cfg.end_silence_ms,
        pre_roll_ms=cfg.pre_roll_ms,
        max_segment_s=cfg.max_segment_s,
    )

    speaker.say("Ready.")
    if cfg.push_to_talk_enabled and sys.platform.startswith("win"):
        ptt = PushToTalk(key=cfg.push_to_talk_key)
        ptt.start()
        print(f"Push-to-talk enabled. Press '{cfg.push_to_talk_key}' to talk.")
        speaker.say(f"Push to talk enabled. Press {cfg.push_to_talk_key} to speak.")
    elif cfg.push_to_talk_enabled:
        ptt = None
        print("Push-to-talk is only supported on Windows; continuing with voice activation.")
    else:
        ptt = None

    if cfg.wake_word_enabled and not cfg.push_to_talk_enabled:
        print(f"Wake word enabled. Say '{cfg.wake_word}'.")
    print("Listening.")

    stop_phrases = [
        "Of course, Sir. Shutting down now. Thank you.",
        "Understood, Sir. I’ll stand by. Have a great day.",
        "Certainly, Sir. Ending the session. It’s been a pleasure.",
        "Right away, Sir. Stopping now. Let me know if you need anything else.",
        "As you wish, Sir. Powering down. Thank you for your time.",
    ]

    awake_until = 0.0

    def _record_one_utterance() -> str:
        seg = next(listener.segments())
        return transcriber.transcribe(seg.pcm16).strip()

    while True:
        if ptt is not None:
            # Wait for keyboard arming.
            ptt.wait_until_armed()
            ptt.disarm()
            text = _record_one_utterance()
        else:
            text = _record_one_utterance()

        print(f"You: {text}")
        if not text:
            continue
        low = text.strip().lower()
        if low in {"quit", "exit", "stop"}:
            speaker.say(random.choice(stop_phrases))
            break

        # Wake word gate (disabled when push-to-talk is on)
        if cfg.wake_word_enabled and ptt is None:
            now = time.time()
            if now > awake_until:
                ww = cfg.wake_word.strip().lower()
                if ww and ww in low:
                    # Activate and optionally strip wake word from the command.
                    awake_until = now + float(cfg.wake_active_seconds)
                    cleaned = low.replace(ww, "").strip(" ,.!?:;-")
                    if cleaned:
                        text = cleaned
                        low = cleaned.lower()
                    else:
                        speaker.say("Yes, Sir?")
                        continue
                else:
                    # Not awake yet; ignore background speech.
                    continue

        tool_call = parse_tool_call(text)
        if tool_call:
            if tool_call.requires_confirmation:
                speaker.say(f"I can do that. Say confirm to proceed: {tool_call.summarize()}.")
                # Wait for a short confirmation utterance.
                confirm_text = _record_one_utterance().strip().lower()
                print(f"You (confirm): {confirm_text}")
                if confirm_text not in {"confirm", "yes", "do it", "proceed"}:
                    speaker.say("Cancelled.")
                    continue
            result = execute_tool(tool_call)
            print(f"Tool: {tool_call.kind} -> {result.ok} ({result.message})")
            speaker.say(result.message)
            # Log tool outcomes into memory for better continuity.
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": result.message})
            history = history[-2 * max(1, int(cfg.memory_turns)) :]
            continue

        reply = responder.respond(text, history=history).text
        print(f"Agent: {reply}")
        speaker.say(reply)
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        history = history[-2 * max(1, int(cfg.memory_turns)) :]


if __name__ == "__main__":
    main()
