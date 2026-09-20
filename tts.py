from __future__ import annotations

import re
import threading
import sys

import pyttsx3


def _polish_for_speech(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # Remove markdown-ish artifacts and codey punctuation that sounds bad.
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = t.replace("…", "...")
    # Collapse excessive whitespace/newlines.
    t = re.sub(r"\s+", " ", t).strip()
    # Keep it speakable: avoid huge run-ons.
    if len(t) > 500:
        t = t[:500].rsplit(" ", 1)[0].rstrip(".!,;:") + "."
    return t


class Speaker:
    def __init__(self, *, rate: int = 185, volume: float = 1.0, voice_contains: str | None = None) -> None:
        # pyttsx3 can occasionally get into a "speaks once then silent" state on Windows.
        # We keep a lock and (on Windows) re-init the engine per utterance for reliability.
        self._lock = threading.Lock()
        self._rate = int(rate)
        self._volume = float(volume)
        self._voice_contains = (voice_contains or "").strip().lower() or None

    def say(self, text: str) -> None:
        text = _polish_for_speech(text)
        if not text:
            return
        with self._lock:
            try:
                if sys.platform.startswith("win"):
                    engine = pyttsx3.init("sapi5")
                else:
                    engine = pyttsx3.init()
                try:
                    engine.setProperty("volume", self._volume)
                except Exception:
                    pass
                try:
                    engine.setProperty("rate", self._rate)
                except Exception:
                    pass

                if self._voice_contains:
                    try:
                        voices = engine.getProperty("voices") or []
                        for v in voices:
                            name = (getattr(v, "name", "") or "").lower()
                            vid = (getattr(v, "id", "") or "").lower()
                            if self._voice_contains in name or self._voice_contains in vid:
                                engine.setProperty("voice", v.id)
                                break
                    except Exception:
                        pass
                engine.say(text)
                engine.runAndWait()
                try:
                    engine.stop()
                except Exception:
                    pass
            except Exception as e:
                # Don't crash the whole agent if TTS fails; surface error to console.
                print(f"[TTS error] {e}")
