from __future__ import annotations

import sys
import threading
import time


class PushToTalk:
    """
    Simple console push-to-talk:
    - Press the configured key (default: 't') to arm one utterance.
    - After an utterance is processed, it disarms automatically.

    Works in Windows PowerShell/cmd using msvcrt. On non-Windows, it stays disabled.
    """

    def __init__(self, key: str = "t") -> None:
        self.key = (key or "t")[:1].lower()
        self._armed = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not sys.platform.startswith("win"):
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="push-to-talk", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def arm(self) -> None:
        self._armed.set()

    def disarm(self) -> None:
        self._armed.clear()

    def wait_until_armed(self) -> None:
        self._armed.wait()

    def is_armed(self) -> bool:
        return self._armed.is_set()

    def _run(self) -> None:
        try:
            import msvcrt  # Windows-only
        except Exception:
            return

        while not self._stop.is_set():
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch and ch.lower() == self.key:
                        self._armed.set()
                else:
                    time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

