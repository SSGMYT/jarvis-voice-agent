from __future__ import annotations

import collections
import json
import queue
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model


def _int16_bytes_from_float32(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def list_devices() -> list[dict]:
    devices = sd.query_devices()
    out: list[dict] = []
    for i, d in enumerate(devices):
        out.append(
            {
                "index": i,
                "name": d.get("name"),
                "max_input_channels": d.get("max_input_channels"),
                "max_output_channels": d.get("max_output_channels"),
                "default_samplerate": d.get("default_samplerate"),
            }
        )
    return out


@dataclass(frozen=True)
class SpeechSegment:
    pcm16: bytes
    sample_rate: int

    def to_wav_bytes(self) -> bytes:
        import io

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.pcm16)
        return buf.getvalue()


class LiveListener:
    """
    Captures mic audio and cuts segments using a simple energy-based VAD.
    Emits segments of PCM16 mono @ sample_rate.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        block_ms: int,
        device: str | int | None,
        vad_mode: int,
        end_silence_ms: int,
        pre_roll_ms: int,
        max_segment_s: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.device = device
        # Energy threshold is derived from vad_mode for a simple "sensitivity" knob.
        # Higher vad_mode => more aggressive => higher threshold.
        self.energy_threshold = {0: 0.010, 1: 0.013, 2: 0.016, 3: 0.020}.get(int(vad_mode), 0.016)
        self.end_silence_ms = end_silence_ms
        self.pre_roll_blocks = max(0, int(pre_roll_ms / block_ms))
        self.max_blocks = max(1, int((max_segment_s * 1000) / block_ms))
        self._q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            # Drop status messages on the floor; they’re usually under/overruns.
            pass
        # indata is float32 in [-1, 1], shape (frames, channels)
        self._q.put(indata.copy())

    def segments(self) -> Iterable[SpeechSegment]:
        frames_per_block = int(self.sample_rate * (self.block_ms / 1000.0))
        ring = collections.deque(maxlen=max(1, self.pre_roll_blocks))

        in_speech = False
        silence_blocks = 0
        seg_blocks: list[bytes] = []

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frames_per_block,
            device=self.device,
            callback=self._callback,
        ):
            while True:
                block = self._q.get()
                if block.shape[0] != frames_per_block:
                    # Guard: some drivers can vary.
                    continue

                mono = block[:, 0]
                pcm16 = _int16_bytes_from_float32(block[:, 0])
                rms = float(np.sqrt(np.mean(mono * mono)))
                is_speech = rms >= self.energy_threshold

                if not in_speech:
                    ring.append(pcm16)
                    if is_speech:
                        in_speech = True
                        seg_blocks = list(ring)
                        ring.clear()
                        silence_blocks = 0
                else:
                    seg_blocks.append(pcm16)
                    if is_speech:
                        silence_blocks = 0
                    else:
                        silence_blocks += 1

                    elapsed_blocks = len(seg_blocks)
                    if elapsed_blocks >= self.max_blocks:
                        in_speech = False
                        yield SpeechSegment(pcm16=b"".join(seg_blocks), sample_rate=self.sample_rate)
                        seg_blocks = []
                        continue

                    if (silence_blocks * self.block_ms) >= self.end_silence_ms:
                        in_speech = False
                        yield SpeechSegment(pcm16=b"".join(seg_blocks), sample_rate=self.sample_rate)
                        seg_blocks = []


class VoskTranscriber:
    def __init__(self, model_path: str, sample_rate: int) -> None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'. Download a model and set VOSK_MODEL_PATH."
            )
        self.model = Model(str(p))
        self.sample_rate = sample_rate

    def transcribe(self, pcm16: bytes) -> str:
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)
        rec.AcceptWaveform(pcm16)
        result = json.loads(rec.FinalResult() or "{}")
        text = (result.get("text") or "").strip()
        return text
