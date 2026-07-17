from livekit.agents import stt, utils
from faster_whisper import WhisperModel
import numpy as np


class FasterWhisperSTT(stt.STT):
    def __init__(self, model_size: str = "base.en"):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    async def _recognize_impl(
        self, buffer: utils.AudioBuffer, *, language=None, conn_options=None
    ) -> stt.SpeechEvent:
        audio_data = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio_data, language=language)
        text = " ".join(seg.text for seg in segments)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or "en")],
        )