import io, wave
from openai import OpenAI
import os

class OpenAI_STT:
    """
    Transcribes 16 kHz mono int16 PCM bytes using OpenAI Whisper.
    """
    def __init__(self, model: str = "whisper-1", sample_rate: int = 16000):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.sample_rate = sample_rate

    def _pcm16_to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        # Wrap raw PCM in a minimal WAV header so the API knows format.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def transcribe(self, wav_bytes: bytes) -> str:
        """
        Orchestrator passes raw PCM; convert to WAV container and send to Whisper.
        """
        wav_container = self._pcm16_to_wav_bytes(wav_bytes)
        f = io.BytesIO(wav_container)
        f.name = "speech.wav"
        resp = self.client.audio.transcriptions.create(
            model=self.model,
            file=f
        )
        text = getattr(resp, "text", "").strip()
        return text
