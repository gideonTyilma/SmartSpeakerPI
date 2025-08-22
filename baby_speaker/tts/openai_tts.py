import io
from openai import OpenAI
import os
from pydub import AudioSegment
from typing import Tuple  

class OpenAI_TTS:
    """
    Synthesizes speech via OpenAI TTS -> WAV, then resamples to 16k mono int16 PCM.
    """
    def __init__(self, model: str = "tts-1", voice: str = "alloy", target_sr: int = 16000):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.voice = voice
        self.target_sr = target_sr


    def _bytes_to_pcm16_mono_16k(self, audio_bytes: bytes) -> bytes:
        """
        Convert arbitrary audio bytes (WAV or MP3) to raw 16kHz mono int16 PCM (s16le).
        We sniff the header to pick the right decoder for pydub/ffmpeg.
        """
        header4 = audio_bytes[:4]
        fmt = "wav"
        # WAV: starts with 'RIFF'
        if header4.startswith(b"RIFF"):
            fmt = "wav"
        # MP3: starts with 'ID3' or 0xFF sync word
        elif audio_bytes[:3] == b"ID3" or (len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
            fmt = "mp3"

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        audio = audio.set_frame_rate(self.target_sr).set_channels(1).set_sample_width(2)  # int16
        out_buf = io.BytesIO()
        audio.export(out_buf, format="s16le")
        return out_buf.getvalue()

    def synth(self, text: str) -> bytes:
        # Use streaming API to get raw WAV bytes (default container)
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
        ) as resp:
            wav_bytes = resp.read()

        return self._bytes_to_pcm16_mono_16k(wav_bytes)