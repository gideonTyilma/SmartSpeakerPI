set -euo pipefail

mkdir -p baby_speaker/{audio,stt,tts,llm,skills,runtime/prompts,util}

# pyproject
cat > pyproject.toml <<'EOF'
[project]
name = "baby-speaker"
version = "0.1.0"
description = "Smart Speaker core (platform-agnostic) with GPT, STT, TTS"
requires-python = ">=3.9"
dependencies = [
  "python-dotenv>=1.0.0",
  "sounddevice>=0.4.6",
  "webrtcvad>=2.0.10",
  "openai>=1.0.0",
  "pydub>=0.25.1",
  "ffmpeg-python>=0.2.0",
  "pyyaml>=6.0",
]
[project.scripts]
baby-speaker = "baby_speaker.main:cli"
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
EOF

# env example
cat > .env.example <<'EOF'
# Copy to .env and set your key:
OPENAI_API_KEY=your_api_key_here
EOF

# package markers
printf "# Package marker\n" > baby_speaker/__init__.py
printf "# audio package\n" > baby_speaker/audio/__init__.py
printf "# stt package\n" > baby_speaker/stt/__init__.py
printf "# tts package\n" > baby_speaker/tts/__init__.py
printf "# llm package\n" > baby_speaker/llm/__init__.py
printf "# skills package\n" > baby_speaker/skills/__init__.py
printf "# util package\n" > baby_speaker/util/__init__.py

# config
cat > baby_speaker/config.py <<'EOF'
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os, yaml

@dataclass
class Settings:
    openai_api_key: str
    stt_backend: str = "openai"
    tts_backend: str = "openai"
    sample_rate: int = 16000
    vad_level: int = 2
    input_device: str | None = None   # None = system default (Mac-friendly)
    output_device: str | None = None
    max_chars: int = 350
    block_ms: int = 20

def load_settings() -> Settings:
    load_dotenv(override=False)
    cfg = {}
    yml = Path(__file__).parent / "runtime" / "config.yaml"
    if yml.exists():
        cfg = yaml.safe_load(yml.read_text()) or {}
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        stt_backend=cfg.get("stt_backend","openai"),
        tts_backend=cfg.get("tts_backend","openai"),
        sample_rate=int(cfg.get("sample_rate",16000)),
        vad_level=int(cfg.get("vad_level",2)),
        input_device=cfg.get("input_device", None),
        output_device=cfg.get("output_device", None),
        max_chars=int(cfg.get("max_chars",350)),
        block_ms=int(cfg.get("block_ms",20)),
    )
EOF

# audio io
cat > baby_speaker/audio/io.py <<'EOF'
import sounddevice as sd
import queue
from typing import Optional

class AudioIO:
    """
    Minimal, platform-agnostic audio I/O using sounddevice/PortAudio.
    16-bit mono PCM at a fixed sample rate.
    """
    def __init__(self, input_device: Optional[str], output_device: Optional[str],
                 sample_rate=16000, block_ms=20):
        self.sample_rate = sample_rate
        self.block = int(sample_rate * block_ms / 1000)
        self.q_in, self.q_out = queue.Queue(), queue.Queue()
        self.in_stream = sd.InputStream(
            device=input_device, channels=1, samplerate=sample_rate,
            dtype="int16", blocksize=self.block, callback=self._in_cb
        )
        self.out_stream = sd.OutputStream(
            device=output_device, channels=1, samplerate=sample_rate,
            dtype="int16", blocksize=self.block, callback=self._out_cb
        )

    def _in_cb(self, indata, frames, time, status):
        if status: print("Input status:", status)
        self.q_in.put(bytes(indata))

    def _out_cb(self, outdata, frames, time, status):
        if status: print("Output status:", status)
        try:
            chunk = self.q_out.get_nowait()
        except queue.Empty:
            chunk = b"\x00\x00" * frames
        outdata[:] = chunk

    def start(self):
        self.in_stream.start(); self.out_stream.start()
    def stop(self):
        self.in_stream.stop(); self.out_stream.stop()

    def read(self) -> bytes:
        return self.q_in.get()
    def play(self, pcm_bytes: bytes):
        self.q_out.put(pcm_bytes)
EOF

# vad
cat > baby_speaker/audio/vad.py <<'EOF'
import webrtcvad

class VADGate:
    """Simple VAD gate with fixed frame size (matches AudioIO block)."""
    def __init__(self, level=2, sample_rate=16000, frame_ms=20):
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # int16 mono
        self.vad = webrtcvad.Vad(level)

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != self.frame_bytes:
            return False
        return self.vad.is_speech(frame, self.sample_rate)
EOF

# llm
cat > baby_speaker/llm/openai_chat.py <<'EOF'
import os, time, random
from openai import OpenAI

class ChatLLM:
    """Minimal OpenAI chat wrapper with retry & max-length trim."""
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system = system_prompt
        self.model = model

    def _backoff(self, fun, tries=5, base=0.5, cap=6):
        for i in range(tries):
            try: return fun()
            except Exception:
                if i == tries-1: raise
                time.sleep(min(cap, base*(2**i) + random.random()*0.2))

    def ask(self, user_text: str, max_chars=350) -> str:
        def call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role":"system","content":self.system},
                    {"role":"user","content":user_text}
                ],
                temperature=0.4,
                max_tokens=512
            )
            return resp.choices[0].message.content.strip()
        txt = self._backoff(call)
        return txt[:max_chars]
EOF

# stt
cat > baby_speaker/stt/openai_stt.py <<'EOF'
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
EOF

# tts
cat > baby_speaker/tts/openai_tts.py <<'EOF'
import io
from openai import OpenAI
import os
from pydub import AudioSegment

class OpenAI_TTS:
    """
    Synthesizes speech via OpenAI TTS -> WAV, then resamples to 16k mono int16 PCM.
    """
    def __init__(self, model: str = "tts-1", voice: str = "alloy", target_sr: int = 16000):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.voice = voice
        self.target_sr = target_sr

    def _wav_bytes_to_pcm16_mono_16k(self, wav_bytes: bytes) -> bytes:
        audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        audio = audio.set_frame_rate(self.target_sr).set_channels(1).set_sample_width(2)  # int16
        out_buf = io.BytesIO()
        audio.export(out_buf, format="raw", codec="pcm_s16le")
        return out_buf.getvalue()

    def synth(self, text: str) -> bytes:
        resp = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            format="wav",
        )
        wav_bytes = resp.content
        return self._wav_bytes_to_pcm16_mono_16k(wav_bytes)
EOF

# skills
cat > baby_speaker/skills/baby_assistant.py <<'EOF'
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "runtime" / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "system.txt").read_text()
RED_FLAGS = set((PROMPTS_DIR / "red_flags.txt").read_text().splitlines())

def postprocess(user_text: str, llm_text: str) -> str:
    low = user_text.lower()
    if any(flag in low for flag in RED_FLAGS):
        return ("This may be urgent. If baby has trouble breathing, blue lips, is unresponsive, "
                "or a fever ≥100.4°F under 3 months, call emergency services or your pediatrician now.")
    return llm_text
EOF

# runtime config + prompts
cat > baby_speaker/runtime/config.yaml <<'EOF'
stt_backend: openai
tts_backend: openai
sample_rate: 16000
vad_level: 2
# input_device: null   # use system default on macOS
# output_device: null  # use system default on macOS
max_chars: 350
block_ms: 20
EOF

cat > baby_speaker/runtime/prompts/system.txt <<'EOF'
You are BabyCare Assistant: friendly, brief (≤2 sentences), non-diagnostic.
If severe red flags (difficulty breathing, blue lips, unresponsive, seizure,
or under 3 months with fever ≥100.4°F), say: "This sounds urgent—call emergency
services or your pediatrician now." Otherwise, give practical, safe steps.
EOF

cat > baby_speaker/runtime/prompts/red_flags.txt <<'EOF'
fever
breathing
blue lips
unresponsive
seizure
choking
not waking
not feeding
dehydrated
no wet diapers
EOF

# util logging
cat > baby_speaker/util/logging.py <<'EOF'
import logging, sys

def setup_logger(name="baby_speaker", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid dupes
        return logger
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(level)
    return logger
EOF

# main
cat > baby_speaker/main.py <<'EOF'
from .config import load_settings
from .audio.io import AudioIO
from .audio.vad import VADGate
from .skills.baby_assistant import SYSTEM_PROMPT, postprocess
from .llm.openai_chat import ChatLLM
from .stt.openai_stt import OpenAI_STT
from .tts.openai_tts import OpenAI_TTS
from .util.logging import setup_logger

def run_loop():
    log = setup_logger()
    cfg = load_settings()
    log.info("Starting baby-speaker (no-PI refactor).")
    log.info(f"Audio SR={cfg.sample_rate}, block={cfg.block_ms}ms, VAD={cfg.vad_level}")

    audio = AudioIO(cfg.input_device, cfg.output_device, cfg.sample_rate, cfg.block_ms)
    vad = VADGate(cfg.vad_level, cfg.sample_rate, cfg.block_ms)
    stt = OpenAI_STT()
    tts = OpenAI_TTS()
    llm = ChatLLM(SYSTEM_PROMPT)

    audio.start()
    try:
        buf = b""
        silence = 0
        # Simple endpointing: accumulate speech; on short silence, finalize
        while True:
            frame = audio.read()
            if vad.is_speech(frame):
                buf += frame
                silence = 0
            else:
                if buf:
                    silence += 1
                    # ~100ms of silence if 20ms frames
                    if silence > 5:
                        # === STT ===
                        text = stt.transcribe(wav_bytes=buf)
                        buf = b""
                        silence = 0
                        if not text:
                            continue
                        log.info(f"User: {text}")
                        # === LLM ===
                        reply = llm.ask(text, max_chars=cfg.max_chars)
                        reply = postprocess(text, reply)
                        log.info(f"Assistant: {reply}")
                        # === TTS ===
                        wav = tts.synth(reply)
                        audio.play(wav)
    finally:
        audio.stop()

def cli():
    run_loop()
EOF
