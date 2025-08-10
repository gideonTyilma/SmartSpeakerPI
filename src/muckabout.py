# push_to_talk_low_latency.py
import os
import io
import sys
import wave
import tempfile
import numpy as np

import sounddevice as sd
import soundfile as sf   # pip install soundfile
import keyboard          # may need Accessibility perms on macOS

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env if present

# ---- OpenAI client (v1 style) ----
try:
    from openai import OpenAI
except ImportError:
    print("Please 'pip install openai' (>=1.0).")
    sys.exit(1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    print("Missing OPENAI_API_KEY. Put it in .env or export it.")
    sys.exit(1)

# Optional: silence trimming (uses pydub; ffmpeg recommended)
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# -------- Audio constants --------
SAMPLE_RATE = 16000  # end-to-end 16 kHz mono
CHANNELS = 1
DTYPE = np.int16
FRAME_SAMPLES = 1024  # small frames to reduce capture latency

VOICE_NAME = "verse"  # change to another built-in voice if you like

SYSTEM_PROMPT = (
    "You are a futuristic home assistant named Jarvis. stay upbeat"
    "Answer in a few sentences unless asked for detail."
)

def record_while_space_held():
    """Press/hold Space to record; release to stop."""
    print("Hold SPACE to talk… (press Ctrl+C to quit)")
    # Wait until space is pressed
    while not keyboard.is_pressed("space"):
        pass
    print("Recording…")
    frames = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE) as stream:
        while keyboard.is_pressed("space"):
            data, _ = stream.read(FRAME_SAMPLES)
            frames.append(data.copy())
    print("Stopped.")
    if not frames:
        return np.zeros((0,), dtype=DTYPE)
    return np.concatenate(frames, axis=0)

def save_wav(np_audio: np.ndarray, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np_audio.tobytes())

def trim_silence_wav(in_wav_path: str, out_wav_path: str,
                     silence_thresh_db: int = -40,
                     min_silence_len_ms: int = 150):
    """Trim leading/trailing silence to reduce STT upload/latency."""
    audio = AudioSegment.from_wav(in_wav_path)
    spans = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db
    )
    if not spans:
        # Nothing detected — write a tiny slice to avoid errors
        audio[:1].export(out_wav_path, format="wav")
        return
    start, end = spans[0][0], spans[-1][1]
    audio[start:end].export(out_wav_path, format="wav")

def transcribe_wav(path: str) -> str:
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    # Some SDKs return .text, others embed in .text/.transcript; guard:
    text = getattr(result, "text", None) or getattr(result, "transcript", "")
    return (text or "").strip()

def get_chat_reply(user_text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        temperature=0.3,
        max_tokens=80
    )
    return resp.choices[0].message.content.strip()

def tts_to_wav_file(text: str, out_path: str):
    # Ask for WAV so we can play directly
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=VOICE_NAME,
        response_format="wav",
        input=text
    ) as response:
        response.stream_to_file(out_path)

def play_wav(path: str):
    data, samplerate = sf.read(path, dtype="int16")
    sd.play(data, samplerate)
    sd.wait()

def main():
    print("Low-latency push-to-talk ready.")
    print("Tip: make AIRHUG your default input/output in macOS Audio MIDI Setup.")
    try:
        while True:
            # 1) Capture
            audio = record_while_space_held()
            if audio.size == 0:
                continue

            # 2) Save and trim to reduce STT time
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as raw_wav, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as trimmed_wav:
                save_wav(audio, raw_wav.name)
                trim_silence_wav(raw_wav.name, trimmed_wav.name)

                # 3) Transcribe
                transcript = transcribe_wav(trimmed_wav.name)

            if not transcript:
                print("(heard nothing intelligible)")
                continue

            print(f"You: {transcript}")
            if transcript.lower() in {"quit", "exit", "goodbye"}:
                print("Bye!")
                break

            print(f"Jarvis: Thinking...")
            # 4) Get concise reply
            reply = get_chat_reply(transcript)
            print(f"Jarvis: Done.")

            # 5) TTS as WAV + immediate playback
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_wav:
                tts_to_wav_file(reply, tts_wav.name)
                play_wav(tts_wav.name)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        sd.stop()

if __name__ == "__main__":
    main()