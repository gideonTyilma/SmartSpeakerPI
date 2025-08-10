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

VOICE_NAMES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse"
]

PROMPT_OPTIONS = [
    "You are a helpful assistant. kinda like Jarvis from Iron man. Answer with clarity, concise responses.",
    "You are a friendly and cheerful assistant who loves to help with a positive attitude.",
    "You are a professional and formal assistant, providing detailed and precise answers.",
    "You are a witty and humorous assistant who makes conversations fun and engaging."
]

# Prompt aliases for easy switching
PROMPT_ALIASES = {
    "concise": 0,
    "cheerful": 1,
    "formal": 2,
    "witty": 3
}

def build_system_prompt(persona_idx: int, voice_name: str, assistant_name: str) -> str:
    return (
        f"{PROMPT_OPTIONS[persona_idx]} "
        f"You are an AI home assistant named {assistant_name}. "
        f"Your current text-to-speech voice preset is '{voice_name}'. "
        "The voice preset is not the user's name. "
        "Never assume the user's name unless they tell you. "
        "If asked 'what is my name?', say you don't know. "
        f"If asked 'what is your name?', say '{assistant_name}'."
    )

print("Select a voice:")
for idx, name in enumerate(VOICE_NAMES):
    print(f"{idx}: {name}")
voice_choice = input("Enter the number of your choice: ")
try:
    voice_idx = int(voice_choice)
    if voice_idx < 0 or voice_idx >= len(VOICE_NAMES):
        raise ValueError
except ValueError:
    print("Invalid choice, defaulting to 'nova'.")
    voice_idx = 7
VOICE_NAME = VOICE_NAMES[voice_idx]

# Choose an assistant display name (separate from voice preset)
default_name = "Jarvis"
name_choice = input(f"\nAssistant display name (leave empty for '{default_name}'): ").strip()
ASSISTANT_NAME = name_choice if name_choice else default_name

print("\nSelect a system prompt/personality:")
for idx, prompt in enumerate(PROMPT_OPTIONS):
    print(f"{idx}: {prompt}")
prompt_choice = input("Enter the number of your choice: ")
try:
    prompt_idx = int(prompt_choice)
    if prompt_idx < 0 or prompt_idx >= len(PROMPT_OPTIONS):
        raise ValueError
except ValueError:
    print("Invalid choice, defaulting to first prompt.")
    prompt_idx = 0

CURRENT_PROMPT_IDX = prompt_idx
SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)

# --- Helper functions for mid-chat switching ---
def choose_voice_interactive():
    global VOICE_NAME, ASSISTANT_NAME, SYSTEM_PROMPT
    print("\n[Voice Switch] Available voices:")
    for idx, name in enumerate(VOICE_NAMES):
        print(f"{idx}: {name}")
    choice = input("Enter voice number or name: ").strip().lower()
    target = None
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(VOICE_NAMES):
            target = VOICE_NAMES[idx]
    else:
        for name in VOICE_NAMES:
            if name.lower() == choice:
                target = name
                break
    if not target:
        print("Invalid choice. Voice unchanged.")
        return False
    VOICE_NAME = target
    SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
    print(f"[Voice Switch] Voice set to '{VOICE_NAME}'.")
    return True

def choose_personality_interactive():
    global SYSTEM_PROMPT, CURRENT_PROMPT_IDX
    print("\n[Personality Switch] Options:")
    for idx, prompt in enumerate(PROMPT_OPTIONS):
        print(f"{idx}: {prompt}")
    choice = input("Enter personality number or alias (concise/cheerful/formal/witty): ").strip().lower()
    target_idx = None
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(PROMPT_OPTIONS):
            target_idx = idx
    else:
        if choice in PROMPT_ALIASES:
            target_idx = PROMPT_ALIASES[choice]
    if target_idx is None:
        print("Invalid choice. Personality unchanged.")
        return False
    CURRENT_PROMPT_IDX = target_idx
    SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
    print(f"[Personality Switch] Set to option {target_idx}.")
    return True

def closest_voice_or_none(name: str):
    lname = name.strip().lower()
    for v in VOICE_NAMES:
        if v.lower() == lname:
            return v
    return None

def parse_and_handle_meta(transcript: str) -> bool:
    """Return True if we handled a meta command like switching voice/personality."""
    global VOICE_NAME, SYSTEM_PROMPT, ASSISTANT_NAME, CURRENT_PROMPT_IDX
    t = transcript.strip().lower()
    # Slash commands (typed)
    if t.startswith("/voice"):
        parts = t.split()
        if len(parts) >= 2:
            cand = closest_voice_or_none(parts[1])
            if cand:
                VOICE_NAME = cand
                SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
                print(f"[Voice Switch] Voice set to '{VOICE_NAME}'.")
            else:
                print("[Voice Switch] Unknown voice. Launching selector.")
                choose_voice_interactive()
        else:
            choose_voice_interactive()
        return True
    if t.startswith("/persona") or t.startswith("/personality"):
        parts = t.split()
        if len(parts) >= 2:
            arg = parts[1]
            if arg.isdigit():
                idx = int(arg)
                if 0 <= idx < len(PROMPT_OPTIONS):
                    CURRENT_PROMPT_IDX = idx
                    SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
                    print(f"[Personality Switch] Set to option {idx}.")
                else:
                    print("[Personality Switch] Invalid index. Launching selector.")
                    choose_personality_interactive()
            else:
                alias = arg.lower()
                if alias in PROMPT_ALIASES:
                    CURRENT_PROMPT_IDX = PROMPT_ALIASES[alias]
                    SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
                    print(f"[Personality Switch] Set to '{alias}'.")
                else:
                    print("[Personality Switch] Unknown alias. Launching selector.")
                    choose_personality_interactive()
        else:
            choose_personality_interactive()
        return True
    if t.startswith("/name"):
        parts = t.split(maxlen:=2)
        if len(parts) >= 2:
            new_name = parts[1].strip()
            if new_name:
                ASSISTANT_NAME = new_name
                SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
                print(f"[Name] Assistant name set to '{ASSISTANT_NAME}'.")
            else:
                print("[Name] Please provide a non-empty name.")
        else:
            print("[Name] Usage: /name <assistant name>")
        return True
    # Spoken commands (via STT)
    if t in {"switch voice", "change voice"}:
        choose_voice_interactive()
        return True
    if t.startswith("set voice to ") or t.startswith("voice "):
        name = t.replace("set voice to ", "").replace("voice ", "").strip()
        cand = closest_voice_or_none(name)
        if cand:
            VOICE_NAME = cand
            SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
            print(f"[Voice Switch] Voice set to '{VOICE_NAME}'.")
        else:
            print("[Voice Switch] Unknown voice. Launching selector.")
            choose_voice_interactive()
        return True
    if t in {"switch personality", "change personality"}:
        choose_personality_interactive()
        return True
    if t.startswith("set personality to ") or t.startswith("personality "):
        arg = t.replace("set personality to ", "").replace("personality ", "").strip()
        # Try numeric index first
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(PROMPT_OPTIONS):
                CURRENT_PROMPT_IDX = idx
                SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
                print(f"[Personality Switch] Set to option {idx}.")
            else:
                print("[Personality Switch] Invalid index. Launching selector.")
                choose_personality_interactive()
        else:
            alias = arg.lower()
            if alias in PROMPT_ALIASES:
                CURRENT_PROMPT_IDX = PROMPT_ALIASES[alias]
                SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
                print(f"[Personality Switch] Set to '{alias}'.")
            else:
                print("[Personality Switch] Unknown alias. Launching selector.")
                choose_personality_interactive()
        return True
    if t.startswith("call yourself "):
        new_name = t.replace("call yourself ", "").strip()
        if new_name:
            ASSISTANT_NAME = new_name
            SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
            print(f"[Name] Assistant name set to '{ASSISTANT_NAME}'.")
        else:
            print("[Name] Please provide a non-empty name.")
        return True
    if t.startswith("your name is "):
        new_name = t.replace("your name is ", "").strip()
        if new_name:
            ASSISTANT_NAME = new_name
            SYSTEM_PROMPT = build_system_prompt(CURRENT_PROMPT_IDX, VOICE_NAME, ASSISTANT_NAME)
            print(f"[Name] Assistant name set to '{ASSISTANT_NAME}'.")
        else:
            print("[Name] Please provide a non-empty name.")
        return True
    return False

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
    # Assistant introduces themselves audibly
    intro_text = f"Hello, I am {ASSISTANT_NAME}, your assistant, using the {VOICE_NAME} voice."
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as intro_wav:
        tts_to_wav_file(intro_text, intro_wav.name)
        play_wav(intro_wav.name)
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

            # Handle meta commands (switch voice/personality) without calling GPT
            if parse_and_handle_meta(transcript):
                # Give a brief audible confirmation using the (possibly new) voice
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_wav:
                    tts_to_wav_file("Settings updated.", tts_wav.name)
                    play_wav(tts_wav.name)
                continue

            print(f"{ASSISTANT_NAME}: thinking...")
            # 4) Get concise reply
            reply = get_chat_reply(transcript)
            print("answered")

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