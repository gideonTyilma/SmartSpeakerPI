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
import re

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

# -------- Debugging --------
DEBUG = True   # leave on; logs aren't user-facing
COLOR = True   # set False if you ever want plain logs

# Minimal ANSI color map per category tag
_COL = {
    "intro": "\033[32m",      # green
    "stt": "\033[34m",        # blue
    "chat": "\033[36m",       # cyan
    "cont": "\033[35m",       # magenta
    "cont-gate": "\033[95m",  # bright magenta
    "ctx": "\033[33m",        # yellow
    "META": "\033[90m",       # grey
    "META?": "\033[90m",      # grey
    "err": "\033[31m",        # red
    "": "\033[37m",           # default grey
    "reset": "\033[0m",
}

def log(msg: str):
    """Print a standardized debug line. If `msg` starts with a category in square
    brackets like "[chat] ...", we'll render it as [DBG][chat] with color."""
    if not DEBUG:
        return
    tag = ""
    body = msg
    if isinstance(msg, str) and msg.startswith("[") and "]" in msg:
        tag = msg[1:msg.find("]")]
        body = msg[msg.find("]")+1:].lstrip()
    if COLOR:
        color = _COL.get(tag, _COL[""])
        reset = _COL["reset"]
        print(f"{color}[DBG]{f'[{tag}]' if tag else ''} {body}{reset}")
    else:
        print(f"[DBG]{f'[{tag}]' if tag else ''} {body}")

def _preview(s: str, n: int = 140) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"

# -------- Reply length controls --------
MAX_TOKENS = 160              # hard cap for the primary reply
BRIEF_WORD_LIMIT = 55         # if reply likely exceeds this, switch to brief mode
BRIEF_MAX_TOKENS = 110        # cap for the brief reply

# -------- Conversation history (for follow-ups) --------
HISTORY_MAX_MESSAGES = 20  # keep last N messages (role/content items)

def _append_history(role: str, content: str):
    CONTEXT.setdefault("history", []).append({"role": role, "content": content})
    # Trim to last N messages to bound context size
    if len(CONTEXT["history"]) > HISTORY_MAX_MESSAGES:
        CONTEXT["history"] = CONTEXT["history"][-HISTORY_MAX_MESSAGES:]

# -------- Continuation state --------
CONTEXT = {
    "active": False,
    "topic": None,
    "round": 0,
    "history": []  # running conversation history for continuation
}

AFFIRMATIVE_KEYS = {
    "yes","y","yeah","yep","sure","ok","okay","pls","please",
    "continue","go on","more","keep going","go ahead","please continue",
    "yes please","sure please","continue please"
}
NEGATIVE_KEYS = {
    "no","n","nope","nah","stop","cancel","that's all","thats all",
    "no thank you","no thanks","not now","we're good","were good","all set","im good","i'm good"
}

def _normalize_phrase(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _is_affirmative(s: str) -> bool:
    t = _normalize_phrase(s)
    if t in AFFIRMATIVE_KEYS:
        return True
    return any(k in t for k in ("continue", "go ahead", "keep going", "go on", "more"))

def _is_negative(s: str) -> bool:
    t = _normalize_phrase(s)
    if t in NEGATIVE_KEYS:
        return True
    return any(k in t for k in ("stop", "cancel", "no thanks", "no thank you", "that's all", "thats all"))

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

BABY_PERSONA = (
    "You are a warm, gentle, encouraging 'baby assistant', meaning you assist in the care of infants. "
    "You help with tasks, questions, and guidance related to caring for a baby, such as feeding schedules, sleep routines, and soothing techniques. "
    "You communicate in a calm, patient, and reassuring tone, using clear, simple language suitable for stressed or tired caregivers. "
    "You provide safe, practical, and evidence-based advice, avoiding medical diagnoses unless quoting reputable sources. "
    "You remain supportive and empathetic, never judgmental, and you focus on helping the caregiver feel confident and informed."
    "If unsure about a recommendation, suggest consulting a pediatrician or trusted healthcare provider."
    "If the user sounds stressed or worried, respond first with a short reassurance before giving advice."
    "If the question is urgent (e.g., crying, choking, fever), give the most important step first in plain language before offering extra detail."
    "When suggesting any physical interaction with the baby, remind the caregiver of safe handling basics."
    "Replace “don’t” phrasing with gentle guidance (“Instead of letting the baby sleep in the car seat for long, try…”)."

)

def build_system_prompt(voice_name: str, assistant_name: str) -> str:
    return (
        f"You are an AI home assistant named {assistant_name}. "
        f"Your current text-to-speech voice preset is '{voice_name}'. "
        "The voice preset is not the user's name. "
        "Never assume the user's name unless they tell you. "
        "If asked 'what is my name?', say you don't know. "
        f"If asked 'what is your name?', say '{assistant_name}'."
        f"{BABY_PERSONA} "
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

SYSTEM_PROMPT = build_system_prompt(VOICE_NAME, ASSISTANT_NAME)

# --- Helper functions for mid-chat switching ---
def choose_voice_interactive():
    global VOICE_NAME, SYSTEM_PROMPT
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
    SYSTEM_PROMPT = build_system_prompt(VOICE_NAME, ASSISTANT_NAME)
    print(f"[Voice Switch] Voice set to '{VOICE_NAME}'.")
    return True


def closest_voice_or_none(name: str):
    lname = name.strip().lower()
    for v in VOICE_NAMES:
        if v.lower() == lname:
            return v
    return None

def parse_and_handle_meta(transcript: str) -> bool:
    """Return True if we handled a meta command like switching voice/personality."""
    global VOICE_NAME, SYSTEM_PROMPT, ASSISTANT_NAME
    t = transcript.strip().lower()
    log(f"[META?] raw='{_preview(transcript)}' lower='{_preview(t)}'")
    # Slash commands (typed)
    if t.startswith("/voice"):
        parts = t.split()
        if len(parts) >= 2:
            cand = closest_voice_or_none(parts[1])
            if cand:
                VOICE_NAME = cand
                SYSTEM_PROMPT = build_system_prompt(VOICE_NAME, ASSISTANT_NAME)
                log(f"[META] set voice to {VOICE_NAME}")
            else:
                log("[META] unknown voice, launching selector")
                choose_voice_interactive()
        else:
            log("[META] launching voice selector")
            choose_voice_interactive()
        return True
    if t.startswith("/name"):
        parts = t.split(maxsplit=1)
        if len(parts) >= 2:
            new_name = parts[1].strip()
            if new_name:
                ASSISTANT_NAME = new_name
                SYSTEM_PROMPT = build_system_prompt(VOICE_NAME, ASSISTANT_NAME)
                log(f"[META] set name to {ASSISTANT_NAME}")
            else:
                log("[META] empty name provided")
        else:
            log("[META] usage: /name <assistant name>")
        return True
    # Spoken commands (via STT)
    if t in {"switch voice", "change voice"}:
        log("[META] launching voice selector (spoken)")
        choose_voice_interactive()
        return True
    if t.startswith("set voice to ") or t.startswith("voice "):
        name = t.replace("set voice to ", "").replace("voice ", "").strip()
        cand = closest_voice_or_none(name)
        if cand:
            VOICE_NAME = cand
            SYSTEM_PROMPT = build_system_prompt(VOICE_NAME, ASSISTANT_NAME)
            log(f"[META] set voice to {VOICE_NAME}")
        else:
            log("[META] unknown voice, launching selector")
            choose_voice_interactive()
        return True
    if t.startswith("call yourself "):
        new_name = t.replace("call yourself ", "").strip()
        if new_name:
            ASSISTANT_NAME = new_name
            SYSTEM_PROMPT = build_system_prompt(VOICE_NAME, ASSISTANT_NAME)
            log(f"[META] set name to {ASSISTANT_NAME}")
        else:
            log("[META] empty name provided")
        return True
    if t.startswith("your name is "):
        new_name = t.replace("your name is ", "").strip()
        if new_name:
            ASSISTANT_NAME = new_name
            SYSTEM_PROMPT = build_system_prompt(VOICE_NAME, ASSISTANT_NAME)
            log(f"[META] set name to {ASSISTANT_NAME}")
        else:
            log("[META] empty name provided")
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


# --- Helper functions for reply truncation detection ---
def _is_incomplete_sentence(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    # consider complete if it ends with typical terminal punctuation
    return not s.endswith((".", "!", "?"))

def _word_count(s: str) -> int:
    return len((s or "").split())

def get_chat_reply(user_text: str) -> tuple[str, bool]:
    # Encourage brevity up front to avoid mid-sentence truncation
    system_content = (
        SYSTEM_PROMPT
        + " Keep answers under two sentences for spoken output. "
          "If the full answer would be long, provide a brief summary and end with: 'Say \"continue\" for more.'"
    )

    log("[chat] requesting primary reply")
    msgs = [{"role": "system", "content": system_content}]
    # include recent conversation history so follow-ups have context
    msgs.extend(CONTEXT.get("history", []))
    msgs.append({"role": "user", "content": user_text})
    log(f"[chat] hist_len={len(CONTEXT.get('history', []))}")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.3,
        max_tokens=MAX_TOKENS
    )
    choice = resp.choices[0]
    text = choice.message.content.strip()
    finish = getattr(choice, "finish_reason", None)
    log(f"[chat] finish_reason={finish} words={_word_count(text)}")
    log(f"[chat] model: {_preview(text)}")
    # Note if the model already included our continuation phrase
    _tlow = text.rstrip().lower()
    if _tlow.endswith('say "continue" for more.') or _tlow.endswith("say 'continue' for more."):
        log("[chat] continue-phrase detected in primary reply")
        return text, True

    too_long = (finish == "length") or _is_incomplete_sentence(text) or (_word_count(text) > BRIEF_WORD_LIMIT)
    if too_long:
        log("[chat] switching to brief mode")
        brief_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": (
                    "Your full answer would be long. Please give a brief, high-level version of your best answer "
                    "in simple, caregiver-friendly language. End with: 'Say \"continue\" for more.'\n\n"
                    f"Question: {user_text}"
                )}
            ],
            temperature=0.2,
            max_tokens=BRIEF_MAX_TOKENS
        )
        brief_text = brief_resp.choices[0].message.content.strip()
        log(f"[chat] brief: {_preview(brief_text)}")
        _b = brief_text.rstrip().lower()
        if _b.endswith('say "continue" for more.') or _b.endswith("say 'continue' for more."):
            log("[chat] continue-phrase present in brief reply")
        return brief_text, True
    
    log(f"[chat] out: {_preview(text)}")
    return text, False

def _sanitize_continuation_text(text: str) -> str:
    """Remove leading greetings or self-introductions so continuations don't start over."""
    if not text:
        return text
    t = text.lstrip()
    lowered = t.lower()
    # Common greeting/self-intro patterns to strip only at the very start
    patterns = [
        ("hello",), ("hi",), ("hey",),
        (f"my name is {ASSISTANT_NAME.lower()}",),
        ("my name is ",),  # any name
        (f"i am {ASSISTANT_NAME.lower()}",),
        ("i am your assistant",),
        ("i am an ai",),
    ]
    changed = True
    # Iteratively strip if multiple greetings stack (e.g., "Hello! Hi again…")
    while changed:
        changed = False
        for p in patterns:
            if lowered.startswith(p[0]):
                # remove up to the first sentence-ending punctuation or line break
                for end_char in ('.', '!', '?', '\n'):
                    idx = t.find(end_char)
                    if idx != -1:
                        t = t[idx+1:].lstrip()
                        lowered = t.lower()
                        changed = True
                        break
                if not changed:
                    # No sentence end found; drop the token itself
                    t = t[len(p[0]):].lstrip()
                    lowered = t.lower()
                    changed = True
                break
    return t if t else text

def get_continuation_reply(topic: str, round_idx: int) -> tuple[str, bool]:
    """Resume where we left off. Returns (text, still_more)."""
    log(f"[cont] hist_len={len(CONTEXT.get('history', []))} round={round_idx} topic_preview='{_preview(topic)}'")
    messages = [
        {
            "role": "system",
            "content": (
                BABY_PERSONA + " "
                "You are continuing the SAME answer for the SAME question in this conversation. "
                "Do NOT greet or introduce yourself. Do NOT say your name or mention your voice. "
                "Do NOT restate the question and do NOT repeat earlier points. "
                "Provide the next most important points in short, spoken-friendly sentences. "
                "Finish the last sentence before you stop. "
                "If more remains after this chunk, end with: 'Say \"continue\" for more.'"
            )
        }
    ]
    # include prior turns
    messages.extend(CONTEXT.get("history", []))
    # user affirmation / continuation request for the next chunk
    messages.append({
        "role": "user",
        "content": f"Please continue your previous answer (part {round_idx + 1}). No greetings or introductions."
    })

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=BRIEF_MAX_TOKENS
    )
    choice = resp.choices[0]
    text = choice.message.content.strip()
    finish = getattr(choice, "finish_reason", None)
    log(f"[cont] model: {_preview(text)}")
    log(f"[cont] finish_reason={finish} words={_word_count(text)}")

    # Sanitize any stray greeting/intro at the start
    orig = text
    text = _sanitize_continuation_text(text)
    if text != orig:
        log("[cont] sanitizer removed greeting/intro")
    else:
        log("[cont] no sanitize needed")

    # Determine if there is likely more content
    tlow = text.rstrip().lower()
    has_phrase = tlow.endswith('say "continue" for more.') or tlow.endswith("say 'continue' for more.")
    incomplete = _is_incomplete_sentence(text)
    hit_cap = (finish == "length")
    still_more = has_phrase or incomplete or hit_cap

    # If we hit the cap or ended mid-sentence but the hint is missing, append it client-side
    if (incomplete or hit_cap) and not has_phrase:
        text = (text.rstrip() + " Say 'continue' for more.").strip()
        log("[cont] appended continue hint client-side due to truncation")

    log(f"[cont] out: {_preview(text)}")
    log(f"[cont] still_more={still_more} words={_word_count(text)}")
    # Append this chunk to history
    CONTEXT.setdefault("history", []).append({"role": "assistant", "content": text})
    return text, still_more

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
    intro_text, _intro_brief = get_chat_reply("Please introduce yourself to the user.")
    log(f"[intro] {_preview(intro_text)}")
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

            log(f"[stt] Transcript: {_preview(transcript)}")
            norm = _normalize_phrase(transcript)
            log(f"[stt] Norm: '{norm}' | active={CONTEXT['active']} round={CONTEXT['round']} hist={len(CONTEXT.get('history', []))}")
            log(f"[stt] Flags: affirm={_is_affirmative(transcript)} neg={_is_negative(transcript)}")

            if not transcript:
                print("(heard nothing intelligible)")
                continue

            print(f"You: {transcript}")
            if transcript.lower() in {"quit", "exit", "goodbye"}:
                print("Bye!")
                break

            # If user responded to a continuation prompt
            if CONTEXT["active"]:
                log(f"[cont-gate] active={CONTEXT['active']} neg={_is_negative(transcript)} affirm={_is_affirmative(transcript)}")
                # Negative/cancel first
                if _is_negative(transcript):
                    log("[cont-gate] cancelling continuation")
                    CONTEXT["active"] = False
                    CONTEXT["topic"] = None
                    CONTEXT["round"] = 0
                    log(f"[ctx] set active={CONTEXT['active']} topic='{_preview(CONTEXT['topic']) if CONTEXT['topic'] else None}' round={CONTEXT['round']} hist={len(CONTEXT.get('history', []))}")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_wav:
                        tts_to_wav_file("Okay, stopping there.", tts_wav.name)
                        play_wav(tts_wav.name)
                    continue
                # Affirmative -> continue
                if _is_affirmative(transcript):
                    log("[cont-gate] continuing with next chunk")
                    print(f"{ASSISTANT_NAME}: thinking...")
                    CONTEXT.setdefault("history", []).append({"role": "user", "content": "[Affirmation] Please continue."})
                    cont_text, still_more = get_continuation_reply(CONTEXT["topic"], CONTEXT["round"])
                    log(f"[cont] final out: {_preview(cont_text)} | still_more={still_more}")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_wav:
                        tts_to_wav_file(cont_text, tts_wav.name)
                        play_wav(tts_wav.name)
                    print("answered")
                    CONTEXT["round"] += 1
                    CONTEXT["active"] = bool(still_more)
                    log(f"[ctx] set active={CONTEXT['active']} topic='{_preview(CONTEXT['topic']) if CONTEXT['topic'] else None}' round={CONTEXT['round']} hist={len(CONTEXT.get('history', []))}")
                    continue

            # Handle meta commands (switch voice/personality) without calling GPT
            if parse_and_handle_meta(transcript):
                # Give a brief audible confirmation using the (possibly new) voice
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_wav:
                    tts_to_wav_file("Okay! Settings updated.", tts_wav.name)
                    play_wav(tts_wav.name)
                continue

            # If we had a pending continuation but the user asked something else, clear it
            if CONTEXT["active"] and not _is_affirmative(transcript):
                CONTEXT["active"] = False
                CONTEXT["topic"] = None
                CONTEXT["round"] = 0
                log(f"[ctx] set active={CONTEXT['active']} topic='{_preview(CONTEXT['topic']) if CONTEXT['topic'] else None}' round={CONTEXT['round']} hist={len(CONTEXT.get('history', []))}")

            print(f"{ASSISTANT_NAME}: thinking...")
            # 4) Get concise reply
            reply, used_brief = get_chat_reply(transcript)
            print("answered")
            log(f"[chat] final out: {_preview(reply)} | brief={used_brief}")

            # 5) TTS as WAV + immediate playback
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_wav:
                tts_to_wav_file(reply, tts_wav.name)
                play_wav(tts_wav.name)

            # Append this turn to rolling history so follow-ups have context
            _append_history("user", transcript)
            _append_history("assistant", reply)

            # Manage continuation state (do not clear history here)
            if used_brief:
                CONTEXT["active"] = True
                CONTEXT["topic"] = transcript
                CONTEXT["round"] = 0
            else:
                CONTEXT["active"] = False
                CONTEXT["topic"] = None
                CONTEXT["round"] = 0
            log(f"[ctx] set active={CONTEXT['active']} topic='{_preview(CONTEXT['topic']) if CONTEXT['topic'] else None}' round={CONTEXT['round']} hist={len(CONTEXT.get('history', []))}")

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        sd.stop()

if __name__ == "__main__":
    main()