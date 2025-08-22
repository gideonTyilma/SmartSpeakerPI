# SmartSpeakerPI (Baby Care Assistant)

A voice-first assistant for new parents that runs on macOS or Raspberry Pi.  
It listens, transcribes speech, chats with an LLM, and speaks back using TTS.

- **Current focus:** Platform-agnostic refactor (works on macOS with USB/Bluetooth speakers).  
- **Pi migration:** handled separately (PipeWire, echo cancellation, systemd).

---

## Features

- **Hands-free VAD** (voice-activated): starts on speech, ends on short silence.
- **STT (Whisper)** via OpenAI.
- **LLM chat** with a safety-focused baby-care persona.
- **TTS** via OpenAI, auto-converted to 16 kHz mono PCM for playback.
- **USB/Bluetooth audio** (macOS default or pinned device).
- **Lightweight memory** of recent turns (configurable).

---

## Quickstart

```bash
# 1) checkout
git checkout -b refactor-no-pi

# 2) Python env
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 3) Install
pip install -e .

# 4) Secrets
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 5) (macOS) Ensure ffmpeg is installed
brew install ffmpeg

# 6) Run
baby-speaker

If you’re on Python 3.9, this repository is ready to use out of the box — just follow the setup steps below to get started. For newer syntax, 3.11+ is ideal.

⸻

Configuration

Edit baby_speaker/runtime/config.yaml:

stt_backend: openai
tts_backend: openai
sample_rate: 16000
vad_level: 3
# input_device: null   # use system default on macOS
# output_device: null  # use system default on macOS
max_chars: 350
block_ms: 20

memory:
  max_turns: 5
  file: "baby_speaker/runtime/session_memory.json"

Pick a specific audio device

List devices:

python -m sounddevice

Then set:

output_device: "AirHug Conference Speaker"   # or numeric index
input_device:  "AirHug Conference Speaker"   # if using its mic


⸻

How it works (high-level)

Mic → VAD (webrtcvad) → buffer (≥0.3s) → Whisper STT → LLM (system prompt + memory)
 → Text reply → OpenAI TTS (stream) → convert to 16k mono PCM → audio out

	•	Audio I/O: baby_speaker/audio/io.py (PortAudio via sounddevice).
	•	VAD: baby_speaker/audio/vad.py.
	•	STT: baby_speaker/stt/openai_stt.py.
	•	LLM: baby_speaker/llm/openai_chat.py.
	•	TTS: baby_speaker/tts/openai_tts.py.
	•	Persona & safety: baby_speaker/runtime/prompts/system.txt, red_flags.txt.
	•	Memory: baby_speaker/memory/memory.py (rolling N turns to JSON).

⸻

Prompts & voices
	•	Persona: baby_speaker/runtime/prompts/system.txt
Tweak tone/length/safety language here.
	•	TTS voice: in baby_speaker/tts/openai_tts.py (default "alloy").
To make it configurable, add to config.yaml:

tts_voice: "alloy"

and read it in OpenAI_TTS init.

⸻

Common issues
	•	No audio output:
	•	Ensure device is default output on macOS or set output_device explicitly.
	•	USB devices often prefer 44.1 kHz; we up/downsample to 16 kHz internally and stream properly now.
	•	“Audio file is too short” (Whisper):
We gate STT to require ≥0.3 s of audio.
	•	TTS decode errors (ffmpeg 183 / invalid RIFF):
We sniff header (MP3/WAV) and decode correctly. Ensure ffmpeg is installed.
	•	Python 3.9 | None type errors:
We use 3.9-safe typing.Optional now. Upgrading to 3.11 is recommended.

⸻

Roadmap (short)
	•	Wake word (KWS) option (Porcupine/openwakeword).
	•	Barge-in (auto pause TTS when user speaks).
	•	Configurable personas/voices via YAML.
	•	Pi AEC via PipeWire + systemd service.

⸻

Development

# lint/test (suggested):
pip install ruff pytest
ruff check baby_speaker
pytest -q


⸻

Security & privacy
	•	Short, rolling memory stored locally in session_memory.json.
	•	Do not commit .env or memory files (covered by .gitignore).
	•	For medical emergencies, the assistant does not diagnose; it escalates to call a pediatrician/emergency services.

⸻

License

TBD — add your preferred license.

⸻

Credits
	•	OpenAI Whisper (STT) & TTS APIs
	•	sounddevice, webrtcvad, pydub/ffmpeg


## Environment Example

Below is a sample `.env.example` file to get started:

```env
# OpenAI API Key
OPENAI_API_KEY=sk-...

# Optional: configure default voice
TTS_VOICE=alloy

# Optional: log level (INFO, DEBUG, WARNING)
LOG_LEVEL=INFO
```