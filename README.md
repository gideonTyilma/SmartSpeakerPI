# Pi Smart Speaker (MVP)

A DIY Raspberry Pi voice assistant using OpenAI GPT models.  
MVP goal: wake word → voice in → GPT → voice out.

## Hardware
- Raspberry Pi 4 Model B (4GB)
- Power supply with on/off switch
- MicroSD card (64GB+)
- USB/Bluetooth speaker-mic (AIRHUG)
- Optional LED strip + case mod

## Software
- Raspberry Pi OS
- Python 3.9+
- OpenAI Python SDK
- `python-dotenv`
- Audio I/O library (`pyaudio` or `sounddevice`)
- Wake word engine (Porcupine)

## Flow
1. Detect wake word.
2. Record speech.
3. Transcribe (Whisper API).
4. Get GPT reply.
5. Convert to speech & play.

## Development
- Start coding on Mac with venv.
- Use `.env` for `OPENAI_API_KEY`.
- Test LLM calls before Pi hardware setup.
