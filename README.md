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

## Current Progress

The project now uses a push-to-talk MVP instead of wake word detection. The text-to-speech (TTS) output is generated in WAV format, chosen over Opus for better compatibility across platforms. Meta-commands have been implemented to allow switching voices and personalities mid-chat. The system is currently working on Mac with the AIRHUG microphone and speaker setup.

## Running the Assistant

1. Clone the repository:
   ```
   git clone <repo-url>
   cd smartSpeakerCore
   ```
2. Install Python dependencies:
   ```
   pip install openai python-dotenv sounddevice simpleaudio pydub keyboard ffmpeg-python
   ```
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the assistant script:
   ```
   python muckabout.py
   ```
5. Hold the SPACE key to talk (push-to-talk).

### Switching Voices and Personalities

The assistant supports multiple voices and personalities. At startup, you can select a voice interactively. During a chat session, you can switch voices or personalities using meta-commands either typed or spoken, such as:

- `/voice alloy`
- `switch voice`
- `/persona cheerful`

These commands allow dynamic customization of the assistant’s tone and style.

### Dependencies

**System dependencies:**
- Python 3.9 or higher
- ffmpeg installed and accessible in your system PATH
- On Mac, Accessibility permissions must be granted for the `keyboard` module to detect key presses

**Python packages (install via pip):**
- openai
- python-dotenv
- sounddevice
- simpleaudio
- pydub
- keyboard
- ffmpeg-python
