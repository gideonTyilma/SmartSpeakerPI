from .config import load_settings
from .audio.io import AudioIO
from .audio.vad import VADGate
from .skills.baby_assistant import SYSTEM_PROMPT, postprocess
from .llm.openai_chat import ChatLLM
from .stt.openai_stt import OpenAI_STT
from .tts.openai_tts import OpenAI_TTS
from .util.logging import setup_logger
from .memory.memory import ConversationMemory
import yaml, os

def run_loop():
    log = setup_logger()
    cfg = load_settings()
    # load memory settings from YAML
    cfg_path = os.path.join(os.path.dirname(__file__), "runtime", "config.yaml")
    yml = yaml.safe_load(open(cfg_path)) or {}
    mem_cfg = (yml.get("memory") or {})
    memory = ConversationMemory(path=mem_cfg.get("file","baby_speaker/runtime/session_memory.json"),
                            max_turns=int(mem_cfg.get("max_turns",5)))  
    MIN_SPEECH_SEC = 0.5  # >= 0.1s required; use 0.3s for safety
    MIN_BYTES = int(cfg.sample_rate * 2 * MIN_SPEECH_SEC)  # int16 mono
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
                        if len(buf) < MIN_BYTES:
                        # too short for Whisper; drop this fragment
                            buf = b""
                            silence = 0
                            continue

                        text = stt.transcribe(wav_bytes=buf)
                        buf = b""
                        silence = 0
                        if not text:
                            continue
                        log.info(f"User: {text}")
                        # === LLM ===
                        # history-aware ask
                        reply = llm.ask(text, max_chars=cfg.max_chars, history=memory.history_as_messages())
                        memory.add("user", text)
                        memory.add("assistant", reply)
                        reply = postprocess(text, reply)
                        log.info(f"Assistant: {reply}")
                        # === TTS ===
                        wav = tts.synth(reply)
                        audio.play(wav)
    finally:
        audio.stop()

def cli():
    run_loop()
