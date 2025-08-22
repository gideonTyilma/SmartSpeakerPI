from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os, yaml

@dataclass
class Settings:
    openai_api_key: str
    stt_backend: str = "openai"
    tts_backend: str = "openai"
    sample_rate: int = 16000
    vad_level: int = 2
    input_device: Optional[str] = None   # None = system default
    output_device: Optional[str] = None
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
