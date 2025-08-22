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
