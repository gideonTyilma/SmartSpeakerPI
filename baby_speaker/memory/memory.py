from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, time

@dataclass
class Turn:
    role: str   # "user" | "assistant"
    content: str
    ts: float

class ConversationMemory:
    def __init__(self, path: str, max_turns: int = 5):
        self.path = Path(path)
        self.max_turns = max_turns
        self.turns: list[Turn] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self.turns = [Turn(**t) for t in raw]
            except Exception:
                self.turns = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(t) for t in self.turns][-2*self.max_turns:], ensure_ascii=False))

    def add(self, role: str, content: str):
        self.turns.append(Turn(role=role, content=content, ts=time.time()))
        # keep last N user+assistant pairs
        self.turns = self.turns[-2*self.max_turns:]
        self._save()

    def history_as_messages(self) -> list[dict]:
        # Return OpenAI-style chat messages (excluding the system message)
        return [{"role": t.role, "content": t.content} for t in self.turns]