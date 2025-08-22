import os, time, random
from openai import OpenAI
from typing import Optional, List, Dict

class ChatLLM:
    """Minimal OpenAI chat wrapper with retry & max-length trim."""
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system = system_prompt
        self.model = model

    def _backoff(self, fun, tries=5, base=0.5, cap=6):
        for i in range(tries):
            try: return fun()
            except Exception:
                if i == tries-1: raise
                time.sleep(min(cap, base*(2**i) + random.random()*0.2))

    def ask(self, user_text: str, max_chars: int = 350, history: Optional[List[Dict]] = None) -> str:
        def call():
            msgs = [{"role":"system","content":self.system}]
            if history:
                msgs.extend(history)  # list of {"role": "...", "content": "..."}
            msgs.append({"role":"user","content":user_text})
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=0.4, max_tokens=512
            )
            return resp.choices[0].message.content.strip()
        txt = self._backoff(call)
        return txt[:max_chars]