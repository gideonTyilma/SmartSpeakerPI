import webrtcvad

class VADGate:
    """Simple VAD gate with fixed frame size (matches AudioIO block)."""
    def __init__(self, level=2, sample_rate=16000, frame_ms=20):
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # int16 mono
        self.vad = webrtcvad.Vad(level)

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != self.frame_bytes:
            return False
        return self.vad.is_speech(frame, self.sample_rate)
