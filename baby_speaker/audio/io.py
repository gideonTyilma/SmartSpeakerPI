import sounddevice as sd
import queue
from typing import Optional
import numpy as np

class AudioIO:
    """
    Minimal, platform-agnostic audio I/O using sounddevice/PortAudio.
    16-bit mono PCM at a fixed sample rate.
    """
    def __init__(self, input_device: Optional[str], output_device: Optional[str],
                 sample_rate=16000, block_ms=20):
        self.sample_rate = sample_rate
        self.block = int(sample_rate * block_ms / 1000)
        self.q_in, self.q_out = queue.Queue(), queue.Queue()
        self.play_buf = np.zeros(0, dtype=np.int16)  # rolling output buffer
        self.play_pos = 0
        self.in_stream = sd.InputStream(
            device=input_device, channels=1, samplerate=sample_rate,
            dtype="int16", blocksize=self.block, callback=self._in_cb
        )
        self.out_stream = sd.OutputStream(
            device=output_device, channels=1, samplerate=sample_rate,
            dtype="int16", blocksize=self.block, callback=self._out_cb
        )

    def _in_cb(self, indata, frames, time, status):
        if status: print("Input status:", status)
        self.q_in.put(bytes(indata))

    def _out_cb(self, outdata, frames, time, status):
        if status:
            print("Output status:", status)

        frames_needed = frames

        # Top up play buffer from queued chunks
        while self.play_buf.size - self.play_pos < frames_needed:
            try:
                chunk = self.q_out.get_nowait()  # bytes or np.int16
            except queue.Empty:
                break
            if isinstance(chunk, (bytes, bytearray)):
                data = np.frombuffer(chunk, dtype=np.int16)
            elif isinstance(chunk, np.ndarray):
                data = chunk.astype(np.int16, copy=False)
            else:
                data = np.zeros(0, dtype=np.int16)

            if self.play_pos == 0 and self.play_buf.size == 0:
                self.play_buf = data
            else:
                if self.play_pos > 0:
                    # drop already-played part to keep buffer small
                    self.play_buf = self.play_buf[self.play_pos:]
                    self.play_pos = 0
                self.play_buf = np.concatenate([self.play_buf, data])

        # If still not enough, pad with zeros
        available = max(0, self.play_buf.size - self.play_pos)
        take = min(frames_needed, available)
        out = np.zeros(frames_needed, dtype=np.int16)
        if take > 0:
            out[:take] = self.play_buf[self.play_pos:self.play_pos+take]
            self.play_pos += take

        outdata[:] = out.reshape(-1, 1)
        

    def start(self):
        self.in_stream.start(); self.out_stream.start()
    def stop(self):
        self.in_stream.stop(); self.out_stream.stop()

    def read(self) -> bytes:
        return self.q_in.get()
    def play(self, pcm_bytes: bytes):
        self.q_out.put(pcm_bytes)
