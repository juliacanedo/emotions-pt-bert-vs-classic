import time
from dataclasses import dataclass
from typing import Optional

try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

@dataclass
class TimeVRAM:
    seconds: float
    peak_mb: Optional[float]

class TimerVRAM:
    def __init__(self, track_vram: bool = False):
        self.track_vram = track_vram and _NVML_OK
        self._start = None
        self._peak = 0.0
        if self.track_vram:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def __enter__(self):
        self._start = time.time()
        return self

    def _poll(self):
        if not self.track_vram:
            return
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        mb = mem.used / (1024**2)
        if mb > self._peak:
            self._peak = mb

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.track_vram:
            pynvml.nvmlShutdown()

    def snapshot(self) -> TimeVRAM:
        self._poll()
        secs = time.time() - (self._start or time.time())
        return TimeVRAM(seconds=secs, peak_mb=self._peak if self.track_vram else None)
