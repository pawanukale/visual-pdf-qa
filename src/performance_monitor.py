import os
import threading
import time
from contextlib import contextmanager
import psutil

_PROCESS = psutil.Process(os.getpid())

def memory_mb() -> float:
    return _PROCESS.memory_info().rss / (1024 * 1024)

class PeakMemorySampler:
    """Samples this process RSS while a measured stage is running."""
    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.peak_mb = memory_mb()
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        while not self._stop.is_set():
            try:
                self.peak_mb = max(self.peak_mb, memory_mb())
            except psutil.Error:
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self.peak_mb = max(self.peak_mb, memory_mb())


def timed_call(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start
