"""Audible alert, rate-limited and non-blocking."""

from __future__ import annotations

import sys
import threading
import time


class Beeper:
    """Plays a short tone when attention is lost. Never blocks the video loop."""

    def __init__(self, enabled: bool = True, freq: int = 880, ms: int = 180,
                 min_gap: float = 1.5):
        self.enabled = enabled and sys.platform == "win32"
        self.freq, self.ms, self.min_gap = freq, ms, min_gap
        self._last = 0.0
        self._busy = threading.Lock()

    def _play(self) -> None:
        try:
            import winsound
            winsound.Beep(self.freq, self.ms)
        except (RuntimeError, ImportError):
            self.enabled = False
        finally:
            self._busy.release()

    def maybe_beep(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last < self.min_gap:
            return
        if not self._busy.acquire(blocking=False):
            return
        self._last = now
        threading.Thread(target=self._play, daemon=True).start()
