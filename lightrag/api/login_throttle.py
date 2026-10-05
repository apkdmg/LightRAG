"""
Throttling for password login attempts.

Failed attempts are counted per username. Once a username reaches the limit
within the window, further attempts are refused (HTTP 429) until the oldest
failure ages out. Counting is per username rather than per client IP because
behind the reverse proxy every request arrives from the proxy's address.

State is in-process: with several Gunicorn workers each worker counts
separately, so the effective limit is the configured one times the worker
count. That still turns an online password guess from unbounded into a
handful per window.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict


class LoginThrottle:
    """Per-username failed-login counter with a sliding window."""

    # Upper bound on tracked usernames so random usernames cannot exhaust memory.
    MAX_TRACKED = 10_000

    def __init__(
        self,
        max_failures: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._clock = clock
        self._failures: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_failures > 0 and self.window_seconds > 0

    @staticmethod
    def _key(username: str) -> str:
        return (username or "").strip().lower()

    def _prune(self, key: str, now: float) -> Deque[float]:
        attempts = self._failures.get(key)
        if attempts is None:
            return deque()
        while attempts and now - attempts[0] >= self.window_seconds:
            attempts.popleft()
        if not attempts:
            del self._failures[key]
        return attempts

    def retry_after(self, username: str) -> int:
        """Seconds until another attempt is allowed; 0 when not locked out."""
        if not self.enabled:
            return 0
        key = self._key(username)
        with self._lock:
            now = self._clock()
            attempts = self._prune(key, now)
            if len(attempts) < self.max_failures:
                return 0
            return max(1, math.ceil(self.window_seconds - (now - attempts[0])))

    def record_failure(self, username: str) -> None:
        if not self.enabled:
            return
        key = self._key(username)
        with self._lock:
            now = self._clock()
            self._prune(key, now)
            if key not in self._failures and len(self._failures) >= self.MAX_TRACKED:
                # Drop the entry whose newest failure is oldest.
                oldest = min(self._failures, key=lambda k: self._failures[k][-1])
                del self._failures[oldest]
            self._failures.setdefault(key, deque()).append(now)

    def reset(self, username: str) -> None:
        with self._lock:
            self._failures.pop(self._key(username), None)
