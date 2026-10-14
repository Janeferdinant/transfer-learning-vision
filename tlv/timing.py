"""Wall-clock timing with a monotonic clock (no TensorFlow dependency)."""
from __future__ import annotations

import time
from typing import Callable, List, Optional


class EpochTimer:
    """Cumulative training time, one lap per epoch.

    Uses ``time.perf_counter`` (monotonic) by default; the clock is injectable so
    the logic can be unit-tested. ``pause``/``resume`` exclude work that is not
    training (for example test-set evaluation or logging).
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self.laps: List[float] = []
        self._started = False
        self._accum = 0.0
        self._t0: Optional[float] = None

    def start(self) -> None:
        self.laps = []
        self._accum = 0.0
        self._t0 = self.clock()
        self._started = True

    def pause(self) -> None:
        if self._t0 is not None:
            self._accum += self.clock() - self._t0
            self._t0 = None

    def resume(self) -> None:
        if self._started and self._t0 is None:
            self._t0 = self.clock()

    def elapsed(self) -> float:
        if not self._started:
            raise RuntimeError("timer not started")
        running = (self.clock() - self._t0) if self._t0 is not None else 0.0
        return self._accum + running

    def lap(self) -> float:
        """Record and return cumulative elapsed training time."""
        t = self.elapsed()
        self.laps.append(t)
        return t
