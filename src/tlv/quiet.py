"""Guard against other programs disturbing wall-clock timings.

The experiments measure wall-clock time on a desktop that may also be used for other things.
Before each timed run the runner waits until system-wide CPU use is low, and during the run it
records how much CPU the *rest* of the system used (``bg_cpu_fraction``), so that disturbed runs
can be identified afterwards.
"""
from __future__ import annotations

import ctypes
import os
import time


def background_fraction(busy_s: float, own_s: float, total_s: float) -> float:
    """Share of all CPU time used by processes other than ours, clamped to [0, 1].

    ``busy_s``: CPU seconds used by the whole system (all cores); ``own_s``: CPU seconds used by
    this process; ``total_s``: CPU seconds available (elapsed time x logical cores).
    """
    if total_s <= 0:
        return 0.0
    return min(1.0, max(0.0, (busy_s - own_s) / total_s))


def system_cpu_seconds() -> tuple[float, float]:
    """(busy, total) CPU seconds summed over all cores since boot. (0, 0) if unavailable."""
    try:
        class FT(ctypes.Structure):
            _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

        idle, kernel, user = FT(), FT(), FT()
        ok = ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))  # type: ignore[attr-defined,unused-ignore]
        if not ok:
            return 0.0, 0.0

        def val(f: FT) -> float:
            return ((f.hi << 32) | f.lo) / 1e7

        total = val(kernel) + val(user)  # kernel time includes idle time
        return total - val(idle), total
    except (AttributeError, OSError):
        pass
    try:
        with open("/proc/stat") as f:
            parts = [float(x) for x in f.readline().split()[1:]]
        total = sum(parts) / 100.0
        return total - (parts[3] + parts[4]) / 100.0, total
    except OSError:
        return 0.0, 0.0


def system_busy_fraction(window: float = 3.0) -> float:
    b0, t0 = system_cpu_seconds()
    time.sleep(window)
    b1, t1 = system_cpu_seconds()
    return 0.0 if t1 <= t0 else min(1.0, max(0.0, (b1 - b0) / (t1 - t0)))


def wait_until_quiet(threshold: float = 0.10, max_wait: float = 1800.0, poll: float = 10.0) -> tuple[float, float]:
    """Block until system-wide CPU use is below ``threshold`` (or ``max_wait`` s have passed).

    Returns (seconds waited, last observed busy fraction).
    """
    start = time.monotonic()
    busy = system_busy_fraction()
    while busy > threshold and time.monotonic() - start < max_wait:
        time.sleep(poll)
        busy = system_busy_fraction()
    return time.monotonic() - start, busy


class BackgroundMeter:
    """Measures background CPU use over a ``with`` block."""

    def __enter__(self) -> BackgroundMeter:
        self._sys0 = system_cpu_seconds()
        self._own0 = time.process_time()
        self._wall0 = time.perf_counter()
        self.fraction = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        busy1, total1 = self._sys0[0], self._sys0[1]
        busy_now, total_now = system_cpu_seconds()
        busy, total = busy_now - busy1, total_now - total1
        own = time.process_time() - self._own0
        if total <= 0:  # no system counters: cannot tell
            self.fraction = float("nan")
        else:
            self.fraction = background_fraction(busy, own, total)


CPUS = os.cpu_count() or 1
