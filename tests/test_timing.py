import time

import pytest

from tlv.timing import EpochTimer


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_laps_are_cumulative_and_start_at_zero():
    clk = FakeClock()
    timer = EpochTimer(clock=clk)
    timer.start()
    clk.advance(2.0)
    assert timer.lap() == pytest.approx(2.0)
    clk.advance(3.5)
    assert timer.lap() == pytest.approx(5.5)
    assert timer.laps == [pytest.approx(2.0), pytest.approx(5.5)]


def test_paused_time_is_excluded():
    clk = FakeClock()
    timer = EpochTimer(clock=clk)
    timer.start()
    clk.advance(1.0)
    timer.pause()
    clk.advance(50.0)  # e.g. test-set evaluation, not training time
    timer.resume()
    clk.advance(1.0)
    assert timer.lap() == pytest.approx(2.0)


def test_lap_before_start_raises():
    with pytest.raises(RuntimeError):
        EpochTimer().lap()


def test_laps_never_decrease_with_real_clock():
    timer = EpochTimer()
    timer.start()
    for _ in range(5):
        time.sleep(0.005)
        timer.lap()
    assert all(b >= a for a, b in zip(timer.laps, timer.laps[1:]))
    assert timer.laps[0] > 0


def test_default_clock_is_monotonic():
    assert EpochTimer().clock is time.perf_counter
