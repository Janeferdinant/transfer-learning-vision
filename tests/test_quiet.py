import math

from tlv.quiet import BackgroundMeter, background_fraction, system_cpu_seconds


def test_background_fraction_basic_and_clamped():
    assert background_fraction(10.0, 10.0, 100.0) == 0.0
    assert math.isclose(background_fraction(30.0, 10.0, 100.0), 0.2)
    assert background_fraction(5.0, 10.0, 100.0) == 0.0  # own use cannot exceed busy time
    assert background_fraction(500.0, 0.0, 100.0) == 1.0
    assert background_fraction(1.0, 0.0, 0.0) == 0.0


def test_system_counters_are_monotonic_when_available():
    b0, t0 = system_cpu_seconds()
    sum(i * i for i in range(200000))
    b1, t1 = system_cpu_seconds()
    assert t1 >= t0 and b1 >= b0 - 1e-6


def test_meter_returns_fraction_in_range():
    with BackgroundMeter() as m:
        sum(i * i for i in range(200000))
    assert math.isnan(m.fraction) or 0.0 <= m.fraction <= 1.0
