import math

import pytest

from frsbot.live.latency import Latency, distribution, latency_arg


def test_latency_distribution_empty_and_units():
    assert distribution([])["median_ms"] is None
    result = distribution([50, 100, 150])
    assert result == dict(n=3, median_ms=100, p90_ms=140, median_ticks=6, p90_ticks=8.4)
    assert latency_arg("auto") == "auto" and latency_arg("0") == 0
    for value in ("nan", "inf", "-1", "1001", "invalid"):
        with pytest.raises(ValueError):
            latency_arg(value)


def test_running_latency_deduplication_and_step_estimate():
    timing = Latency()
    assert timing.estimate == 0
    packet = dict(latencies=[dict(id=1, at=10, pollAt=30, inputAt=31, frameAt=60)])
    timing.ingest(packet)
    timing.ingest(packet)
    assert len(timing.traces) == 1 and timing.since == 1
    assert timing.estimate == .05
    timing.steps = [dict(ms=100, tick_delta=4), dict(ms=None, error="timeout")]
    result = timing.summary()
    assert result["step_response"]["n"] == 1
    assert result["step_response"]["effect_tick_median"] == 4
    assert result["transport"]["write_to_poll"]["median_ms"] == 20
    assert result["transport"]["poll_to_input"]["median_ms"] == 1
    assert result["transport"]["input_to_frame"]["median_ms"] == 29
    assert math.isclose(timing.estimate, .1)
