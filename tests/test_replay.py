from pathlib import Path

import pytest

from frsbot.replay.rbrp import parse

ROOT = Path(__file__).resolve().parents[1]


def test_cached_replay_fuel_and_frame_rate():
    paths = sorted((ROOT / "data" / "raw").glob("*.rbrp"))
    if not paths:
        pytest.skip("requires a cached data/raw/*.rbrp replay")
    rep = parse(paths[0].read_bytes(), fuel=True)
    assert rep.start["fuelCount"] == 504
    assert len(rep.frames) == len(rep.fuel) > 30
    ticks = [tick for tick, _ in rep.frames]
    assert set(b - a for a, b in zip(ticks, ticks[1:])) == {2}
    assert (ticks[-1] - ticks[0]) / 60 == pytest.approx((len(ticks) - 1) / 30)
    for pos, where in rep.fuel:
        assert pos.shape == (504, 3)
        assert where.shape == (504,)
    assert rep.radius.shape == (504,)


def test_invalid_replay():
    with pytest.raises(ValueError, match="not an RBRP replay"):
        parse(b"invalid")
