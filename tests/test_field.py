import math

import pytest

from frsbot.core.field import is_active, phase_at, seconds_to_toggle
from frsbot.core.state import Phase
from frsbot.tools.validate_schedule import RAW, read_head, validate_metas


@pytest.mark.parametrize("t,phase,blue,red", [
    (-3, Phase.COUNTDOWN, False, False), (0, Phase.AUTO, True, True),
    (19.999, Phase.AUTO, True, True), (20, Phase.GAP, False, False),
    (23, Phase.TRANSITION, True, True), (33, Phase.SHIFT1, False, True),
    (58, Phase.SHIFT2, True, False), (83, Phase.SHIFT3, False, True),
    (108, Phase.SHIFT4, True, False), (133, Phase.ENDGAME, True, True),
    (163, Phase.SETTLE, False, False), (166, Phase.DONE, False, False),
])
def test_timeline(t, phase, blue, red):
    assert phase_at(t) == phase
    assert is_active("blue", t, "blue") == blue
    assert is_active("red", t, "blue") == red
    assert is_active("blue", t, "red") == red


@pytest.mark.parametrize("alliance,t,expected", [
    ("blue", -3, 3), ("blue", 0, 20), ("blue", 20, 3),
    ("blue", 23, 10), ("red", 23, 35), ("blue", 33, 25),
    ("blue", 108, 55), ("red", 108, 25), ("blue", 133, 30),
    ("blue", 163, math.inf), ("red", 166, math.inf),
])
def test_seconds_until_actual_toggle(alliance, t, expected):
    assert seconds_to_toggle(alliance, t, "blue") == expected


def test_invalid_schedule_inputs():
    for args in (("green", 0, "blue"), ("blue", 0, "green"), ("blue", 33, None),
                 ("blue", float("nan"), "blue")):
        with pytest.raises(ValueError):
            is_active(*args)


def test_all_cached_metadata():
    paths = sorted(RAW.glob("*.rbrp"))
    if not paths:
        pytest.skip("requires cached data/raw/*.rbrp replays")
    for path in paths:
        assert validate_metas(read_head(path)["metas"]) == [], path.name


def test_validation_detects_disagreement():
    metas = [[0, {"phase": "auto", "hubs": {"blue": {"active": False}, "red": {"active": True}}}]]
    assert validate_metas(metas) == [{"frame": 0, "t": 0, "field": "blue", "observed": False}]
