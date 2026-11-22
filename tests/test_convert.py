import math
from pathlib import Path

import numpy as np
import pytest

from frsbot.core.specs import load_specs
from frsbot.core.state import Phase
from frsbot.replay.convert import auto_start_seconds, convert
from frsbot.replay.rbrp import Replay, parse


def test_cached_replay_conversion():
    paths = sorted((Path(__file__).resolve().parents[1] / "data" / "raw").glob("*.rbrp"))
    if not paths:
        pytest.skip("requires cached data/raw/*.rbrp replays")
    rep = parse(paths[0].read_bytes(), fuel=True)
    states = convert(rep)
    assert len(states) == len(rep.frames)
    assert states[0].t == -auto_start_seconds(rep.metas)
    assert states[0].me == 0
    for i in (0, 1, len(states) - 1):
        state = states[i]
        raw = rep.frames[i][1]
        robot = state.robots[0]
        assert (robot.id, robot.team, robot.alliance) == tuple(rep.robots[0][k] for k in ("id", "team", "alliance"))
        assert robot.x == raw[0] * .0005
        assert robot.z == raw[2] * .0005
        assert robot.vx == raw[7] * .001
        assert robot.vz == raw[9] * .001
        assert robot.a == math.pi - 2 * math.atan2(raw[4], raw[6])
        assert robot.hopper == raw[13]
        assert robot.capacity == next(s.hopper_capacity for s in load_specs() if s.team == robot.team)
        assert state.fuel.pos.shape == (504, 3)
        assert state.fuel.pos.dtype == np.float32
        assert state.fuel.where.dtype == np.uint8
        np.testing.assert_allclose(state.fuel.pos, rep.fuel[i][0] * .0005, atol=1e-6)
        assert not np.shares_memory(state.fuel.where, rep.fuel[i][1])


def test_countdown_and_multiple_robots():
    raw = [0] * 30
    raw[6] = raw[21] = 32767
    rep = Replay({}, {}, [{"id": 1, "team": 254, "alliance": "blue"},
                          {"id": 2, "team": 971, "alliance": "red"}],
                 [{"at": 0}, {"at": 15}], 30,
                 frames=[(0, raw), (180, raw), (2160, raw)],
                 fuel=[(np.zeros((0, 3), np.int16), np.zeros(0, np.uint8)) for _ in range(3)])
    rep.metas = [[0, {"phase": "countdown"}], [90, {"phase": "auto"}],
                 [1080, {"phase": "shift1", "firstInactive": "red"}]]
    states = convert(rep)
    assert [s.t for s in states] == [-3, 0, 33]
    assert [s.phase for s in states] == [Phase.COUNTDOWN, Phase.AUTO, Phase.SHIFT1]
    assert states[-1].hubs["blue"].active
    assert not states[-1].hubs["red"].active
    assert len(states[0].robots) == 2
    assert states[0].fuel.pos.shape == (0, 3)


def test_empty_and_missing_fuel_or_metadata():
    rep = Replay({}, {}, [], [], 0)
    assert convert(rep) == []
    rep.frames = [(0, [])]
    with pytest.raises(ValueError, match="fuel=True"):
        convert(rep)
    rep.fuel = [(np.zeros((0, 3), np.int16), np.zeros(0, np.uint8))]
    with pytest.raises(ValueError, match="auto start"):
        convert(rep)
