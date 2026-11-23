from pathlib import Path

import numpy as np
import pytest

from frsbot.analysis.matches import aggregate, analyze, contact
from frsbot.core.specs import load_specs
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState
from frsbot.replay.convert import convert
from frsbot.replay.rbrp import parse


def state(t, where, point, robots=None):
    robots = robots or [RobotState(1, 1690, "blue", -6, 0, 0, 0, 0, 2, 51),
                        RobotState(2, 1690, "red", -5.2, 0, 0, 0, 0, 2, 51)]
    return GameState(t, Phase.AUTO, {a: HubState(True, 20 - t) for a in ("blue", "red")},
                     robots, FuelState(np.array([point], dtype=np.float32), np.array([where], dtype=np.uint8)), 0)


def test_delayed_score_requires_robot_flight_hub_chain():
    states = [state(0, 2, [-6, .7, 0]), state(1, 1, [-5.8, 1, 0]),
              state(2, 3, [-3.645, 1.8, 0]), state(3, 3, [-3.645, 1, 0]),
              state(4, 4, [-3.645, .1, 0])]
    result = analyze(states, events=[[4, {"t": "score", "fuel": 0, "alliance": "blue", "points": 1}]])
    assert result["robots"][0]["tracked_active_fuel"] == 1
    assert result["robots"][1]["tracked_active_fuel"] == 0
    assert sum(result["robots"][0]["role_seconds"].values()) == pytest.approx(4)
    assert result["robots"][0]["near_opponent_seconds"] == 4
    assert result["robots"][0]["contact_proxy_seconds"] == 4
    assert result["robots"][0]["defended_rate"]["fuel"] == 1
    assert result["robots"][0]["undefended_rate"]["fuel_per_second"] is None


def test_human_player_entry_and_recycled_ball_are_not_credited_to_robot():
    states = [state(0, 2, [-6, .7, 0]), state(1, 1, [-5.8, 1, 0]),
              state(2, 3, [-3.645, 1.8, 0]), state(3, 4, [-3.645, .1, 0]),
              state(4, 1, [-4, 1.5, 0]), state(5, 3, [-3.645, 1.8, 0])]
    events = [[2, {"t": "score", "fuel": 0, "alliance": "blue", "points": 1}],
              [5, {"t": "score", "fuel": 0, "alliance": "blue", "points": 1}]]
    result = analyze(states, events=events)
    assert result["robots"][0]["tracked_active_fuel"] == 1
    assert result["unattributed_active_fuel"]["blue"] == 1


def test_bouncing_flight_is_tracked_but_cannot_score_twice_without_relaunch():
    states = [state(0, 2, [-6, .7, 0]), state(1, 1, [-5.8, 1, 0]),
              state(2, 0, [-5, .08, 0]), state(3, 1, [-4, 1, 0]),
              state(4, 3, [-3.645, 1.8, 0]), state(5, 1, [-3.5, 1, 0]),
              state(6, 3, [-3.645, 1.8, 0])]
    events = [[4, {"t": "score", "fuel": 0, "alliance": "blue", "points": 1}],
              [6, {"t": "score", "fuel": 0, "alliance": "blue", "points": 1}]]
    result = analyze(states, events=events)
    assert result["robots"][0]["tracked_active_fuel"] == 1
    assert result["unattributed_active_fuel"]["blue"] == 1
    assert analyze(states)["robots"][0]["tracked_active_fuel"] == 1


def test_opponent_hub_score_is_not_own_alliance_credit():
    states = [state(0, 2, [-6, .7, 0]), state(1, 1, [-5.8, 1, 0]), state(2, 3, [3.645, 1.8, 0])]
    result = analyze(states, events=[[2, {"t": "score", "fuel": 0, "alliance": "red", "points": 1}]])
    assert result["robots"][0]["tracked_active_fuel"] == 0
    assert len(result["robots"][0]["opponent_hub_scores"]) == 1


def test_pass_can_land_before_ally_collects_but_interception_is_not_ally_transfer():
    def match(alliance):
        robots = [RobotState(1, 1690, "blue", 0, 0, 0, 0, 0, 3, 51),
                  RobotState(2, 1690, alliance, 2, 0, 0, 0, 0, 0, 51)]
        return analyze([state(0, 2, [0, .7, 0], robots), state(1, 1, [1, 1, 0], robots),
                        state(2, 0, [2, .1, 0], robots), state(3, 2, [2, .7, 0], robots)])
    friendly = match("blue")
    assert friendly["robots"][0]["completed_ally_transfers"] == 1
    assert friendly["robots"][1]["collected"] == 1
    assert friendly["robots"][0]["pass_launch_proxy"] == 1
    assert match("red")["transfers"] == []


def test_hub_recycled_fuel_is_not_an_ally_pass():
    robots = [RobotState(1, 1690, "blue", 0, 0, 0, 0, 0, 3, 51),
              RobotState(2, 1690, "blue", 2, 0, 0, 0, 0, 0, 51)]
    result = analyze([state(0, 2, [0, .7, 0], robots), state(1, 1, [1, 1, 0], robots),
                      state(2, 3, [-3.645, 1.8, 0], robots), state(3, 1, [1, .5, 0], robots),
                      state(4, 2, [2, .7, 0], robots)])
    assert result["transfers"] == []


def test_empty_and_invalid_match():
    assert analyze([])["robots"] == []
    assert aggregate([])["matches"] == 0
    s = state(0, 0, [0, 0, 0])
    with pytest.raises(ValueError, match="increasing"):
        analyze([s, s])
    s.robots[1].id = 1
    with pytest.raises(ValueError, match="roster"):
        analyze([s])


def test_contact_is_oriented_footprint_proxy():
    spec = next(s for s in load_specs() if s.team == 1690)
    a, b = state(0, 0, [0, 0, 0]).robots
    assert contact(a, b, spec, spec)
    b.x = 0
    assert not contact(a, b, spec, spec)


def test_actual_multiplayer_fixture():
    path = Path(__file__).parent / "fixtures" / "multiplayer_excerpt.rbrp.gz"
    rep = parse(path.read_bytes(), fuel=True)
    result = analyze(convert(rep), events=rep.events, metas=rep.metas)
    assert len(result["robots"]) == 4
    assert result["duration_seconds"] == pytest.approx(2 / 60)
    for robot in result["robots"]:
        assert sum(robot["role_seconds"].values()) == pytest.approx(2 / 60)


def test_complete_fj5s_when_cached():
    path = Path(__file__).resolve().parents[1] / "data/raw_mp/ranked-FJ5S-1795384599586.0.rbrp"
    if not path.exists():
        pytest.skip("complete FJ5S recording not cached")
    rep = parse(path.read_bytes(), fuel=True)
    result = analyze(convert(rep), events=rep.events, metas=rep.metas, partial=rep.meta["partial"])
    assert not result["partial"]
    assert result["winner"] == "red"
    assert [result["final_score"][a]["total"] for a in ("blue", "red")] == [1108, 1242]
    assert [r["tracked_active_fuel"] for r in result["robots"]] == [582, 505, 728, 468]
    for robot in result["robots"]:
        tally = robot["engine_tally"]
        assert robot["tracked_active_fuel"] == tally["autoFuel"] + tally["teleFuel"]
        assert sum(robot["role_seconds"].values()) == pytest.approx(163)
    assert len(result["robots"][0]["opponent_hub_scores"]) == 1
    assert result["fouls"][0]["rule"] == "G419"
    assert isinstance(result["fouls"][0]["time"], float)
