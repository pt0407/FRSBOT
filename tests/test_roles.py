import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from frsbot.core.field import is_active, phase_at, seconds_to_toggle
from frsbot.core.specs import load_specs
from frsbot.core.state import FuelState, GameState, HubState, RobotState
from frsbot.strategy.roles import Role, StrategyLayer, active_seconds, scoring_prior


@pytest.fixture
def specs():
    base = next(s for s in load_specs() if s.team == 254)
    return {254: replace(base, fire_rate=1., total_capacity=20, max_speed=2., intake_rate=2.),
            2910: replace(base, team=2910, fire_rate=32., total_capacity=100, max_speed=5.),
            1114: replace(base, team=1114, fire_rate=40., total_capacity=150, max_speed=5.)}


def observation(t=60., robots=None, pos=None, where=None):
    if robots is None:
        robots = [RobotState(10, 254, "blue", -6., 0., 0., 0., 0., 10, 20),
                  RobotState(11, 2910, "blue", -6., 2., 0., 0., 0., 40, 100),
                  RobotState(20, 1114, "red", 6., 0., 0., 0., 0., 60, 150)]
    return GameState(t, phase_at(t), {a: HubState(is_active(a, t, "blue"), seconds_to_toggle(a, t, "blue"))
                                     for a in ("blue", "red")}, robots,
                     FuelState(np.array(pos if pos is not None else [], np.float32).reshape(-1, 3),
                               np.array(where if where is not None else [], np.uint8)), 0)


def layer(specs):
    strategy = StrategyLayer(specs[254], specs)
    def prediction(tracker, i):
        robot = tracker.history[-1].robots[i]
        return {"probabilities": {"Collect": 0., "Score": float(robot.id != 10), "Feed": 0.,
                                  "Defend": 0., "Idle": float(robot.id == 10)}, "defend_target": None}
    strategy.model.predict = prediction
    return strategy


def settle(strategy, t=60., **kwargs):
    for dt in (0., 1., 2.):
        state = observation(t + dt, **kwargs)
        strategy.observe(state)
    return strategy.role(state)


def test_solo_no_opponents_and_one_vs_one_no_feeder(specs):
    strategy = layer(specs)
    robot = observation().robots[0]
    assert settle(strategy, robots=[robot]) == Role("Scorer")
    strategy = layer(specs)
    state = observation(60, robots=[robot, observation().robots[-1]])
    strategy.observe(state)
    assert Role("Feeder") not in strategy.role_values(state)


def test_weak_scorer_defends_best_unguarded_opponent(specs):
    strategy = layer(specs)
    assert settle(strategy) == Role("Defender", 20)
    state = strategy._previous
    strategy.intents[11] = {"probabilities": {"Defend": .9}, "defend_target": 20}
    assert strategy.role_values(state)[Role("Defender", 20)] < 0


def test_big_late_deficit_always_scores_even_from_defender(specs):
    strategy = layer(specs)
    assert settle(strategy) == Role("Defender", 20)
    strategy.score_estimate["red"] = 10000
    state = observation(145)
    strategy.observe(state)
    assert strategy.role(state) == Role("Scorer")
    assert list(strategy.role_values(state)) == [Role("Scorer")]


def test_big_lead_late_prefers_denial(specs):
    strategy = layer(specs)
    strategy.score_estimate["blue"] = 10000
    assert settle(strategy, t=140).kind in {"Defender", "Contester"}


@pytest.mark.parametrize("t", [0., 15., 160., 163.])
def test_no_defender_in_auto_or_final_three_seconds(specs, t):
    strategy = layer(specs)
    state = observation(t)
    strategy.observe(state)
    assert all(role.kind != "Defender" for role in strategy.role_values(state))


def test_hysteresis_requires_continuous_winner_and_ten_percent(specs):
    strategy = layer(specs)
    value = [30.]
    strategy.role_values = lambda state: {Role("Scorer"): 0., Role("Defender", 20): value[0]}
    for t in range(60, 66):
        value[0] = 30. if t % 2 == 0 else -1.
        state = observation(t)
        strategy.observe(state)
        assert strategy.role(state) == Role("Scorer")
    for t in (66., 67., 68.):
        value[0] = 30.
        state = observation(t)
        strategy.observe(state)
    assert strategy.role(state) == Role("Defender", 20)
    strategy = layer(specs)
    strategy.role_values = lambda state: {Role("Scorer"): 0., Role("Defender", 20): .01}
    assert settle(strategy) == Role("Scorer")


def test_defended_by_duration_zone_and_evasion(specs):
    strategy = layer(specs)
    for k in range(31):
        state = observation(140 + k / 10)
        state.robots = [state.robots[0], replace(state.robots[-1], x=-5.)]
        strategy.observe(state)
        if k < 15:
            assert strategy.defended_by(state) is None
    assert strategy.defended_by(state) == 20
    assert strategy.evasion(state) == "stay"
    strategy.suppression_ratio = 1.
    assert strategy.evasion(state) == "stay"
    strategy.suppression_ratio = 0.
    assert strategy.evasion(state) == "alternate_slot"
    state.robots.append(replace(observation().robots[1], x=-7.))
    strategy.intents[11] = {"probabilities": {"Score": .9}}
    strategy.rates[11] = 100.
    assert strategy.evasion(state) == "feed"
    strategy = layer(specs)
    for k in range(31):
        state = observation(k / 10)
        state.robots[0].x = 1.
        state.robots[-1].x = 2.
        strategy.observe(state)
    assert strategy.defended_by(state) is None


def test_teammate_cluster_targets_empty_ray_and_slots(specs):
    strategy = layer(specs)
    state = observation(pos=[[-4., 0.1, 2.], [-4.1, .1, 2.1], [-7., .1, 2.], [-5., 1., 2.]], where=[0, 0, 0, 0])
    state.robots[1].vx = 1.
    strategy.observe(state)
    strategy.intents[11] = {"probabilities": {"Collect": .9}}
    targets = strategy.teammate_targets(state)
    assert targets and all(-4.2 < x < -3.9 and 1.9 < z < 2.2 for x, z in targets)
    assert (-6., 2.) in strategy.teammate_slots(state)
    state.robots[1].vx = 0.
    assert strategy.teammate_targets(state) == []
    state.fuel = FuelState(np.empty((0, 3), np.float32), np.empty(0, np.uint8))
    assert strategy.teammate_targets(state) == []


def test_score_entries_rate_pseudocount_and_trend(specs):
    strategy = layer(specs)
    first = observation(0, pos=[[-3.645, 1.8, 0], [3.645, 1.8, 0]], where=[1, 3])
    strategy.observe(first)
    assert strategy.score_estimate == {"blue": 0, "red": 0}
    following = observation(1, pos=[[-3.645, 1.8, 0], [3.645, 1.8, 0]], where=[3, 3])
    following.robots[0].hopper = 8
    strategy.observe(following)
    assert strategy.score_estimate == {"blue": 1, "red": 0}
    assert strategy.score_difference(following) == 1 and strategy.score_trend(following) == 1
    assert strategy.rates[10] == pytest.approx((20 * scoring_prior(specs[254]) + 2) / 21)
    following = observation(2, pos=[[-3.645, 1.8, 0], [3.645, 1.8, 0]], where=[3, 3])
    strategy.observe(following)
    assert strategy.score_estimate["blue"] == 1
    assert active_seconds("blue", 60, 30, "blue") == 23
    assert active_seconds("red", 60, 30, "blue") == 7
    assert active_seconds("blue", 60, 30) == 15
    assert active_seconds("red", 160, 30) == 3
    assert active_seconds("blue", -3, 3) == 0
    assert scoring_prior(replace(specs[254], fire_rate=0.)) == 0
    strategy = layer(specs)
    strategy.observe(observation(39, pos=[[-3.645, 1.8, 0], [10., 0., 0]], where=[1, 1]))
    strategy.observe(observation(40, pos=[[-3.645, 1.8, 0], [10., 0., 0]], where=[3, 3]))
    assert strategy.score_estimate == {"blue": 0, "red": 0}


def test_feeding_and_contesting_candidates(specs):
    strategy = layer(specs)
    state = observation(40)
    strategy.observe(state)
    assert Role("Feeder") in strategy.role_values(state)
    state.robots[-1].x = 0.
    strategy.intents[20] = {"probabilities": {"Collect": 1.}}
    assert strategy.role_values(state)[Role("Contester")] > 0
    state = observation(70)
    strategy = layer(specs)
    strategy.score_estimate["red"] = 1
    strategy.observe(state)
    assert list(strategy.role_values(state)) == [Role("Scorer")]


def test_errors_fallback_and_runtime_import_boundary(specs, monkeypatch, tmp_path):
    import frsbot.strategy.roles as roles
    monkeypatch.setattr(roles, "STATS_PATH", tmp_path / "absent.json")
    strategy = layer(specs)
    assert strategy.suppression_ratio == .65
    (tmp_path / "absent.json").write_text('{"suppression_ratio": "invalid"}')
    assert layer(specs).suppression_ratio == .65
    with pytest.raises(ValueError):
        Role("Unknown")
    with pytest.raises(RuntimeError):
        strategy.role(observation())
    for relative in ("strategy/roles.py", "strategy/intent.py"):
        source = Path(roles.__file__).parents[1] / relative
        tree = ast.parse(source.read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(name.startswith(("frsbot.analysis", "frsbot.sim", "frsbot.replay", "sklearn")) for name in imports)
