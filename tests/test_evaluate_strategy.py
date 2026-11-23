from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from frsbot.bot.planner import Goal
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState
from frsbot.strategy.bot import StrategicImAI
from frsbot.strategy.roles import Role
from frsbot.tools import evaluate_strategy as evaluation


def pair(seed, pure, strategic):
    def variant(margin, foul, role):
        return dict(margin=margin, foul_points_conceded=foul,
                    role_fractions={r: float(r == role) for r in evaluation.ROLES})
    return dict(mode="2v2", seed=seed, pure=variant(pure, 15, "Scorer"),
                strategic=variant(strategic, 0, "Defender"))


def test_composition_reproducibility_balanced_sides_and_default_lanes():
    for mode, size in evaluation.MODES.items():
        a, ego = evaluation.composition(mode, 1)
        b, again = evaluation.composition(mode, 1)
        assert a == b and ego == again == 0
        assert len(a) == 2 * size
        assert set(s["team"] for s in a) <= set(evaluation.TEAMS)
        assert len(set(s["start"]["fy"] for s in a[:size])) == size
        c, red = evaluation.composition(mode, 2)
        assert red == size and c[red]["alliance"] == "red"
        assert all(s["start"]["headingDeg"] == 180 for s in c[size:])


def test_paired_statistics_empty_singleton_and_draw_definition():
    assert evaluation.summarize([]) == []
    first = pair(1, 0, 10)
    second = pair(2, -10, 20)
    single = evaluation.summarize([first])[0]
    assert single["paired_se"] is None
    assert "n/a" in evaluation.markdown([single])
    summary = evaluation.summarize([first, second])[0]
    assert summary["paired_diff"] == 20
    assert summary["paired_se"] == pytest.approx(10)
    assert summary["win_rate_pure"] == 0 and summary["win_rate_strategic"] == 1
    assert summary["foul_points_pure"] == 15 and summary["foul_points_strategic"] == 0
    assert summary["role_fractions"]["Defender"] == 1
    assert "+20.0 ± 10.0" in evaluation.markdown([summary])


def test_effective_role_counts_scorer_fallback_and_feeder_collection():
    spec = next(s for s in load_specs() if s.team == 254)
    bot = StrategicImAI(spec)
    bot.layer.role = lambda state: Role("Feeder")
    bot.goal = Goal("Contest", (0., 0.))
    assert evaluation.effective_role(bot, None) == "Feeder"
    bot.goal = Goal("Score", (-6., 0.))
    assert evaluation.effective_role(bot, None) == "Scorer"
    bot.goal = Goal("Feed", (-2., 2.))
    assert evaluation.effective_role(bot, None) == "Feeder"


class FakeEnv:
    def __init__(self):
        self.resets = []
        self.count = 0

    def reset(self, robots, ego, **kwargs):
        self.resets.append((robots, ego, kwargs))
        self.slots, self.ego, self.count = robots, ego, 0
        entities = [RobotState(10+i, slot.team, slot.alliance, -6. if slot.alliance == "blue" else 6.,
                               2. if i % 2 == 0 else -2., 0., 0., 0., 5, 72)
                    for i, slot in enumerate(robots)]
        self.state = GameState(0., Phase.AUTO, {a: HubState(True, 20.) for a in ("blue", "red")}, entities,
                               FuelState(np.empty((0, 3), np.float32), np.empty(0, np.uint8)), ego)
        return self.state

    def step(self, actions, ticks):
        assert set(actions) == {self.ego} and ticks == 3
        for i, slot in enumerate(self.slots):
            if i != self.ego:
                assert isinstance(slot.policy.act(self.state_for(i), {"scored": 0}), Inputs)
        self.count += 1
        self.state = replace(self.state, t=self.count * .05)
        # A teammate (not ego) commits this foul; its points are credited to red.
        event = dict(t="foul", by=self.state.robots[1].id, alliance="red", points=15, rule="G418")
        return SimpleNamespace(t=self.state.t, views={i: self.state_for(i) for i in range(len(self.slots))},
                               scores={"blue": 2*self.count, "red": self.count}, done=False, events=[event],
                               observation=SimpleNamespace(driver=[{"scored": 3*self.count}] * len(self.slots)))

    def state_for(self, i):
        return replace(self.state, me=i)


def test_play_api_metrics_alliance_fouls_and_paired_setup():
    specs = {s.team: s for s in load_specs()}
    lineup, ego = evaluation.composition("2v2", 1)
    env = FakeEnv()
    pure = evaluation.play(env, lineup, ego, 1, False, specs, opponent_defender=True, duration=.1)
    strategic = evaluation.play(env, lineup, ego, 1, True, specs, opponent_defender=True, duration=.1)
    assert pure["foul_points_conceded"] == strategic["foul_points_conceded"] == 30
    assert pure["our_fouls"] == strategic["our_fouls"] == []
    assert pure["robot_fuel"] == strategic["robot_fuel"] == 6
    assert pure["margin"] == strategic["margin"] == 2
    assert sum(strategic["role_fractions"].values()) == pytest.approx(1)
    assert len(strategic["timeline"]) == 1
    assert env.resets[0][2] == env.resets[1][2] == dict(seed=1, auto="none", event="event", hp=True, unload=True)
    assert [(s.team, s.alliance, s.start) for s in env.resets[0][0]] == [(s.team, s.alliance, s.start) for s in env.resets[1][0]]
    opponent = next(s for s in env.slots if s.alliance == "red")
    assert opponent.policy.bot.layer.role(None).target == env.state.robots[ego].id
    with pytest.raises(ValueError, match="duration"):
        evaluation.play(env, lineup, ego, 1, False, specs, duration=0)


def test_cli_rejects_empty_seed_set(monkeypatch):
    monkeypatch.setattr("sys.argv", ["evaluate_strategy", "--seeds", "0"])
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2
