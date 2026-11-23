import math

import pytest

from frsbot.bot.imai import ImAI
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.sim.multi_env import MultiWasmEnv, RobotSlot, StepResult, default_starts


def slots():
    return [RobotSlot(team=team, alliance=alliance, start=dict(fx=x, fy=y, headingDeg=0))
            for team, alliance, x, y in [(254, "blue", 1, 2), (2910, "red", 15, 2),
                                        (1114, "blue", 1, 6), (4414, "red", 15, 6)]]


def test_multi_validation_before_reset():
    with MultiWasmEnv() as env:
        for robots in ([], None, [RobotSlot(254)] * 7):
            with pytest.raises(ValueError, match="slots"):
                env.reset(robots)
        with pytest.raises(RuntimeError, match="reset"):
            env.step({})


@pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").is_file(), reason="engine assets absent")
def test_two_vs_two_imai():
    specs = {s.team: s for s in load_specs()}
    with MultiWasmEnv() as env:
        state = env.reset(slots(), seed=7, ego=2)
        assert state.me == 2
        states = {i: env.observation.state(i) for i in range(4)}
        assert [s.me for s in states.values()] == list(range(4))
        assert states[0].t == 0
        initial = [(r.x, r.z) for r in states[0].robots]
        bots = [ImAI(specs[r.team]) for r in states[0].robots]
        for _ in range(200):
            result = env.step({i: b.act(states[i]) for i, b in enumerate(bots)}, ticks=3)
            states = result.views
        assert states[0].t == pytest.approx(10)
        assert all([r.id for r in s.robots] == [0, 1, 2, 3] for s in states.values())
        assert isinstance(result, StepResult) and result.ego == 2 and not result.done
        assert result.scores == {a: b["total"] for a, b in result.breakdowns.items()}
        positions = [(r.x, r.z) for r in states[0].robots]
        assert len(set(positions)) == 4
        for alliance in ("blue", "red"):
            assert any(math.dist(initial[i], positions[i]) > .2
                       for i, r in enumerate(states[0].robots) if r.alliance == alliance)
        assert set(env.view["score"]) == {"red", "blue"}
        assert all("foulPoints" in score for score in env.view["score"].values())
        assert isinstance(env.view["events"], list)
        for inputs, ticks in [({}, 0), ({4: Inputs()}, 1), ({True: Inputs()}, 1), ([], 1)]:
            with pytest.raises(ValueError):
                env.step(inputs, ticks)
        assert len(env.step({}, ticks=3).views) == 4
        assert env.state().me == 2
        state = env.reset(slots()[:2], hp=False, unload=False, countdown=True)
        assert len(state.robots) == 2 and state.t == -3
        with pytest.raises(RuntimeError, match="unknown team"):
            env.reset([RobotSlot(-1)])


@pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").is_file(), reason="engine assets absent")
def test_contact_fouls_credit_opponent():
    with MultiWasmEnv() as env:
        env.reset([RobotSlot(team=254, alliance=a, start=dict(fx=x, fy=7.4, headingDeg=0))
                   for a, x in [("blue", 7.5), ("red", 9)]], hp=False)
        env.step({0: Inputs(y=1, field_relative=True)}, ticks=600)
        fouls = [e for e in env.view["events"] if e["t"] == "foul"]
        assert any(e["rule"] == "G403" and e["points"] == 30 for e in fouls)
        assert any(e["rule"] == "G418" and e["points"] == 15 for e in fouls)
        assert all(e["by"] == 0 and e["alliance"] == "red" for e in fouls)
        assert env.view["score"]["red"]["foulPoints"] == sum(e["points"] for e in fouls)
        assert env.view["score"]["blue"]["foulPoints"] == 0
        env.step({}, ticks=1)
        assert not any(e["t"] == "foul" for e in env.view["events"])


def test_default_start_lane_spread():
    robots = [RobotSlot(254, a) for a in ("blue", "red") for _ in range(3)]
    starts = default_starts(robots)
    assert [s["fy"] for s in starts] == [2., 6.07, 4.035] * 2
    assert [s["headingDeg"] for s in starts] == [0] * 3 + [180] * 3
    robots[0].start = dict(fx=1, fy=1, headingDeg=0)
    assert default_starts(robots)[1]["fy"] == 2.


@pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").is_file(), reason="engine assets absent")
def test_slot_policy_signature_explicit_idle_and_validation():
    class Policy:
        def __init__(self):
            self.calls = []

        def act(self, state, info):
            self.calls.append((state.me, info))
            return Inputs(y=.5, field_relative=True)

    policy = Policy()
    with MultiWasmEnv() as env:
        for ego in (-1, True, 2):
            with pytest.raises(ValueError, match="ego"):
                env.reset([RobotSlot(254)], ego=ego)
        env.reset([RobotSlot(254, policy=policy), RobotSlot(2910, "red")])
        result = env.step({}, 3)
        assert policy.calls[0][0] == 0 and "scored" in policy.calls[0][1]
        assert result.observation.state(1).me == 1
        env.step({0: None}, 3)
        assert len(policy.calls) == 1
        with pytest.raises(TypeError, match="Inputs"):
            env.step({0: object()})
