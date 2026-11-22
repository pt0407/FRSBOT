import math

import pytest

from frsbot.bot.imai import ImAI
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.sim.multi_env import MultiWasmEnv


def slots():
    return [dict(team=team, alliance=alliance, start=dict(fx=x, fy=y, headingDeg=0))
            for team, alliance, x, y in [(254, "blue", 1, 2), (2910, "red", 15, 2),
                                        (1114, "blue", 1, 6), (4414, "red", 15, 6)]]


def test_multi_validation_before_reset():
    with MultiWasmEnv() as env:
        for robots in ([], None, [{}] * 7):
            with pytest.raises(ValueError, match="slots"):
                env.reset(robots)
        with pytest.raises(RuntimeError, match="reset"):
            env.step({})


@pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").is_file(), reason="engine assets absent")
def test_two_vs_two_imai():
    specs = {s.team: s for s in load_specs()}
    with MultiWasmEnv() as env:
        states = env.reset(slots(), seed=7)
        assert [s.me for s in states] == list(range(4))
        assert states[0].t == 0
        initial = [(r.x, r.z) for r in states[0].robots]
        bots = [ImAI(specs[r.team]) for r in states[0].robots]
        for _ in range(200):
            states = env.step({i: b.act(states[i]) for i, b in enumerate(bots)}, ticks=3)
        assert states[0].t == pytest.approx(10)
        assert all([r.id for r in s.robots] == [0, 1, 2, 3] for s in states)
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
        assert len(env.step({}, ticks=3)) == 4
        assert [s.me for s in env.state()] == list(range(4))
        states = env.reset(slots()[:2], hp=False, unload=False, countdown=True)
        assert len(states) == 2 and states[0].t == -3
        with pytest.raises(RuntimeError, match="unknown team"):
            env.reset([dict(team=-1, alliance="blue")])


@pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").is_file(), reason="engine assets absent")
def test_contact_fouls_credit_opponent():
    with MultiWasmEnv() as env:
        env.reset([dict(team=254, alliance=a, start=dict(fx=x, fy=7.4, headingDeg=0))
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
