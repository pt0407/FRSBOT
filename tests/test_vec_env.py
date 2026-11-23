import json
import struct
import subprocess

import numpy as np
import pytest

from frsbot.bot.planner import Goal
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.sim.ipc import VEC_SERVER, EngineProcess, encode_command
from frsbot.sim.policies import GoalImAIPolicy, IdlePolicy, ImAIPolicy, ReplayPolicy
from frsbot.sim.vec_env import Latency, RobotSlot, VecWasmEnv, default_starts
from frsbot.sim.wasm_env import WasmEnv, find_node

SPECS = {s.team: s for s in load_specs()}
needs_engine = pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").exists(), reason="requires python3 -m frsbot.sim.engine.fetch")


def test_commands_and_starts_offline():
    assert encode_command(2, None) == bytes([2, 2])
    assert len(encode_command(1, Inputs(x=.5, intake=True))) == 16
    assert len(encode_command(0, Goal("Score", (1.0, 2.0)))) == 11
    with pytest.raises(TypeError):
        encode_command(0, "go")
    with pytest.raises(ValueError):
        encode_command(0, Goal("Score", (float("nan"), 0.0)))
    starts = default_starts([RobotSlot(254, "blue"), RobotSlot(2910, "red"), RobotSlot(1114, "blue", dict(fx=1, fy=2, headingDeg=3))])
    assert starts[0]["fx"] < 8.27 < starts[1]["fx"] and starts[2] == dict(fx=1, fy=2, headingDeg=3)
    rng = np.random.default_rng(0)
    draws = [Latency().draw(rng) for _ in range(200)]
    assert all(0 <= d <= 9 and 1 <= h <= 3 for d, h in draws) and {h for _, h in draws} == {1, 2, 3}
    with pytest.raises(ValueError):
        Latency((.2, .1)).draw(rng)


def test_replay_policy_follows_recording():
    recording = dict(samples=[dict(tick=0, robots=[dict(pos=[-4, 0, 1], hopperFuel=0)]),
                              dict(tick=80, robots=[dict(pos=[-2, 0, 1], hopperFuel=5)]),
                              dict(tick=160, robots=[dict(pos=[-3, 0, 0], hopperFuel=1)])])
    policy = ReplayPolicy(recording)

    class S:
        t = 0.5
    assert policy.act(S()) == Goal("Collect", (-2.0, 1.0))
    S.t = 2
    assert policy.act(S()) == Goal("Score", (-3.0, 0.0))
    assert IdlePolicy().act(S()) is None


@needs_engine
def test_cumulative_robot_fouls_and_alliance_credit_reset():
    with WasmEnv() as env:
        env.reset(robots=[RobotSlot(254, a, dict(fx=x, fy=7.4, headingDeg=0))
                          for a, x in [("blue", 7.5), ("red", 9)]], hp=False)
        assert all(d["fouls"] == 0 for d in env.observation.driver)
        result = env.step({0: Inputs(y=1, field_relative=True)}, ticks=600)
        counts = [d["fouls"] for d in result.observation.driver]
        assert counts[0] >= 2 and counts[1] == 0
        assert result.breakdowns["red"]["foulPoints"] >= 45
        assert result.breakdowns["blue"]["foulPoints"] == 0
        result = env.step({}, ticks=1)
        assert [d["fouls"] for d in result.observation.driver] == counts
        env.reset(team=254)
        assert env.observation.driver[0]["fouls"] == 0


@needs_engine
def test_multi_robot_views_scores_and_policies():
    robots = [RobotSlot(254, "blue"), RobotSlot(2910, "red", policy=GoalImAIPolicy(SPECS[2910])),
              RobotSlot(1114, "blue", policy=ImAIPolicy(SPECS[1114])), RobotSlot(4414, "red")]
    with WasmEnv() as env:
        state = env.reset(robots=robots, ego=2, seed=3)
        assert state.me == 2 and [r.team for r in state.robots] == [254, 2910, 1114, 4414]
        assert [r.alliance for r in state.robots] == ["blue", "red", "blue", "red"]
        start = [(r.x, r.z) for r in state.robots]
        result = None
        for _ in range(40):
            result = env.step({0: Inputs(y=1, field_relative=True)}, 6)
        assert set(result.views) == {0, 1, 2, 3} and result.views[1].me == 1 and result.ego == 2
        assert result.views[0].robots is result.views[3].robots and result.t == 4
        moved = [abs(r.x - s[0]) + abs(r.z - s[1]) for r, s in zip(result.views[0].robots, start)]
        assert moved[0] > 1 and moved[1] > .5 and moved[2] > .5 and moved[3] < .05
        assert set(result.scores) == {"blue", "red"} and result.breakdowns["blue"]["total"] == result.scores["blue"]
        assert env.view["robots"][2]["local"] and not env.view["robots"][0]["local"]
        assert env.step(Inputs(), 1).me == 2
        with pytest.raises(RuntimeError, match="unknown robot"):
            env.step({7: Inputs()})
        with pytest.raises(RuntimeError, match="must list 1 to 6"):
            env.reset(robots=[RobotSlot(254)] * 7)


@needs_engine
def test_vec_matches_are_independent_and_deterministic():
    config = lambda seed: dict(robots=[RobotSlot(254, "blue", policy=GoalImAIPolicy(SPECS[254])), RobotSlot(2910, "red")], seed=seed)
    with VecWasmEnv(2, 2) as env:
        assert len(env) == 4
        env.reset([config(1), config(2), config(1), config(2)])
        for _ in range(30):
            results = env.step(None, 10)
        a, b, c, d = (r.views[0] for r in results)
        assert a.robots == c.robots and b.robots == d.robots and a.robots != b.robots
        np.testing.assert_array_equal(a.fuel.pos, c.fuel.pos)
        assert env.step([{1: Inputs(x=1)}], 1, matches=[3])[0].t == pytest.approx(5 + 1 / 60)
        with pytest.raises(RuntimeError, match="unknown team"):
            env.reset([config(1), dict(robots=[RobotSlot(-5)])])
        assert len(env.step(None, 1)) == 4


@needs_engine
def test_rollout_runs_each_match_to_the_requested_time():
    with VecWasmEnv(2, 2) as env:
        env.reset([dict(robots=[RobotSlot(254, policy=GoalImAIPolicy(SPECS[254]))], seed=s) for s in (1, 2, 1, 2)])
        env.step([{}], 7, matches=[1])
        results = env.rollout(20, 6)
        assert [r.t for r in results] == [20, 20, 20, 20]
        assert results[0].views[0].robots == results[2].views[0].robots and results[0].scores["blue"] > 0
        assert env.rollout(20, 6, matches=[3])[0].t == 20
        with pytest.raises(ValueError):
            env.rollout(30, 0)


@needs_engine
def test_latency_delays_observations_and_holds_inputs():
    with WasmEnv(latency=(.1, .1)) as env:
        state = env.reset(254, seed=5)
        delay, hold = env.delay_ticks, env.hold_ticks
        assert delay == 6 and 1 <= hold <= 3 and state.t == 0
        state = env.step(Inputs(), 3)
        assert state.t == 0
        state = env.step(Inputs(), 30)
        assert state.t == pytest.approx((33 - delay) / 60)
    with WasmEnv() as plain, WasmEnv(latency=(0, 0)) as held:
        plain.reset(254, seed=5)
        held.reset(254, seed=5)
        lag = held.hold_ticks
        for env in (plain, held):
            env.step(Inputs(), 1)
        a, b = plain.step(Inputs(y=1, field_relative=True), 1), held.step(Inputs(y=1, field_relative=True), 1)
        assert (a.robots[0].vx != 0) and (b.robots[0].vx == 0) == (lag > 1)


@needs_engine
def test_vec_server_rejects_malformed_frames():
    frames = [bytes([9]), bytes([2]) + struct.pack("<HHH", 1, 1, 0), bytes([1]) + b"{not json", bytes([5])]
    data = b"".join(struct.pack("<I", len(f)) + f for f in frames)
    result = subprocess.run([find_node(), str(VEC_SERVER), str(ENGINE_DIR)], input=data, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    out, statuses = result.stdout, []
    while out:
        size = struct.unpack_from("<I", out)[0]
        statuses.append(out[4])
        out = out[4 + size:]
    assert statuses == [0, 1, 1, 1, 0]


@needs_engine
def test_engine_process_matches_json_backend():
    from frsbot.sim.wasm_env import JsonWasmEnv
    with JsonWasmEnv() as old, WasmEnv() as new:
        a, b = old.reset(2910, seed=9), new.reset(2910, seed=9)
        for inputs in (Inputs(y=1, intake=True, field_relative=True), Inputs(x=-.5, turn=.3, score=True)):
            a, b = old.step(inputs, 45), new.step(inputs, 45)
        assert a.t == b.t and a.phase == b.phase and a.hubs == b.hubs
        for ra, rb in zip(a.robots, b.robots):
            assert (ra.id, ra.team, ra.hopper, ra.gate, ra.mode) == (rb.id, rb.team, rb.hopper, rb.gate, rb.mode)
            assert ra.x == pytest.approx(rb.x, abs=1e-6) and ra.a == pytest.approx(rb.a, abs=1e-6)
        np.testing.assert_allclose(a.fuel.pos, b.fuel.pos, atol=1e-6)
        np.testing.assert_array_equal(a.fuel.where, b.fuel.where)
