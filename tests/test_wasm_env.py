import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from frsbot.core.field import is_active, phase_at, seconds_to_toggle
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.core.state import Phase
from frsbot.replay.convert import convert
from frsbot.replay.rbrp import Replay
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.sim.wasm_env import SERVER, WasmEnv, find_node, view_to_state


def sample_view():
    return dict(phase="auto", elapsed=0, clock=20, firstInactive=None,
                hubs={a: dict(active=True) for a in ("blue", "red")},
                robots=[dict(id=7, team=254, alliance="blue", local=True, pos=[1, .1, -2],
                             quat=[0, .5, 0, .866], vel=[.5, 0, -1], hopperFuel=3, hopperCapacity=72)],
                fuel=dict(count=1, pos=[.1, .2, .3], where=[1]))


def test_view_coordinates_match_replay_converter():
    view = sample_view()
    raw = [2000, 200, -4000, 0, 16384, 0, 28377, 500, 0, -1000, 0, 0, 0, 3, 0]
    view["robots"][0]["quat"] = [0, raw[4] / 32767, 0, raw[6] / 32767]
    replay = Replay({}, {}, [dict(id=7, team=254, alliance="blue")], [dict(at=0)], 15,
                    frames=[(0, raw)], fuel=[(np.array([[200, 400, 600]], np.int16), np.array([1], np.uint8))])
    replay.metas = [[0, dict(phase="auto")]]
    expected, actual = convert(replay)[0], view_to_state(view)
    assert vars(actual.robots[0]) == vars(expected.robots[0])
    assert actual.me == expected.me == 0
    np.testing.assert_allclose(actual.fuel.pos, expected.fuel.pos)
    np.testing.assert_array_equal(actual.fuel.where, expected.fuel.where)
    assert actual.fuel.pos.dtype == np.float32 and actual.fuel.where.dtype == np.uint8


def test_view_countdown_freeplay_empty_and_error():
    view = sample_view()
    view.update(phase="countdown", clock=3)
    for hub in view["hubs"].values():
        hub["active"] = False
    state = view_to_state(view)
    assert state.t == -3 and state.phase == Phase.COUNTDOWN
    assert state.hubs["blue"].seconds_to_toggle == 3
    view.update(phase="transition", elapsed=40, firstInactive=None)
    assert math.isinf(view_to_state(view, freeplay=True).hubs["blue"].seconds_to_toggle)
    view.update(phase="auto", elapsed=0, fuel=dict(count=0, pos=[], where=[]), robots=[])
    assert view_to_state(view).fuel.pos.shape == (0, 3)
    view["fuel"] = dict(count=1, pos=[0, 0, 0], where=[])
    with pytest.raises(ValueError, match="counts disagree"):
        view_to_state(view)


def test_missing_assets_and_lifecycle(tmp_path):
    env = WasmEnv(engine_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="engine assets missing"):
        env.reset(254)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(Inputs())
    with pytest.raises(RuntimeError, match="reset"):
        env.state()
    env.close()
    env.close()


def test_node_path_nvm_fallback_and_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("frsbot.sim.wasm_env.shutil.which", lambda _: "/custom/node")
    assert find_node() == "/custom/node"
    monkeypatch.setattr("frsbot.sim.wasm_env.shutil.which", lambda _: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="Node.js is required"):
        find_node()
    for version in ("v9.0.0", "v24.22.2"):
        node = tmp_path / ".nvm" / "versions" / "node" / version / "bin" / "node"
        node.parent.mkdir(parents=True)
        node.write_text("")
        node.chmod(0o755)
    assert "v24.22.2" in find_node()


def test_local_robot_index_and_red_coordinates():
    view = sample_view()
    view["robots"][0]["local"] = False
    view["robots"].append({**view["robots"][0], "id": 8, "alliance": "red", "local": True,
                           "pos": [2, .1, -3]})
    state = view_to_state(view)
    assert state.me == 1
    assert (state.robots[1].x, state.robots[1].z, state.robots[1].alliance) == (2, -3, "red")


@pytest.fixture
def env():
    if not (ENGINE_DIR / "manifest.json").exists():
        pytest.skip("requires python3 -m frsbot.sim.engine.fetch")
    with WasmEnv() as instance:
        yield instance


@pytest.mark.parametrize("team", [2910, 254])
def test_real_engine_accelerates_to_max_speed(env, team):
    state = env.reset(team, start=dict(fx=0, fy=0, headingDeg=0), seed=1)
    speeds = []
    for ticks in (6, 24, 30):
        state = env.step(Inputs(y=1, field_relative=True), ticks)
        robot = state.robots[state.me]
        speeds.append(math.hypot(robot.vx, robot.vz))
    assert 0 < speeds[0] < speeds[1] < speeds[2]
    maximum = next(s.max_speed for s in load_specs() if s.team == team)
    assert speeds[-1] == pytest.approx(maximum, rel=.03)
    assert state.t == 1
    assert "WASM" in env.engine
    assert state.fuel.pos.shape == (504, 3) and state.fuel.where.shape == (504,)
    assert isinstance(env.view["hubs"]["blue"]["net"], list)
    assert "score" in env.view and "gate" in env.view["robots"][0]


def test_real_engine_intake_collects_fuel(env):
    state = env.reset(2910, start=dict(fx=7.5, fy=4, headingDeg=180), seed=1)
    initial = state.robots[0].hopper
    state = env.step(Inputs(y=1, intake=True, field_relative=True), ticks=60)
    assert state.robots[0].hopper > initial


def test_real_engine_matches_entire_schedule(env):
    state = env.reset(254, seed=1, countdown=True)
    assert state.t == -3
    state = env.step(Inputs(), 180)
    for boundary in (20, 23, 33, 58, 83, 108, 133, 163, 166):
        state = env.step(Inputs(), round((boundary - state.t) * 60) - 1)
        for ticks in (0, 1):
            if ticks:
                state = env.step(Inputs(), ticks)
            assert state.phase == phase_at(state.t)
            for alliance, hub in state.hubs.items():
                first = env.view["firstInactive"]
                assert hub.active == is_active(alliance, state.t, first)
                assert hub.seconds_to_toggle == seconds_to_toggle(alliance, state.t, first)
    assert state.t == 166 and state.phase == Phase.DONE


def test_real_engine_reset_determinism_freeplay_and_errors(env):
    env.reset(2910, seed=123)
    first = env.step(Inputs(intake=True), 60)
    expected_fuel = first.fuel.pos.copy()
    env.reset(2910, seed=123)
    second = env.step(Inputs(intake=True), 60)
    assert first.robots == second.robots
    np.testing.assert_array_equal(expected_fuel, second.fuel.pos)
    with pytest.raises(RuntimeError, match="unknown team"):
        env.reset(-1)
    with pytest.raises(RuntimeError, match="unknown command"):
        env._request("invalid")
    with pytest.raises(RuntimeError, match="unknown robot"):
        env._request("step", inputs={"99": {}}, n=1)
    with pytest.raises(RuntimeError, match=r"must be in"):
        env.step(Inputs(x=2))
    for ticks in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="ticks"):
            env.step(Inputs(), ticks)
    assert env.state().t == 1
    state = env.reset(254, freeplay=True)
    state = env.step(Inputs(), 2400)
    assert all(h.active and math.isinf(h.seconds_to_toggle) for h in state.hubs.values())
    process = env.process
    env.close()
    assert process.poll() == 0


def test_server_malformed_commands(env):
    result = subprocess.run([find_node(), str(SERVER), str(ENGINE_DIR)],
                            input='not json\n{"cmd":"state"}\n{"cmd":"close"}\n',
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["ok"] for r in responses] == [False, False, True]
