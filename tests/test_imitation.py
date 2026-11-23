from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState
from frsbot.imitation.buttons import agreement, interval_buttons, transition_buttons
from frsbot.imitation.contacts import FIELD, FUEL, ROBOT, contact_flags
from frsbot.imitation.dataset import examples
from frsbot.imitation.labels import label_segments
from frsbot.imitation.ownership import carried_owners
from frsbot.imitation.recover import excluded, windows, write_run
from frsbot.imitation.report import summarize
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.sim.wasm_env import JsonWasmEnv


SPECS = {spec.team: spec for spec in load_specs()}


def state(x, hopper, pos, where, t=0):
    robot = RobotState(54, 254, "blue", x, 0., 0., 0., 0., hopper, 72)
    return GameState(t, Phase.AUTO, {a: HubState(True, 20.) for a in ("blue", "red")}, [robot],
                     FuelState(np.array(pos, np.float32), np.array(where, np.uint8)), 0)


def test_labels_use_local_transfers_and_shooting_arc():
    collect = [state(0, 0, [[.4, .1, 0]], [0]), state(.3, 1, [[.4, .5, 0]], [2])]
    assert label_segments(collect, SPECS)[0].kind == "Collect"
    assert label_segments(collect, SPECS)[0].point == pytest.approx((.4, 0.))

    score = [state(-6, 5, [[-6, .6, 0]], [2]), state(-6, 4, [[-5.8, 1.2, 0]], [1]),
             state(-6, 4, [[-3.645, 1.8, 0]], [3])]
    assert label_segments(score, SPECS)[0].kind == "Score"
    assert label_segments(score, SPECS)[0].point == (-6, 0.)
    assert label_segments([score[0], score[0]], SPECS)[0].kind == "Stage"


def test_solo_only_driver_exclusion_checks_both_replay_headers():
    replay = SimpleNamespace(robots=[dict(name="Pro")], meta={"robots": [dict(name=" 404 ")]})
    assert excluded(replay, "solo")
    assert not excluded(replay, "multiplayer")
    replay.meta = {}
    assert not excluded(replay, "solo")
    replay.robots[0]["name"] = "404"
    assert excluded(replay, "solo")


def test_segments_preserve_tick_duration_and_drop_capture_gaps():
    ticks = list(range(-4, 241, 2))
    ticks = [tick for tick in ticks if not 150 <= tick <= 170]
    states = [SimpleNamespace(t=tick / 60) for tick in ticks]
    segment, = windows(ticks, states, 6)
    assert [ticks[i] for i in segment] == list(range(0, 121, 6))


def test_carried_owner_uses_intake_event_until_ball_leaves():
    start = state(0, 1, [[0., .5, 0.], [0., .5, 0.]], [2, 2])
    start = replace(start, robots=[start.robots[0], replace(start.robots[0], id=55, x=4)])
    ground = replace(start, fuel=replace(start.fuel, where=np.array([0, 2])))
    snapshots = [start, start, ground, *([ground] * 6), start]
    owners = carried_owners(snapshots, [[0, {"t": "intake", "robot": 55, "fuel": 0}]], {0, 2, 9})
    assert owners == {0: [1, 0], 2: [-1, 0], 9: [0, 0]}


def test_transition_buttons_attribute_pickup_and_release_and_skip_capture_gaps():
    start = state(0, 0, [[0., .08, 0.], [4., .2, 0.]], [0, 1])
    start = replace(start, robots=[start.robots[0], replace(start.robots[0], id=55, x=4.)])
    held = replace(start, fuel=replace(start.fuel, where=np.array([2, 2])))
    shot = replace(start, fuel=replace(start.fuel, where=np.array([1, 2])))
    states = [start, held, shot, start]
    ticks = [0, 2, 6, 40]
    labels = transition_buttons(states, ticks, [[1, {"t": "intake", "robot": 55, "fuel": 0}]], radius=1)
    assert np.flatnonzero(labels[:, 1, 0]).tolist() == [1, 2, 3]
    assert np.flatnonzero(labels[:, 1, 1]).tolist() == [5, 6, 7]
    assert not labels[:, 0, :].any()
    assert interval_buttons(labels, ticks, [0, 1, 2]) == [
        [[False, False], [True, False]], [[False, False], [True, True]],
    ]


def test_button_agreement_aligns_overlap_by_robot_tick():
    new = np.array([[[1, 1]], [[1, 2]]], dtype=np.uint8)
    old = {"tick": np.array([1, 6]), "ticks": np.array([2, 1]),
           "inputs": np.array([[[1, 3]], [[1, 0]]], dtype=np.uint8)}
    assert agreement(np.array([0, 6]), np.array([2, 2]), new, old).tolist() == [
        [1, 0, 0, 1], [0, 1, 1, 0],
    ]


def test_contact_proxies_detect_robot_wall_hub_and_fuel_stall():
    start = state(0, 0, [[0., .08, 0.], [.1, .08, 0.], [0., .08, .1]], [0, 0, 0])
    robot = start.robots[0]
    assert contact_flags([start], SPECS) == [0]
    assert contact_flags([replace(start, robots=[replace(robot, vx=2)]), start], SPECS) == [FUEL]
    for x in (8., 3.645):
        assert contact_flags([replace(start, robots=[replace(robot, x=x)])], SPECS) == [FIELD]
    duel = replace(start, robots=[robot, replace(robot, id=55, x=.8)])
    assert contact_flags([duel], SPECS) == [ROBOT, ROBOT]


def test_dataset_round_trip_and_segment_endpoint_metrics(tmp_path):
    states = [state(-6, 2, [[-6, .6, 0]], [2], i / 60) for i in (0, 2, 120)]
    goal = label_segments(states, SPECS)
    result = dict(
        controls=[[[.1234567, 0., 0., True, False]], [[0., 0., 0., False, True]]],
        residuals=[[[.5, 0., 0., 4, .5, .1]], [[.05, 0., 0., 0, 1., .75]]],
        initialHoppers=[2],
    )
    path = tmp_path / "run.npz"
    write_run(path, "solo", states, [0, 2, 120], [[0, 1, 2]], [goal], [result])
    [(obs, inputs, residual, label), _] = list(examples(path))
    assert obs.robots[obs.me] == states[0].robots[0]
    np.testing.assert_array_equal(obs.fuel.pos, states[0].fuel.pos)
    assert inputs == Inputs(x=.1234567, intake=True, field_relative=True)
    assert residual["pose_m"] == .5
    assert label == goal[0]
    metrics = summarize([path])["solo"]
    assert metrics["segments"] == 1
    assert metrics["examples"] == 2
    assert metrics["flagged"] == 1
    assert metrics["median_cm"] == pytest.approx(5)
    assert (metrics["initial_hopper"], metrics["hopper"], metrics["fuel"]) == (1., 1., .75)
    assert metrics["labels"] == {"Stage": 1}
    with np.load(path) as run:
        assert run["weight"][:, 0] == pytest.approx(np.exp(-np.array([.5, .05]) / .1))


def test_report_excludes_whole_contact_segments(tmp_path):
    path = tmp_path / "run.npz"
    residual = np.zeros((2, 2, 6))
    residual[:, :, 0] = [[.01, .03], [.4, .2]]
    np.savez(path, segment=[0, 1], source="multiplayer", residual=residual, contacts=[[0, 0], [FIELD, 0]],
             goal_kind=[["Stage", "Stage"]] * 2, initial_hopper=[[0, 0]] * 2, hopper=[[0, 0]] * 2, flagged=[],
             ticks=[2, 2], inputs=np.zeros((2, 2, 10), dtype=np.uint8))
    metrics = summarize([path])["all"]
    assert metrics["contact_free_count"] == 2
    assert metrics["contact_free_pose"] == pytest.approx([2, 2.8])
    assert metrics["causes"]["Wall/hub"] == (1, 1, 40.)
    assert metrics["causes"]["No detected contact"] == pytest.approx((3, 1, 16.6))


@pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").exists(), reason="requires real engine assets")
def test_recovery_does_not_prime_hopper_or_search_recorded_buttons():
    start = dict(fx=0, fy=0, headingDeg=0)
    spec = SPECS[254]
    with JsonWasmEnv() as env:
        env.reset(254, start=start, seed=12, hp=False, unload=False, freeplay=True)
        robot = env.view["robots"][0]
        fuel = {k: env.view["fuel"][k] for k in ("pos", "where")}
        body = {k: robot[k] for k in ("pos", "quat", "vel")}
        body["vel"] = [.5, 0., .2]
        fuel["where"][:12] = [2] * 12
        frames = [dict(tick=tick, bodies=[dict(body, hopper=10, omega=.4)], fuel=fuel,
                       buttons=[[True, False]]) for tick in (0, 2)]
        env._request("new", config=dict(team=254, recovery=True))
        result = env._request("recover", segment=dict(
            robots=[dict(id=0, team=254, alliance="blue", start=start, speed=spec.max_speed, turnRate=spec.turn_rate)],
            seed=12, frames=frames, initialFuel=dict(fuel, vel=[0.] * len(fuel["pos"]))))
    assert result["initialHoppers"] == [0]
    assert result["controls"][0][0][3:] == [True, False]
    assert result["initialBodies"][0]["vel"] == pytest.approx(body["vel"])
    assert result["initialBodies"][0]["omega"] == pytest.approx(.4)


@pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").exists(), reason="requires real engine assets")
def test_recovers_real_wasm_movement_open_loop():
    start = dict(fx=0, fy=0, headingDeg=0)
    spec = SPECS[254]
    frames = []
    with JsonWasmEnv() as env:
        env.reset(254, start=start, seed=12, hp=False, unload=False, freeplay=True)
        for tick in range(0, 121, 6):
            if tick:
                env.step(Inputs(y=-.23, field_relative=True), ticks=6)
            robot = env.view["robots"][0]
            fuel = {k: env.view["fuel"][k] for k in ("pos", "where")}
            body = {k: robot[k] for k in ("pos", "quat", "vel")}
            frames.append(dict(tick=tick, bodies=[dict(body, hopper=robot["hopperFuel"], omega=0)],
                               fuel=fuel, buttons=[[False, False]]))
        actual_displacement = abs(frames[-1]["bodies"][0]["pos"][0] - frames[0]["bodies"][0]["pos"][0])
        assert actual_displacement > .5
        env._request("new", config=dict(team=254, recovery=True))
        result = env._request("recover", segment=dict(
            robots=[dict(id=0, team=254, alliance="blue", start=start, speed=spec.max_speed, turnRate=spec.turn_rate)],
            seed=12, frames=frames, initialFuel=dict(frames[0]["fuel"], vel=[0.] * len(fuel["pos"]))))
    assert any(abs(row[0][1]) > .12 for row in result["controls"])
    assert result["residuals"][-1][0][0] < .1
