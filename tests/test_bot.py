import ast
import math
from pathlib import Path

import numpy as np
import pytest

from frsbot.bot.driver import Driver, intersects, inverse_deadzone
from frsbot.bot.imai import ImAI
from frsbot.bot.planner import Goal, Planner
from frsbot.core.field import is_active, phase_at, seconds_to_toggle
from frsbot.core.specs import load_specs
from frsbot.core.state import FuelState, GameState, HubState, RobotState
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.sim.wasm_env import WasmEnv


@pytest.fixture
def spec():
    return next(s for s in load_specs() if s.team == 254)


def observation(hopper=0, t=0, alliance="blue"):
    return GameState(t, phase_at(t),
                     {a: HubState(is_active(a, t, "blue"), seconds_to_toggle(a, t, "blue")) for a in ("blue", "red")},
                     [RobotState(0, 254, alliance, -6, 1, 0, 0, 0, hopper, 72)],
                     FuelState(np.array([[-1, .075, 1]] * 10, np.float32), np.zeros(10, np.uint8)), 0)


@pytest.mark.parametrize("hopper,t,toggle,kind", [(5, 0, 1, "Score"), (65, 0, 20, "Score"),
                                                 (71, 34, 24, "Stage"), (5, 34, .1, "Stage"),
                                                 (0, 0, 20, "Collect"), (10, 34, 24, "Collect")])
def test_planner_rules(spec, hopper, t, toggle, kind):
    state = observation(hopper, t)
    state.hubs["blue"].seconds_to_toggle = toggle
    assert Planner(spec).plan(state).kind == kind


def test_actual_capacity_and_empty_fuel(spec):
    state = observation(43)
    state.robots[0].capacity = 50
    assert Planner(spec).plan(state).kind == "Score"
    state = observation(1)
    state.fuel = FuelState(np.empty((0, 3), np.float32), np.empty(0, np.uint8))
    assert Planner(spec).plan(state).kind == "Score"
    state.robots[0].hopper = 0
    assert Planner(spec).plan(state).kind == "Stage"


def test_last_three_seconds_and_staging_transition(spec):
    state = observation(5, 160)
    assert Planner(spec).plan(state).kind == "Score"
    state.robots[0].x = 0
    bot = ImAI(spec)
    inputs = bot.act(state)
    assert not inputs.score and not inputs.intake and inputs.x == inputs.y == inputs.turn == 0
    state = observation(5, 57.9)
    planner = Planner(spec)
    assert planner.plan(state).kind == "Stage"
    state.t = 58
    state.phase = phase_at(58)
    state.hubs["blue"] = HubState(True, 25)
    assert planner.plan(state).kind == "Score"


def test_grid_hysteresis_and_airborne_filter(spec):
    state = observation()
    state.robots[0].x, state.robots[0].z = -1, 0
    points = [[-1, .075, 1]] * 10 + [[-1, .075, -1]] * 11 + [[-1, .075, 0]] * 100
    state.fuel = FuelState(np.array(points, np.float32), np.array([0] * 21 + [1] * 100, np.uint8))
    planner = Planner(spec)
    assert planner.clusters(state)[0][1][1] == -1
    planner.goal = Goal("Collect", (-1, 1))
    assert planner.clusters(state)[0][1][1] == 1


def test_score_radius_and_zone(spec):
    planner, state = Planner(spec), observation(72)
    for alliance, x in (("blue", 0), ("red", 0)):
        state.robots[0].alliance, state.robots[0].x = alliance, x
        point = planner.score_spot(state)
        assert math.dist(point, planner.hub(state)) == pytest.approx(3.5)
        assert point[0] < -4.3 if alliance == "blue" else point[0] > 4.3


def test_replanning_interval_completion_and_invalid_goal(spec):
    state, bot = observation(), ImAI(spec)
    calls, original = [], bot.planner.plan

    def plan(s):
        calls.append(s.t)
        return original(s)

    bot.planner.plan = plan
    bot.act(state)
    state.t = .1
    bot.act(state)
    assert calls == [0]
    state.t = .5
    bot.act(state)
    assert calls == [0, .5]
    state.robots[0].x, state.robots[0].z = bot.goal.point
    state.t = .51
    bot.act(state)
    assert calls[-1] == .51
    state.fuel.where[:] = 2
    state.t = .52
    bot.act(state)
    assert bot.goal.kind == "Stage"


@pytest.mark.parametrize("alliance", ["blue", "red"])
@pytest.mark.parametrize("around_hub", [False, True])
def test_driver_arrives_without_overshoot(spec, alliance, around_hub):
    state = observation(alliance=alliance)
    robot = state.robots[0]
    robot.x, robot.z = ((-3.4 if alliance == "blue" else 3.4), 2.5) if around_hub else ((-6 if alliance == "blue" else 6), -1)
    state.fuel.where[:] = 2
    driver, goal = Driver(spec), Goal("Stage", (robot.x, -2.5 if around_hub else 1))
    maximum, minimum = robot.z, robot.z
    for _ in range(600):
        command = driver.act(goal, state)
        norm = math.hypot(command.x, command.y)
        shape = max(0, (norm - .12) / .88)
        sign = 1 if alliance == "blue" else -1
        wanted = np.array([command.y, command.x]) / norm * shape * spec.max_speed * sign if norm else np.zeros(2)
        velocity = np.array([robot.vx, robot.vz], dtype=float)
        change = wanted - velocity
        velocity += change * min(1, spec.accel / 60 / max(1e-9, np.linalg.norm(change)))
        robot.vx, robot.vz = velocity
        robot.x += robot.vx / 60
        robot.z += robot.vz / 60
        state.t += 1 / 60
        maximum = max(maximum, robot.z)
        minimum = min(minimum, robot.z)
        hub_x = -3.645 if alliance == "blue" else 3.645
        assert not (abs(robot.x - hub_x) < .597 and abs(robot.z) < .597)
    assert math.dist((robot.x, robot.z), goal.point) < .3
    assert minimum >= -2.53 if around_hub else maximum <= 1.03
    assert math.hypot(robot.vx, robot.vz) < .01


def test_avoidance_and_deadzone(spec):
    driver = Driver(spec)
    point = driver.waypoint((-6, 0), (-1, 0))
    assert abs(point[1]) > 1
    assert not intersects((-6, 0), point, (-4.3, -3, -.6, .6))
    assert inverse_deadzone(0) == 0
    assert inverse_deadzone(.5) == pytest.approx(.56)
    assert inverse_deadzone(-1) == -1


def test_closed_gate_recovery_and_unknown_gates(spec):
    state, driver = observation(72), Driver(spec)
    goal = Goal("Score", (-6, 1))
    driver.act(goal, state)
    assert not driver.recovering
    state.robots[0].gate = dict(open=False, heading=True)
    driver.act(goal, state)
    state.t = 1.6
    driver.act(goal, state)
    assert driver.recovering
    state.robots[0].gate["open"] = True
    driver.act(goal, state)
    assert not driver.recovering


def test_bot_backend_dependency_boundary():
    for path in (Path(__file__).resolve().parents[1] / "frsbot" / "bot").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            assert not any(n.startswith(("frsbot.sim", "frsbot.replay", "frsbot.live")) for n in names)


@pytest.mark.parametrize("alliance", ["blue", "red"])
def test_bot_real_engine_scores_without_human_player(spec, alliance):
    if not (ENGINE_DIR / "manifest.json").exists():
        pytest.skip("requires python3 -m frsbot.sim.engine.fetch")
    with WasmEnv() as env:
        state = env.reset(spec.team, alliance=alliance, seed=1, hp=False, unload=False)
        bot, hub_x = ImAI(spec), []
        while state.t < 40:
            state = env.step(bot.act(state), 3)
            hub_x.extend(state.fuel.pos[state.fuel.where == 3][:, 0].tolist())
            tally = env.view["score"][alliance]
            if tally["autoFuel"] + tally["teleFuel"] > 0:
                break
        assert tally["autoFuel"] + tally["teleFuel"] > 0
        assert hub_x and (max(hub_x) < 0 if alliance == "blue" else min(hub_x) > 0)
