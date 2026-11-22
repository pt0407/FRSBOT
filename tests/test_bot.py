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
    assert not inputs.score and not inputs.intake and (inputs.x or inputs.y)
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


def test_vision_profile_uses_validated_arc_angles():
    spec = next(s for s in load_specs() if s.team == 2910)
    planner, state = Planner(spec), observation(47)
    for alliance in ("blue", "red"):
        state.robots[0].alliance = alliance
        point = planner.score_spot(state)
        cx, cz = planner.hub(state)
        theta = math.atan2(point[1] - cz, point[0] - cx)
        back = math.pi if alliance == "blue" else 0
        error = (theta - back + math.pi) % (2 * math.pi) - math.pi
        assert abs(math.degrees(error)) == pytest.approx(35)


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
    state.t = 2.1
    driver.act(goal, state)
    assert driver.recovering
    state.robots[0].gate["open"] = True
    state.robots[0].hopper -= 1
    driver.act(goal, state)
    assert not driver.recovering


def test_progress_recovery_open_gate_and_residual_abandon(spec):
    state, driver = observation(3), Driver(spec)
    state.robots[0].gate = {"open": True}
    goal = Goal("Score", (-6, 1))
    driver.act(goal, state)
    state.t = 1.01
    driver.act(goal, state)
    assert driver.recovery_stage == 1
    assert not driver.act(goal, state).score
    state.t = 2.7
    driver.act(goal, state)
    assert driver.recovery_point is not None
    state.robots[0].x, state.robots[0].z = driver.recovery_point
    state.t = 3.8
    driver.act(goal, state)
    assert driver.wiggle_until > state.t
    state.t = 5.1
    driver.act(goal, state)
    assert driver.abandon


def test_closed_to_open_transition_does_not_count_closed_time_as_open(spec):
    state, driver = observation(30), Driver(spec)
    state.robots[0].gate = {"open": False}
    goal = Goal("Score", (-6, 1))
    driver.act(goal, state)
    state.t = 1.5
    driver.act(goal, state)
    state.robots[0].gate["open"] = True
    state.t = 1.6
    driver.act(goal, state)
    assert not driver.recovering
    state.t = 2.01
    driver.act(goal, state)
    assert driver.recovering


def test_moving_collection_uses_turret_without_disabling_intake_steering():
    spec = next(s for s in load_specs() if s.team == 4414)
    state = observation(20)
    driver = Driver(spec)
    inputs = driver.act(Goal("Collect", (-7, 2)), state)
    assert inputs.score and inputs.intake and inputs.turn != 0
    state.hubs["blue"].active = False
    assert not driver.act(Goal("Collect", (-7, 2)), state).score


@pytest.mark.parametrize("team", [254, 2910, 4414])
def test_reaim_pauses_score_while_requesting_manual_heading(team):
    spec = next(s for s in load_specs() if s.team == team)
    state, driver = observation(20), Driver(spec)
    state.robots[0].gate = {"open": True}
    goal = Goal("Score", (-6, 1))
    driver.act(goal, state)
    state.t = 1.01
    inputs = driver.act(goal, state)
    assert driver.recovering and not inputs.score


@pytest.mark.parametrize("alliance,x", [("blue", 6), ("red", -6)])
def test_late_return_does_not_mistake_opponent_zone_for_own(spec, alliance, x):
    state = observation(20, 160, alliance)
    state.robots[0].x = x
    goal = Planner(spec).plan(state)
    assert goal.kind == "Score"
    assert goal.point[0] < -4.3 if alliance == "blue" else goal.point[0] > 4.3


def test_recovery_collection_does_not_return_empty_after_two_seconds(spec):
    state, bot = observation(3, 5), ImAI(spec)
    bot.collect_until, bot.collect_deadline, bot.collect_target = 4, 10, 10
    bot.act(state)
    assert bot.goal.kind == "Collect" and bot.collect_target == 10
    state.robots[0].hopper = 12
    bot.act(state)
    assert bot.collect_target == 0


def test_bot_backend_dependency_boundary():
    for path in (Path(__file__).resolve().parents[1] / "frsbot" / "bot").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            assert not any(n.startswith(("frsbot.sim", "frsbot.replay", "frsbot.live")) for n in names)


def test_measured_score_time_and_deadline(spec):
    planner, state = Planner(spec), observation(40, 150)
    time = planner.score_time(state)
    state.robots[0].hopper = 60
    assert planner.score_time(state) > time
    state.t = 158
    assert planner.plan(state).kind == "Score"
    state.robots[0].hopper = 0
    state.t = 162
    assert planner.plan(state).kind == "Stage"


def test_empty_hopper_resets_recovery(spec):
    state, driver = observation(0), Driver(spec)
    driver.recovery_stage = 3
    driver.recovering = True
    assert not driver.act(Goal("Score", (-6, 1)), state).score
    assert not driver.recovering


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
