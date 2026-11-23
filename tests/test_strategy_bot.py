import math
from dataclasses import replace

import numpy as np
import pytest

from frsbot.bot.imai import ImAI
from frsbot.bot.planner import FIELD_X, FIELD_Z, Goal, Planner
from frsbot.core.field import HUB_CENTRES, is_active, phase_at, seconds_to_toggle
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.core.state import FuelState, GameState, HubState, RobotState
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.sim.vec_env import RobotSlot
from frsbot.sim.wasm_env import WasmEnv
from frsbot.strategy import fouls
from frsbot.strategy.bot import NEW_GOAL_KINDS, StrategicImAI
from frsbot.strategy.driver import LANE_BOXES, StrategicDriver, half_diagonal, inside
from frsbot.strategy.roles import Role


@pytest.fixture
def specs():
    return {s.team: s for s in load_specs()}


def state(t=60., robots=None, fuel=()):
    robots = robots or [RobotState(0, 254, "blue", -6., 1., 0., 0., 0., 8, 72),
                        RobotState(1, 2910, "red", 6., 1., math.pi, 0., 0., 30, 55)]
    return GameState(t, phase_at(t), {a: HubState(is_active(a, t, "blue"), seconds_to_toggle(a, t, "blue"))
                                    for a in ("blue", "red")}, robots,
                     FuelState(np.array(fuel, np.float32).reshape(-1, 3), np.zeros(len(fuel), np.uint8)), 0)


def force(bot, role):
    bot.layer.role = lambda observation: role


def team_state(*args, **kwargs):
    s = state(*args, **kwargs)
    s.robots.extend([replace(r, id=r.id + 2, z=-3.) for r in s.robots])
    return s


@pytest.mark.parametrize("alliance", ["blue", "red"])
@pytest.mark.parametrize("opponents", [0, 1, 3])
def test_no_teammates_identical_to_imai(specs, monkeypatch, alliance, opponents):
    original, strategic = ImAI(specs[254]), StrategicImAI(specs[254])

    def unexpected(*args):
        pytest.fail("strategy must not run without teammates")

    monkeypatch.setattr(strategic.layer, "observe", unexpected)
    monkeypatch.setattr(strategic.driver, "guard", unexpected)
    monkeypatch.setattr(strategic.driver, "act", unexpected)
    for t in (-1., 0., .1, 1., 10., 25., 40., 60., 100., 140., 160., 163., 166.):
        s = state(t)
        me = replace(s.robots[0], alliance=alliance)
        other = "red" if alliance == "blue" else "blue"
        s.robots = [replace(s.robots[1], id=i + 1, alliance=other, x=me.x+.5) for i in range(opponents)] + [me]
        s.me = opponents
        s.fuel = state(fuel=[[-5., .075, 2.], [-4.8, .075, 2.]]).fuel
        assert strategic.act(s) == original.act(s)
        assert strategic.goal == original.goal


@pytest.mark.parametrize("size", [2, 3])
def test_teammates_keep_strategy_enabled(specs, monkeypatch, size):
    bot = StrategicImAI(specs[254])
    s = team_state()
    if size == 3:
        s.robots.extend([replace(r, id=r.id + 4, z=3.) for r in s.robots[:2]])
    observed = []
    monkeypatch.setattr(bot.layer, "observe", lambda state: observed.append(state))
    force(bot, Role("Defender", 1))
    bot.act(s)
    assert observed == [s]
    assert bot.goal.kind == "Defend"


def test_default_off_planner_deconfliction(specs):
    planner = Planner(specs[254])
    s = state(fuel=[[-5., .075, 2.]] * 12 + [[-5., .075, -2.]] * 12)
    original = planner.clusters(s)
    occupied = original[0][1]
    planner.avoid_points = [occupied]
    assert all(math.dist(p, occupied) >= .8 for _, p in planner.clusters(s))
    planner.avoid_points = [point for _, point in original]
    assert planner.clusters(s) == original
    spot = planner.score_spot(s)
    planner.avoid_slots = [spot]
    assert math.dist(planner.score_spot(s), spot) >= 1.5


def test_scorer_replaces_cached_occupied_slot_and_evades_away(specs):
    bot = StrategicImAI(specs[254])
    force(bot, Role("Scorer"))
    s = team_state()
    s.robots[0].hopper = 70
    spot = bot.scorer.planner.score_spot(s)
    bot.scorer.goal = bot.scorer.planner.goal = Goal("Score", spot)
    bot.layer.teammate_slots = lambda observation: [spot]
    bot.act(s)
    assert math.dist(bot.goal.point, spot) >= 1.5
    assert bot.scorer.planner.goal == bot.goal
    bot.layer.teammate_slots = lambda observation: []
    bot.layer.defended_by = lambda observation: 1
    bot.layer.evasion = lambda observation: "alternate_slot"
    s = team_state(60.1)
    s.robots[0].hopper = 70
    bot.act(s)
    assert abs(bot.scorer.planner.score_offset) == math.pi / 6
    assert bot.goal.point == bot.scorer.planner.score_spot(s)


def test_defend_screen_points_bounds_hubs_lanes_and_no_score(specs):
    bot = StrategicImAI(specs[254])
    force(bot, Role("Defender", 1))
    s = team_state()
    for x in np.linspace(-8, 8, 9):
        for z in np.linspace(-3.9, 3.9, 7):
            for hopper in (0, 10):
                s.robots[1] = replace(s.robots[1], x=float(x), z=float(z), hopper=hopper)
                point = bot.screen_point(s, s.robots[1])
                pad = half_diagonal(specs[254])
                assert abs(point[0]) <= FIELD_X - pad and abs(point[1]) <= FIELD_Z - pad
                assert not any(inside(point, box, pad) for box in LANE_BOXES)
                assert all(not inside(point, (cx-.597, cx+.597, cz-.597, cz+.597), pad)
                           for cx, cz, _ in HUB_CENTRES.values())
    assert not bot.act(team_state()).score
    assert bot.goal.kind == "Defend"
    assert NEW_GOAL_KINDS == ("Defend", "Contest", "Feed")


def test_pin_guard_retreat_and_clear_interval(specs):
    bot = StrategicImAI(specs[254])
    force(bot, Role("Defender", 1))
    for t in (60., 60.5, 61.):
        s = team_state(t)
        s.robots[0].x = 5.4
        bot.act(s)
        assert (bot.driver.retreat_target is not None) == (t >= 61)
    point = bot.driver.retreat_point
    assert math.dist(point, (s.robots[1].x, s.robots[1].z)) > fouls.PIN_CLEAR_DISTANCE_METERS + .1
    for t in (61.1, 62.7, 62.9):
        s = team_state(t)
        s.robots[0].x = 1.
        bot.act(s)
        assert (bot.driver.retreat_target is not None) == (t < 62.9)


@pytest.mark.parametrize("role", [Role("Defender", 1), Role("Contester"), Role("Feeder")])
def test_no_defend_or_contest_in_auto(specs, role):
    bot = StrategicImAI(specs[254])
    force(bot, role)
    bot.act(team_state(10., fuel=[[0., .075, 0.]]))
    assert bot.goal.kind not in ("Defend", "Contest")


def test_auto_contact_guard_and_target_exemption(specs):
    driver = StrategicDriver(specs[254], specs)
    s = state(10.)
    s.robots[0].x, s.robots[1].x = 0., 1.
    inputs = driver.guard(Inputs(y=1., field_relative=True), Goal("Score", (6., 1.)), s)
    assert driver.velocity(inputs, s.robots[0])[0] < 0
    driver = StrategicDriver(specs[254], specs)
    s.t, s.phase = 60., phase_at(60.)
    original = Inputs(y=1., field_relative=True)
    assert driver.guard(original, Goal("Defend", (6., 1.)), s, s.robots[1]) == original


@pytest.mark.parametrize("kind", ["Contest", "Feed"])
@pytest.mark.parametrize("alliance", ["blue", "red"])
def test_g408_near_either_hub(specs, kind, alliance):
    driver = StrategicDriver(specs[254], specs)
    s = state(fuel=[])
    cx, cz = HUB_CENTRES[alliance][:2]
    s.robots[0].x, s.robots[0].z = cx + .8, cz
    guarded = driver.guard(Inputs(intake=True, field_relative=True), Goal(kind, (0., 0.)), s)
    assert not guarded.intake
    s.robots[0].x, s.robots[0].z = -6., 2.
    assert driver.guard(Inputs(intake=True), Goal(kind, (0., 0.)), s).intake


def test_endgame_tower_falls_back_to_scorer(specs):
    bot = StrategicImAI(specs[254])
    force(bot, Role("Defender", 1))
    s = team_state(140.)
    s.robots[1].x, s.robots[1].z = fouls.TOWER_CENTRES["red"]
    bot.act(s)
    assert bot.goal.kind != "Defend"


def test_all_kind_avoidance_and_paired_lane(specs):
    driver = StrategicDriver(specs[254], specs)
    s = state()
    me = s.robots[0]
    s.robots.append(replace(me, id=2, x=me.x+1.))
    inputs = driver.guard(Inputs(y=1., field_relative=True), Goal("Score", (-5., 1.)), s)
    assert inputs.x != 0.
    # Occupying one bump prevents driving into the opposite bump at this barrier.
    me.x, me.z = -5., -1.8
    s.robots[2].x, s.robots[2].z = -3.6, 1.8
    inputs = driver.guard(Inputs(y=1., field_relative=True), Goal("Contest", (0., -1.8)), s)
    future = np.array([me.x, me.z]) + .3 * driver.velocity(inputs, me)
    assert not inside(future, fouls.BUMP_BOXES[0], half_diagonal(specs[254]))


def test_contest_target_feed_release_and_empty_fallback(specs):
    bot = StrategicImAI(specs[254])
    force(bot, Role("Contester"))
    s = team_state(fuel=[[0., .075, 1.]] * 4 + [[2., .075, -1.]] * 4)
    s.robots[1].x, s.robots[1].z, s.robots[1].vx = 1., -1., 1.
    bot.act(s)
    assert bot.goal == Goal("Contest", (2., -1.))
    bot = StrategicImAI(specs[254])
    force(bot, Role("Feeder"))
    s = team_state(60.1)
    s.robots[0].hopper = 60
    bot.act(s)
    assert bot.goal.kind == "Feed"
    s = team_state(60.2)
    s.robots[0].hopper = 60
    s.robots[0].x, s.robots[0].z = bot.goal.point
    assert bot.act(s).score
    s = team_state(60.3)
    s.robots[0].hopper = 0
    bot.act(s)
    assert bot.goal.kind not in ("Feed", "Contest")
    driver = StrategicDriver(specs[254], specs)
    with pytest.raises(ValueError, match="target"):
        driver.act(Goal("Defend", (0., 0.)), state())


engine = pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").is_file(), reason="engine assets absent")


@engine
@pytest.mark.parametrize("team", [254, 2910])
def test_native_pass_lands_in_own_zone(team, capsys):
    with WasmEnv() as env:
        s = env.reset(robots=[RobotSlot(team=team, alliance="blue", start=dict(fx=2, fy=1.8, headingDeg=0))],
                           seed=7, hp=False, unload=False)
        carried = np.flatnonzero(s.fuel.where == 2)
        assert len(carried) == 8
        for _ in range(300):
            if s.robots[0].x > -2.:
                break
            s = env.step({0: Inputs(y=.7, field_relative=True)}, 3).views[0]
        s = env.step({}, 60).views[0]
        assert -4.2418 < s.robots[0].x < 0
        for _ in range(180):
            env.step({0: Inputs(score=True, field_relative=True)}, 3)
        s = env.step({}, 300).views[0]
        landed = sum(s.fuel.where[i] == 0 and s.fuel.pos[i, 0] < -4.2418 for i in carried)
        with capsys.disabled():
            print(f"\nNative pass team={team}: released={sum(s.fuel.where[i] != 2 for i in carried)}/8 landed_own_zone={landed}/8")
        assert landed >= 6


@engine
def test_defender_engine_smoke(specs, capsys):
    results = []
    for defended in (False, True):
        with WasmEnv() as env:
            env.reset(robots=[RobotSlot(team=254, alliance="blue", start=dict(fx=1.5, fy=2, headingDeg=0)),
                       RobotSlot(team=2910, alliance="red", start=dict(fx=15, fy=2, headingDeg=0)),
                       RobotSlot(team=254, alliance="blue", start=dict(fx=1, fy=7, headingDeg=0)),
                       RobotSlot(team=2910, alliance="red", start=dict(fx=15, fy=7, headingDeg=0))], seed=7)
            states = env.step({}, 1500).views
            bot, opponent = StrategicImAI(specs[254]), ImAI(specs[2910])
            force(bot, Role("Defender", 1))
            before = env.observation.driver[1]["scored"]
            minimum_distance = math.inf
            for _ in range(800):
                inputs = {1: opponent.act(states[1])}
                if defended:
                    inputs[0] = bot.act(states[0])
                states = env.step(inputs, 3).views
                a, b = states[0].robots[:2]
                minimum_distance = min(minimum_distance, math.hypot(a.x-b.x, a.z-b.z))
            after = env.observation.driver[1]["scored"]
            fouls_by_me = env.observation.driver[0]["fouls"]
            results.append((after - before, fouls_by_me, minimum_distance))
            assert fouls_by_me == 0
    with capsys.disabled():
        print(f"\nDefense smoke seed=7 t=25..65: baseline={results[0][0]} FUEL ({results[0][0]/40:.3f}/s), "
              f"defended={results[1][0]} FUEL ({results[1][0]/40:.3f}/s), fouls_by_defender={results[1][1]}, "
              f"minimum_distance={results[1][2]:.3f}m, lower={results[1][0] < results[0][0]}")
    assert all(count >= 0 for count, _, _ in results)
