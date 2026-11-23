import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from frsbot.bot.calibration import profile
from frsbot.bot.driver import Driver
from frsbot.bot.imai import ImAI
from frsbot.bot.planner import Goal
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.live.page_driver import PagePlanner, browser_source
from frsbot.live.predict import predict
from frsbot.sim.wasm_env import find_node, view_to_state

SPEC = next(s for s in load_specs() if s.team == 254)
ROOT = Path(__file__).resolve().parents[1]


def view(t):
    return dict(phase='auto', clock=20-t, elapsed=t, firstInactive='blue',
                hubs={a:dict(active=True) for a in ('blue','red')},
                robots=[dict(id=0, team=254, alliance='blue', local=True,
                             pos=[-6, 0, 1], quat=[0,1,0,0], vel=[0,0,0],
                             hopperFuel=30, hopperCapacity=72, gate=dict(open=True))],
                fuel=dict(count=0,pos=[],where=[]))


def test_browser_driver_port_parity_prediction_and_safety():
    driver, calls, expected = Driver(SPEC), [], []
    for i in range(100):
        raw = view(i / 10)
        goal = Goal(['Collect','Score','Stage'][i // 35 % 3], (-6, .5))
        calls.append(dict(view=raw, goal=asdict(goal)))
        command = driver.act(goal, view_to_state(raw))
        expected.append((command, driver.abandon, driver.recovering))
    result = subprocess.run([find_node(), str(ROOT/'tests/page_driver.test.mjs')],
                            input=json.dumps(dict(source=browser_source(), spec=asdict(SPEC),
                                                  profile=profile(254), calls=calls)),
                            text=True, capture_output=True, check=True, timeout=20)
    output = json.loads(result.stdout)
    for status, (command, abandon, recovering) in zip(output['results'], expected):
        for axis in ('x','y','turn'):
            assert status['inputs'][axis] == pytest.approx(getattr(command, axis), abs=1e-10)
        for key in ('intake','score','outtake','climb'):
            assert status['inputs'][key] == getattr(command, key)
        assert status['abandon'] == abandon and status['recovering'] == recovering
    state = view_to_state(calls[-1]['view'])
    predicted = predict(state, Inputs(x=.2,y=.7,turn=.3,field_relative=True), .1, SPEC)
    assert output['predicted']['t'] == pytest.approx(predicted.t)
    for key in ('x','z','a','vx','vz'):
        assert output['predicted']['robots'][0][key] == pytest.approx(getattr(predicted.robots[0], key))


def test_page_planner_uses_imai_collect_and_abandon_logic():
    planner, reference = PagePlanner(SPEC), ImAI(SPEC)
    state = view_to_state(view(1))
    state.robots[0].hopper = 0
    for bot in (planner.bot, reference):
        bot.collect_target, bot.collect_until, bot.collect_deadline = 20, 3, 9
    reference.act(state)
    goal = planner.plan(state, dict(abandon=False))
    assert goal == asdict(reference.goal)
    # A latched abandonment from the actual page driver reaches unchanged ImAI logic.
    state.robots[0].hopper = 3
    planner.bot.collect_target = 0
    planner.bot.goal = Goal('Score', (-6, 1))
    planner.bot.planner.goal = planner.bot.goal
    planner.bot.next_plan = 100
    planner.plan(state, dict(abandon=True))
    assert planner.bot.collect_target >= 10
