import copy
import math

import numpy as np
import pytest

from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState
from frsbot.live.predict import Observations, predict
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.sim.wasm_env import WasmEnv

SPEC = next(s for s in load_specs() if s.team == 254)


def sample():
    state = GameState(32.95, Phase.TRANSITION, {a: HubState(True, .05) for a in ('blue', 'red')},
                      [RobotState(0, 254, 'blue', -6, 2, 0, 0, 0, 0, 72),
                       RobotState(1, 2910, 'red', 6, 2, 0, 1, -2, 0, 72)],
                      FuelState(np.array([[0, .1, 0], [1, 2, 3], [2, 1, 2]], np.float32),
                                np.array([0, 1, 2], np.uint8)), 0)
    state._first_inactive = 'blue'
    state._fuel_velocity = np.array([[10, 10, 10], [2, 3, -1], [10, 10, 10]], np.float32)
    return state


def test_prediction_cv_ballistics_hubs_and_immutability():
    state = sample()
    result = predict(state, Inputs(y=1), .1, SPEC)
    assert result.t == pytest.approx(33.05) and result.phase == Phase.SHIFT1
    assert not result.hubs['blue'].active and result.hubs['red'].active
    assert result.hubs['blue'].seconds_to_toggle == pytest.approx(24.95)
    assert result.robots[0].x > -6
    assert result.robots[1].x == pytest.approx(6.1)
    np.testing.assert_allclose(result.fuel.pos[1], [1.2, 2.3 - .04905, 2.9])
    np.testing.assert_array_equal(result.fuel.pos[[0, 2]], state.fuel.pos[[0, 2]])
    assert state.t == 32.95 and state.robots[0].x == -6
    assert predict(state, Inputs(), 1, SPEC).robots[1].x == pytest.approx(6.3)
    unchanged = predict(state, Inputs(), 0, SPEC)
    unchanged.fuel.pos[0] = 99
    assert state.fuel.pos[0, 0] == 0
    for invalid in (-1, math.inf, math.nan, 1.1):
        with pytest.raises(ValueError):
            predict(state, Inputs(), invalid, SPEC)


def test_empty_state_and_observation_history():
    view = dict(phase='auto', elapsed=1, clock=19, firstInactive='blue',
                hubs={a:dict(active=True) for a in ('blue','red')},
                robots=[], fuel=dict(count=1, pos=[0, 2, 0], where=[1]))
    observations = Observations()
    first = observations.state(view)
    later = copy.deepcopy(view)
    later['elapsed'] += .1
    later['fuel']['pos'] = [.2, 2.1, 0]
    second = observations.state(later)
    assert second._fuel_velocity[0, 0] == pytest.approx(2)
    assert second._fuel_velocity[0, 1] == pytest.approx(1 - .4905, abs=1e-6)
    assert np.all(first._fuel_velocity == 0)
    view['fuel'] = dict(count=0, pos=[], where=[])
    empty = observations.state(view)
    assert predict(empty, Inputs(), .1, SPEC).fuel.pos.shape == (0, 3)


def test_gap_to_transition_before_auto_winner_is_published():
    state = sample()
    state.t, state.phase, state._first_inactive = 22.95, Phase.GAP, None
    result = predict(state, Inputs(), .1, SPEC)
    assert result.phase == Phase.TRANSITION
    assert all(h.active and h.seconds_to_toggle == pytest.approx(9.95) for h in result.hubs.values())


@pytest.mark.parametrize('L', [.05, .1, .15])
@pytest.mark.parametrize('command', [Inputs(y=1, field_relative=True), Inputs(x=.5, y=.5, field_relative=True), Inputs(turn=.5, field_relative=True)])
def test_ego_prediction_against_held_wasm_rollout(L, command):
    if not (ENGINE_DIR / 'manifest.json').exists():
        pytest.skip('engine assets not fetched')
    with WasmEnv() as env:
        state = env.reset(254, start=dict(fx=2, fy=2, headingDeg=180), seed=7)
        state = env.step(Inputs(), ticks=15)
        expected = env.step(command, ticks=round(L * 60))
    result = predict(state, command, L, SPEC)
    a, b = result.robots[0], expected.robots[0]
    error = math.hypot(a.x - b.x, a.z - b.z)
    yaw_error = abs((a.a - b.a + math.pi) % (2 * math.pi) - math.pi)
    assert error < .12, (L, command, error)
    assert yaw_error < .15, (L, command, yaw_error)
