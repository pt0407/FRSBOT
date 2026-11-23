from types import SimpleNamespace

import numpy as np
import pytest

from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState
from frsbot.tools import measure_shooting


@pytest.mark.parametrize("mode", ["normal", "empty", "gated"])
def test_measure_startup_and_drain_or_report_failure(monkeypatch, mode):
    class FakeEnv:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def reset(self, *args, **kwargs):
            assert not kwargs["hp"] and not kwargs["unload"]
            self.state = GameState(0, Phase.AUTO, {a: HubState(True, 20) for a in ("blue", "red")},
                                   [RobotState(0, 254, "blue", -6, 1, 0, 0, 0, 0 if mode == "empty" else 40, 72)],
                                   FuelState(np.empty((0, 3)), np.empty(0)), 0)
            return self.state

        def step(self, inputs, ticks):
            self.state.t += ticks / 60
            if mode == "normal":
                self.state.robots[0].hopper -= 1
            return self.state

    monkeypatch.setattr(measure_shooting, "WasmEnv", FakeEnv)
    monkeypatch.setattr(measure_shooting, "ImAI", lambda spec: SimpleNamespace(act=lambda state: Inputs(score=True)))
    spec = next(s for s in load_specs() if s.team == 254)
    result = measure_shooting.measure(spec, 1)
    if mode == "normal":
        assert result["startup_seconds"] == pytest.approx(1 / 60)
        assert result["drain_rate"] == pytest.approx(60)
    else:
        assert "error" in result
