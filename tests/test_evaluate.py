import json

import numpy as np
import pytest

from frsbot.core.specs import load_specs
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState
from frsbot.tools import build_viewer, evaluate


def test_evaluation_trace_viewer_and_tally(tmp_path, monkeypatch):
    class FakeEnv:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def reset(self, *args, **kwargs):
            assert kwargs["hp"] and kwargs["unload"]
            assert kwargs["auto"] == "none"
            self.state = GameState(0, Phase.AUTO, {a: HubState(True, 20) for a in ("blue", "red")},
                                   [RobotState(0, 254, "blue", -6, 1, 0, 0, 0, 0, 72)],
                                   FuelState(np.empty((0, 3), np.float32), np.empty(0, np.uint8)), 0)
            self.view = dict(robots=[dict(pos=[-6, 0, 1], quat=[0, 0, 0, 1], vel=[0, 0, 0],
                                         hopperFuel=0, drumRpm=0, state="IDLE", gate={})],
                             score={a: dict(total=3, autoFuel=3, teleFuel=0, inactiveFuel=0) for a in ("blue", "red")})
            return self.state

        def step(self, inputs, ticks):
            self.state.t += ticks / 60
            return self.state

    monkeypatch.setattr(evaluate, "WasmEnv", FakeEnv)
    spec = next(s for s in load_specs() if s.team == 254)
    output = tmp_path / "data" / "sim" / "imai"
    called_at = []
    act = evaluate.ImAI.act

    def observe_act(self, state):
        called_at.append(state.t)
        return act(self, state)

    monkeypatch.setattr(evaluate.ImAI, "act", observe_act)
    duration = .1
    result = evaluate.evaluate_run(spec, "blue", 1, output, duration=duration)
    assert result["score"] == result["fuel"] == 3
    assert result["low_hopper_score_seconds"] == 0
    assert result["auto"] == "none"
    assert sum(result["goal_seconds"].values()) == duration
    assert min(called_at) == 0
    path = output / "imai-254-blue-1.json"
    compact = build_viewer.compact(path)
    assert compact["kind"] == "sim" and compact["score"] == 3
    assert len(compact["samples"]) == 2 and compact["samples"][-1][-1] == 0
    monkeypatch.setattr(build_viewer, "ROOT", tmp_path)
    monkeypatch.setattr(build_viewer, "TEL", tmp_path / "telemetry")
    monkeypatch.setattr(build_viewer, "OUT", tmp_path / "viewer" / "replays.js")
    (output / "summary.json").write_text("{}")
    build_viewer.main()
    bundled = json.loads(build_viewer.OUT.read_text()[len("window.REPLAYS = "):-2])
    assert len(bundled) == 1 and bundled[0]["id"] == "imai-254-blue-1"


@pytest.mark.parametrize("duration,ticks", [(0, 3), (167, 3), (float("nan"), 3), (40, 0)])
def test_evaluation_rejects_invalid_limits(tmp_path, duration, ticks):
    with pytest.raises(ValueError, match="duration"):
        evaluate.evaluate_run(None, "blue", 1, tmp_path, duration=duration, ticks=ticks)
