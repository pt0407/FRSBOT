import json
import math
import subprocess
from pathlib import Path

import pytest

from frsbot.core.inputs import Inputs
from frsbot.core.state import Phase
from frsbot.live.bindings import AXES, BUTTONS, inputs_to_pad
from frsbot.live.bridge import state_summary
from frsbot.sim.wasm_env import find_node, view_to_state

ROOT = Path(__file__).resolve().parents[1]


def test_default_bindings():
    assert AXES == dict(x=(0, 1), y=(1, -1), turn=(2, -1))
    assert BUTTONS == dict(intake=6, score=7, outtake=4, block=5, stow=2, hood=1)
    axes, buttons = inputs_to_pad(Inputs(x=.01, y=.5, turn=-.7, climb=True, field_relative=True))
    assert axes == [.01, -.5, .7, 0]
    assert len(buttons) == 17 and not any(b["pressed"] for b in buttons)
    for name, index in BUTTONS.items():
        _, buttons = inputs_to_pad(Inputs(**{name: True}, climb=True))
        assert [i for i, b in enumerate(buttons) if b["pressed"]] == [index]
        assert buttons[index] == dict(pressed=True, value=1, touched=True)


def test_pad_neutral_invalid_and_limits():
    axes, buttons = inputs_to_pad(Inputs())
    assert axes == [0] * 4
    assert buttons == [dict(pressed=False, value=0, touched=False)] * 17
    axes, _ = inputs_to_pad(Inputs(x=math.nan, y=math.inf, turn=-2))
    assert axes == [0, 0, 1, 0]
    assert inputs_to_pad(Inputs(field_relative=False)) == inputs_to_pad(Inputs(field_relative=True))


def test_js_decoder_and_safety():
    result = subprocess.run([find_node(), str(ROOT / "tests" / "live.test.mjs")],
                            check=True, capture_output=True, text=True, timeout=15)
    decoded = json.loads(result.stdout)
    assert decoded["result"] == "live injection checks passed"
    state = view_to_state(decoded["view"])
    assert state.phase == Phase.AUTO and state.t == 1
    assert state.robots[0].hopper == 3
    assert state.fuel.pos.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert state.fuel.where.tolist() == [0, 1]
    assert view_to_state(decoded["empty"]).fuel.pos.shape == (0, 3)
    state.hubs["blue"].seconds_to_toggle = math.inf
    assert state_summary(state)["hubs"]["blue"]["seconds_to_toggle"] is None
    json.dumps(state_summary(state), allow_nan=False)


@pytest.mark.parametrize("args", [("--mode", "ranked"), ("--mode", "online"), ("--team", "123456")])
def test_cli_rejects_unsafe_or_unknown_setup(args):
    result = subprocess.run(["/usr/bin/python3", "-m", "frsbot.live.bridge", "--team", "254", *args],
                            cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert "error:" in result.stderr
