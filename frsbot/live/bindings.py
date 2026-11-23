"""Default standard-pad bindings; the camera controls field-relative mode."""
import math

from frsbot.core.inputs import Inputs

AXES = {"x": (0, 1), "y": (1, -1), "turn": (2, -1)}
BUTTONS = {"intake": 6, "score": 7, "outtake": 4, "block": 5, "stow": 2, "hood": 1}


def inputs_to_pad(inputs: Inputs) -> tuple[list[float], list[dict]]:
    axes = [0.0] * 4
    for name, (index, sign) in AXES.items():
        value = getattr(inputs, name)
        axes[index] = sign * max(-1.0, min(1.0, value)) if math.isfinite(value) else 0.0
    buttons = [dict(pressed=False, value=0, touched=False) for _ in range(17)]
    for name, index in BUTTONS.items():
        pressed = bool(getattr(inputs, name))
        buttons[index] = dict(pressed=pressed, value=int(pressed), touched=pressed)
    return axes, buttons
