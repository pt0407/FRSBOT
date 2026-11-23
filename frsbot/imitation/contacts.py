"""Replay geometry proxies for contacts, not physics-engine contact logs."""
import math
from itertools import combinations

import numpy as np

from frsbot.analysis.matches import contact
from frsbot.bot.planner import FIELD_X, FIELD_Z
from frsbot.core.field import HUB_CENTRES

ROBOT = 1
FIELD = 2
FUEL = 4
CAUSES = {ROBOT: "Robot–robot", FIELD: "Wall/hub", FUEL: "FUEL beaching", 0: "No detected contact"}


def _field_contact(robot, half_length, half_width, c, s):
    if (abs(robot.x) + abs(c) * half_length + abs(s) * half_width >= FIELD_X - .04 or
            abs(robot.z) + abs(s) * half_length + abs(c) * half_width >= FIELD_Z - .04):
        return True
    for hx, hz, _ in HUB_CENTRES.values():
        dx, dz = hx - robot.x, hz - robot.z
        # Approximate the central hub footprint by its 1.22 m square.
        if (abs(dx) <= .65 + abs(c) * half_length + abs(s) * half_width and
                abs(dz) <= .65 + abs(s) * half_length + abs(c) * half_width and
                abs(dx * c + dz * s) <= half_length + .65 * (abs(c) + abs(s)) and
                abs(-dx * s + dz * c) <= half_width + .65 * (abs(c) + abs(s))):
            return True
    return False


def contact_flags(states, specs):
    flags = [0] * len(states[0].robots)
    peak_speed = [0.] * len(flags)
    for state in states:
        robots = state.robots
        for (i, a), (j, b) in combinations(enumerate(robots), 2):
            if math.hypot(a.x - b.x, a.z - b.z) < 1.8 and contact(a, b, specs[a.team], specs[b.team]):
                flags[i] |= ROBOT
                flags[j] |= ROBOT
        fuel = state.fuel
        ground = fuel.pos[(fuel.where == 0) & (fuel.pos[:, 1] < .18)]
        for i, robot in enumerate(robots):
            c, s = math.cos(robot.a), math.sin(robot.a)
            spec = specs[robot.team]
            hl, hw = spec.bumper_length / 2, spec.bumper_width / 2
            if _field_contact(robot, hl, hw, c, s):
                flags[i] |= FIELD
            speed = math.hypot(robot.vx, robot.vz)
            if not flags[i] & FUEL and speed < .25 and peak_speed[i] > 1:
                dx, dz = ground[:, 0] - robot.x, ground[:, 2] - robot.z
                if np.count_nonzero((np.abs(c * dx + s * dz) <= hl + .10) &
                                    (np.abs(-s * dx + c * dz) <= hw + .10)) >= 3:
                    flags[i] |= FUEL
            peak_speed[i] = max(peak_speed[i], speed)
    return flags
