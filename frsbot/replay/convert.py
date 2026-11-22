"""Convert decoded replay frames to backend-independent observations."""
import math

import numpy as np

from frsbot.core.field import ALLIANCES, is_active, phase_at, seconds_to_toggle
from frsbot.core.specs import RobotSpec, load_specs
from frsbot.core.state import FuelState, GameState, HubState, RobotState
from frsbot.replay.rbrp import POS_SCALE, VEL_SCALE, Replay


def auto_start_seconds(metas: list) -> float:
    for frame_index, meta in metas:
        if meta["phase"] == "auto":
            return frame_index / 30
    raise ValueError("replay metadata must contain auto start")


def convert(rep: Replay, specs: list[RobotSpec] | None = None) -> list[GameState]:
    if not rep.frames:
        return []
    if len(rep.fuel) != len(rep.frames):
        raise ValueError("conversion requires parse(fuel=True) with FUEL for every frame")
    metas = getattr(rep, "metas", [])
    origin = rep.frames[0][0] / 60 + auto_start_seconds(metas)
    first_inactive = next((m["firstInactive"] for _, m in metas if m.get("firstInactive") is not None), None)
    by_team = {spec.team: spec for spec in (load_specs() if specs is None else specs)}
    states = []
    for (tick, raw), (pos, where) in zip(rep.frames, rep.fuel):
        t = tick / 60 - origin
        robots = []
        for robot, offset in zip(rep.robots, rep.robot_offsets):
            o = offset["at"]
            robots.append(RobotState(
                id=robot["id"], team=robot["team"], alliance=robot["alliance"],
                x=raw[o] * POS_SCALE, z=raw[o + 2] * POS_SCALE,
                a=math.pi - 2 * math.atan2(raw[o + 4], raw[o + 6]),
                vx=raw[o + 7] * VEL_SCALE, vz=raw[o + 9] * VEL_SCALE,
                hopper=raw[o + 13], capacity=by_team[robot["team"]].hopper_capacity,
            ))
        states.append(GameState(
            t=t, phase=phase_at(t),
            hubs={a: HubState(is_active(a, t, first_inactive), seconds_to_toggle(a, t, first_inactive))
                  for a in ALLIANCES},
            robots=robots, fuel=FuelState(pos.astype(np.float32) * np.float32(POS_SCALE),
                                         where.astype(np.uint8, copy=True)), me=0,
        ))
    return states
