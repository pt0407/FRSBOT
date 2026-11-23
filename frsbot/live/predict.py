"""Short-horizon, non-mutating prediction of tapped observations (SI seconds)."""
import copy
import math

import numpy as np

from frsbot.core.field import is_active, phase_at, seconds_to_toggle
from frsbot.core.state import HubState, Phase
from frsbot.sim.drive import TICK_HZ, TURN_ACCEL, shape
from frsbot.sim.wasm_env import view_to_state


class Observations:
    def __init__(self):
        self.previous = None

    def state(self, view):
        state = view_to_state(view)
        state._first_inactive = view.get("firstInactive")
        state._fuel_velocity = np.zeros_like(state.fuel.pos)
        state._omega = 0.0
        previous = self.previous
        if previous is not None and 0 < state.t - previous.t <= .3:
            dt = state.t - previous.t
            if state.fuel.pos.shape == previous.fuel.pos.shape:
                same = state.fuel.where == previous.fuel.where
                state._fuel_velocity[same] = (state.fuel.pos[same] - previous.fuel.pos[same]) / dt
                state._fuel_velocity[same & (state.fuel.where == 1), 1] -= 9.81 * dt / 2
            if state.robots and previous.robots:
                a, b = state.robots[state.me], previous.robots[previous.me]
                if a.id == b.id:
                    state._omega = ((a.a - b.a + math.pi) % (2 * math.pi) - math.pi) / dt
        self.previous = state
        return state


def predict(state, last_inputs, L, spec):
    if not math.isfinite(L) or L < 0 or L > 1:
        raise ValueError("prediction horizon must be 0..1 seconds")
    result = copy.deepcopy(state)
    if not L:
        return result
    result.t += L
    result.phase = phase_at(result.t)
    first = getattr(state, "_first_inactive", None)
    if first is None and state.phase in (Phase.SHIFT1, Phase.SHIFT2, Phase.SHIFT3, Phase.SHIFT4):
        off = next(a for a, h in state.hubs.items() if not h.active)
        first = off if state.phase in (Phase.SHIFT1, Phase.SHIFT3) else ("red" if off == "blue" else "blue")
    result.hubs = {}
    for a in state.hubs:
        active = is_active(a, result.t, first)
        if first is None and 23 <= result.t < 33:
            # Auto's winner is not published until gap ends; 33 is the next possible toggle.
            toggle = 33 - result.t
        else:
            toggle = seconds_to_toggle(a, result.t, first)
        result.hubs[a] = HubState(active, toggle)
    for i, robot in enumerate(result.robots):
        if i != state.me:
            robot.x += robot.vx * min(L, .3)
            robot.z += robot.vz * min(L, .3)
            continue
        omega, remaining = getattr(state, "_omega", 0.0), L
        while remaining > 1e-9:
            dt = min(1 / TICK_HZ, remaining)
            enabled = phase_at(state.t + L - remaining) not in (Phase.COUNTDOWN, Phase.GAP, Phase.SETTLE, Phase.DONE)
            x, y, turn = (last_inputs.x, last_inputs.y, last_inputs.turn) if enabled else (0, 0, 0)
            norm = math.hypot(x, y)
            scale = shape(min(norm, 1)) / norm if norm else 0
            sign = 1 if robot.alliance == "blue" else -1
            vx, vz = sign * y * scale * spec.max_speed, sign * x * scale * spec.max_speed
            if not last_inputs.field_relative:
                vx, vz = vx * math.cos(robot.a) - vz * math.sin(robot.a), vx * math.sin(robot.a) + vz * math.cos(robot.a)
            dx, dz = vx - robot.vx, vz - robot.vz
            change = math.hypot(dx, dz)
            factor = min(1, spec.accel * dt / change) if change else 0
            robot.vx += dx * factor
            robot.vz += dz * factor
            target_w = -shape(turn) * spec.turn_rate
            omega += max(-TURN_ACCEL * dt, min(TURN_ACCEL * dt, target_w - omega))
            robot.a += omega * dt
            robot.x += robot.vx * dt
            robot.z += robot.vz * dt
            remaining -= dt
    flying = state.fuel.where == 1
    velocity = getattr(state, "_fuel_velocity", np.zeros_like(state.fuel.pos))
    result.fuel.pos[flying] += velocity[flying] * L
    result.fuel.pos[flying, 1] -= .5 * 9.81 * L * L
    result.fuel.pos[flying, 1] = np.maximum(.075, result.fuel.pos[flying, 1])
    return result
