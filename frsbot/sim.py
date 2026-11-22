"""Offline robot drive model, ported from the game's drive code.

Stick input -> target speed and turn rate (per-robot "feel" values from the game bundle),
then the robot moves toward the target no faster than its acceleration limit.

Assumptions still to verify against replays (see validate.py):
  - heading: forward = (cos a, sin a) in (x, z), where a = pi - quat_yaw (checked against replay velocities,
    median alignment 0.79; robots slip, so it isn't exact)
  - turn acceleration limit (not in the bundle we read; set to TURN_ACCEL)
  - tick rate of 60 Hz
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEADZONE = 0.12  # from the game's drive input shaping
TICK_HZ = 60.0
TURN_ACCEL = 10.0  # rad/s^2, assumed

FEEL_PATH = Path(__file__).resolve().parent.parent / "data" / "robot_feel.json"


def shape(stick: float) -> float:
    """Deadzone plus rescale, as the game does to stick input."""
    if abs(stick) < DEADZONE:
        return 0.0
    return math.copysign((abs(stick) - DEADZONE) / (1 - DEADZONE), stick)


def load_feel():
    return json.loads(FEEL_PATH.read_text())


@dataclass
class Feel:
    max_speed: float  # m/s
    turn_rate: float  # rad/s
    accel: float  # m/s^2
    hopper_capacity: int  # FUEL the robot can carry

    @classmethod
    def from_spec(cls, spec: dict):
        return cls(spec["maxSpeed"], spec["turnRate"], spec["accel"], spec["hopperCapacity"])


class Robot:
    def __init__(self, feel: Feel, x=0.0, z=0.0, yaw=0.0):
        self.feel = feel
        self.x, self.z, self.yaw = x, z, yaw
        self.v = 0.0  # forward speed along heading
        self.omega = 0.0  # yaw rate
        self.hopper = 0  # FUEL carried
        self.overflow = 0  # FUEL that couldn't fit (intake beyond capacity)

    def intake(self, n: int):
        take = min(n, self.feel.hopper_capacity - self.hopper)
        self.hopper += take
        self.overflow += n - take
        return take

    def score(self, n: Optional[int] = None):
        n = self.hopper if n is None else min(n, self.hopper)
        self.hopper -= n
        return n

    def step(self, forward: float, turn: float, dt: float = 1 / TICK_HZ):
        target_v = shape(forward) * self.feel.max_speed
        target_w = shape(turn) * self.feel.turn_rate
        dv = max(-self.feel.accel * dt, min(self.feel.accel * dt, target_v - self.v))
        dw = max(-TURN_ACCEL * dt, min(TURN_ACCEL * dt, target_w - self.omega))
        self.v += dv
        self.omega += dw
        self.yaw += self.omega * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.z += self.v * math.sin(self.yaw) * dt
        return self.x, self.z, self.yaw
