"""Backend-independent observations in meters and seconds from auto start."""
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Phase(str, Enum):
    COUNTDOWN = "countdown"
    AUTO = "auto"
    GAP = "gap"
    TRANSITION = "transition"
    SHIFT1 = "shift1"
    SHIFT2 = "shift2"
    SHIFT3 = "shift3"
    SHIFT4 = "shift4"
    ENDGAME = "endgame"
    SETTLE = "settle"
    DONE = "done"


@dataclass
class RobotState:
    id: int
    team: int
    alliance: str
    x: float
    z: float
    a: float
    vx: float
    vz: float
    hopper: int
    capacity: int


@dataclass
class FuelState:
    pos: np.ndarray  # (N, 3), float32 meters
    where: np.ndarray  # (N,), uint8; replay codes 0/1/2/3/4/6


@dataclass
class HubState:
    active: bool
    seconds_to_toggle: float


@dataclass
class GameState:
    t: float
    phase: Phase
    hubs: dict[str, HubState]
    robots: list[RobotState]
    fuel: FuelState
    me: int
