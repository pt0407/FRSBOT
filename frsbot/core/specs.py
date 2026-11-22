"""Shared robot specifications extracted from the game bundle."""
import json
from dataclasses import dataclass
from pathlib import Path

SPECS_PATH = Path(__file__).resolve().parents[2] / "data" / "robot_feel.json"


@dataclass
class Feel:
    max_speed: float  # m/s
    turn_rate: float  # rad/s
    accel: float  # m/s^2
    hopper_capacity: int

    @classmethod
    def from_spec(cls, spec):
        if isinstance(spec, Feel):
            return cls(spec.max_speed, spec.turn_rate, spec.accel, spec.hopper_capacity)
        return cls(spec["maxSpeed"], spec["turnRate"], spec["accel"], spec["hopperCapacity"])


@dataclass
class RobotSpec(Feel):
    team: int
    robot_name: str
    drive_motor: str | None
    frame_length: float  # m
    frame_width: float  # m
    bumper_length: float  # m
    bumper_width: float  # m

    @classmethod
    def from_spec(cls, spec: dict):
        return cls(
            max_speed=spec["maxSpeed"], turn_rate=spec["turnRate"], accel=spec["accel"],
            hopper_capacity=spec["hopperCapacity"], team=spec["team"], robot_name=spec["robotName"],
            drive_motor=spec["driveMotor"], frame_length=spec["frameLength"], frame_width=spec["frameWidth"],
            bumper_length=spec["bumperLength"], bumper_width=spec["bumperWidth"],
        )


def load_records(path: Path = SPECS_PATH) -> list[dict]:
    return json.loads(path.read_text())


def load_specs(path: Path = SPECS_PATH) -> list[RobotSpec]:
    return [RobotSpec.from_spec(record) for record in load_records(path)]
