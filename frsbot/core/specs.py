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
    brain_kind: str
    min_range: float | None  # m; None when the brain has no equivalent bound
    max_range: float | None
    heading_tolerance_deg: float | None
    max_shot_speed: float | None  # m/s
    hub_shot: dict[str, list[float]]
    fire_rate: float  # balls/s
    intake_rate: float  # balls/s
    hopper_storage_capacity: int
    total_capacity: int  # hopper plus feed lanes
    autos: list[str]
    brain_equivalents: dict

    @classmethod
    def from_spec(cls, spec: dict):
        brain = spec["brain"]
        equivalents = spec.get("brainEquivalents", {})
        shooting = {**equivalents, **brain}
        return cls(
            max_speed=spec["maxSpeed"], turn_rate=spec["turnRate"], accel=spec["accel"],
            hopper_capacity=spec["hopperCapacity"], team=spec["team"], robot_name=spec["robotName"],
            drive_motor=spec["driveMotor"], frame_length=spec["frameLength"], frame_width=spec["frameWidth"],
            bumper_length=spec["bumperLength"], bumper_width=spec["bumperWidth"],
            brain_kind=brain["kind"], min_range=shooting.get("minRange"), max_range=shooting.get("maxRange"),
            heading_tolerance_deg=shooting.get("headingToleranceDeg"), max_shot_speed=shooting.get("maxShotSpeed"),
            hub_shot=shooting.get("hubShot", {}), fire_rate=spec["fireRate"], intake_rate=spec["intakeRate"],
            hopper_storage_capacity=spec["hopperStorageCapacity"],
            total_capacity=spec["totalCapacity"], autos=brain["autos"], brain_equivalents=equivalents,
        )


def load_records(path: Path = SPECS_PATH) -> list[dict]:
    return json.loads(path.read_text())


def load_specs(path: Path = SPECS_PATH) -> list[RobotSpec]:
    return [RobotSpec.from_spec(record) for record in load_records(path)]
