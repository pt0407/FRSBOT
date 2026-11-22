import json

import pytest

from frsbot.core.specs import Feel, load_records, load_specs
from frsbot.sim.drive import Feel as DriveFeel
from frsbot.sim.drive import load_feel


def test_specs_preserve_drive_feel_and_dimensions():
    records = load_records()
    specs = load_specs()
    assert len(specs) == len({spec.team for spec in specs}) == 20
    assert load_feel() == records
    assert DriveFeel is Feel
    for raw, spec in zip(records, specs):
        assert (spec.team, spec.robot_name, spec.drive_motor) == (raw["team"], raw["robotName"], raw["driveMotor"])
        assert Feel.from_spec(raw) == Feel.from_spec(spec)
        assert (spec.max_speed, spec.turn_rate, spec.accel, spec.hopper_capacity) == (
            raw["maxSpeed"], raw["turnRate"], raw["accel"], raw["hopperCapacity"])
        for attr, key in (("frame_length", "frameLength"), ("frame_width", "frameWidth"),
                          ("bumper_length", "bumperLength"), ("bumper_width", "bumperWidth")):
            assert getattr(spec, attr) == raw[key] > 0


def test_empty_specs(tmp_path):
    path = tmp_path / "specs.json"
    path.write_text("[]")
    assert load_specs(path) == []


def test_invalid_specs(tmp_path):
    path = tmp_path / "specs.json"
    with pytest.raises(FileNotFoundError):
        load_specs(path)
    path.write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        load_specs(path)
    path.write_text("[{}]")
    with pytest.raises(KeyError):
        load_specs(path)
