import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from frsbot.core.field import HUB_CENTRES
from frsbot.core.specs import load_records, load_specs
from frsbot.replay.rbrp import Replay
from frsbot.tools.validate_shooting import fraction, summarize_replay


def test_all_shooting_specs():
    for raw, spec in zip(load_records(), load_specs()):
        assert spec.brain_kind == raw["brain"]["kind"]
        assert spec.fire_rate == raw["fireRate"] > 0
        assert spec.intake_rate == raw["intakeRate"] > 0
        assert spec.hopper_storage_capacity == raw["hopperStorageCapacity"] > 0
        assert spec.total_capacity == spec.hopper_storage_capacity + raw["feedLanes"] * raw["feedPlaces"]
        assert spec.autos == raw["brain"]["autos"]
        shot = spec.hub_shot
        assert len(shot["distance"]) >= 2
        assert all(b > a for a, b in zip(shot["distance"], shot["distance"][1:]))
        assert all(len(values) == len(shot["distance"]) for values in shot.values())
        if spec.brain_kind == "generic":
            assert 0 < spec.min_range < spec.max_range
            assert spec.heading_tolerance_deg == raw["brain"]["headingToleranceDeg"] > 0
            assert spec.max_shot_speed == raw["brain"]["maxShotSpeed"] > 0
            assert spec.hub_shot == raw["brain"]["hubShot"]
        else:
            assert spec.brain_kind == "reblitz"
            assert spec.min_range == 1.5
            assert spec.heading_tolerance_deg == 4
            assert spec.max_range is None and spec.max_shot_speed is None
            assert spec.brain_equivalents["scoreSpeedTol"] == .15
            assert spec.brain_equivalents["flightTime"]["seconds"] == [.5, .83, 1.1]


def test_bundle_extraction_matches_records():
    bundle = Path.home() / "game" / "2c3zgga2f0b2g.js"
    if not shutil.which("node") or not bundle.exists():
        pytest.skip("requires Node and downloaded game bundle")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["node", str(root / "frsbot/tools/extract_robot_specs.mjs"), "--check"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "verified 20 robot shooting/intake configurations" in result.stdout


def test_extraction_empty_and_invalid_inputs():
    if not shutil.which("node"):
        pytest.skip("requires Node")
    script = (Path(__file__).resolve().parents[1] / "frsbot/tools/extract_robot_specs.mjs").as_uri()
    result = subprocess.run(["node", "--input-type=module", "-e", f"""
        import assert from 'node:assert/strict';
        import {{ enrich, loadBundle }} from '{script}';
        assert.deepEqual(enrich([], {{ robots: new Map() }}), []);
        assert.throws(() => enrich([{{team: 1}}], {{robots: new Map()}}), /no bundle robot/);
        assert.throws(() => enrich([{{team: 1, robotName: 'one'}}],
          {{robots: new Map([[1, {{robotName: 'two'}}]])}}), assert.AssertionError);
        assert.throws(() => loadBundle('nonexistent-bundle.js'), /ENOENT/);
    """], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def replay(samples, alliance="blue"):
    frames = []
    cx, cz, _ = HUB_CENTRES[alliance]
    for tick, hopper, distance, speed in samples:
        raw = [0] * 15
        raw[0], raw[2] = round((cx + distance) / .0005), round(cz / .0005)
        raw[7], raw[13] = round(speed / .001), hopper
        frames.append((tick, raw))
    return Replay({}, {}, [{"team": 254, "alliance": alliance}], [{"at": 0}], 15, frames=frames)


@pytest.mark.parametrize("alliance", ["blue", "red"])
def test_envelope_inclusive_bounds_and_all_drops(alliance):
    spec = dataclasses.replace(next(s for s in load_specs() if s.team == 254),
                               min_range=1, max_range=5, max_shot_speed=1)
    rep = replay([(0, 10, 2, 0), (2, 9, 1, 1), (4, 7, 5, .5), (6, 8, 2, 0),
                  (8, 7, 5.1, 0), (10, 6, 2, 1.01)], alliance)
    stats = summarize_replay(rep, {254: spec})[254]
    assert stats["runs"] == 1
    assert stats["drop_frames"] == 4
    assert stats["net_balls"] == 5
    assert stats["range_frames"] == stats["speed_frames"] == 3
    assert stats["inside_frames"] == 2
    assert stats["rate_seconds"] == stats["rate_bursts"] == 0
    assert fraction(stats, "inside_frames", True) == "2/4 (50.0%)"


def test_rate_includes_intershot_gaps_and_excludes_first_drop():
    spec = next(s for s in load_specs() if s.team == 254)
    rep = replay([(i * 6, 100 - i, 2, 0) for i in range(21)])
    stats = summarize_replay(rep, {254: spec})[254]
    assert stats["drop_frames"] == 20
    assert stats["net_balls"] == 20
    assert stats["rate_balls"] == 19
    assert stats["rate_seconds"] == pytest.approx(1.9)
    assert stats["rate_balls"] / stats["rate_seconds"] == pytest.approx(10)
    assert stats["rate_bursts"] == 1


def test_rate_splits_long_pauses():
    spec = next(s for s in load_specs() if s.team == 254)
    rep = replay([(0, 20, 2, 0), (2, 19, 2, 0), (4, 18, 2, 0), (120, 17, 2, 0)])
    assert summarize_replay(rep, {254: spec})[254]["rate_bursts"] == 0


def test_empty_no_drops_missing_bounds_and_invalid_replay():
    spec = next(s for s in load_specs() if s.team == 254)
    assert summarize_replay(replay([]), {}) == {}
    stats = summarize_replay(replay([(0, 0, 2, 0), (2, 1, 2, 0)]), {254: spec})[254]
    assert stats["drop_frames"] == stats["net_balls"] == 0
    assert fraction(stats, "inside_frames", True) == "N/A"
    spec = dataclasses.replace(spec, min_range=None, max_range=None, max_shot_speed=None)
    stats = summarize_replay(replay([(0, 2, 2, 0), (2, 1, 2, 0)]), {254: spec})[254]
    assert stats["drop_frames"] == 1
    assert fraction(stats, "inside_frames", False) == "N/A"
    with pytest.raises(KeyError):
        summarize_replay(replay([(0, 0, 2, 0)]), {})
    with pytest.raises(ValueError, match="ticks must increase"):
        summarize_replay(replay([(2, 0, 2, 0), (2, 0, 2, 0)]), {254: spec})
