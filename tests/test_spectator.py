import base64
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from frsbot.core.state import Phase
from frsbot.replay.convert import convert
from frsbot.replay.rbrp import gunzip_if_needed, parse
from frsbot.replay.spectator import capture_replays, package
from frsbot.sim.wasm_env import find_node


FIXTURE = Path(__file__).parent / "fixtures" / "multiplayer_excerpt.rbrp.gz"


def excerpt():
    raw = gunzip_if_needed(FIXTURE.read_bytes())
    length = struct.unpack_from("<I", raw, 5)[0]
    head = json.loads(raw[9:9 + length])
    at = 13 + length
    frames = []
    while at < len(raw):
        size = struct.unpack_from("<I", raw, at)[0]
        frames.append(raw[at + 4:at + 4 + size])
        at += 4 + size
    return {"start": head["start"], "frames": frames, "metas": head["metas"],
            "events": head["events"], "source": head["meta"]["source"]}


def test_real_ranked_2v2_excerpt():
    rep = parse(FIXTURE.read_bytes(), fuel=True)
    states = convert(rep)
    assert len(states) == 3
    assert states[0].t == pytest.approx(101 + 34 / 60)
    assert states[0].phase == Phase.SHIFT3
    assert [(r.id, r.team, r.alliance) for r in states[0].robots] == [
        (24924, 4414, "blue"), (24921, 1323, "blue"),
        (24745, 1690, "red"), (23228, 4414, "red")]
    assert [r.hopper for r in states[0].robots] == [109, 12, 47, 104]
    assert not states[0].hubs["blue"].active and states[0].hubs["red"].active
    for state in states:
        assert state.fuel.pos.shape == (504, 3)
        assert np.isfinite(state.fuel.pos).all()
    assert rep.meta["partial"]


def test_jsonl_to_replay_preserves_wire_frames_and_events(tmp_path):
    record = excerpt()
    lines = [{"type": "capture", **record["source"]},
             {"type": "text", "message": {"t": "start", "match": record["start"]}},
             {"type": "text", "message": {"t": "meta", "meta": record["metas"][0][1]}},
             {"type": "text", "message": {"t": "ev", "events": [{"t": "intake", "robot": 24924}]}}]
    lines.extend({"type": "binary", "data": base64.b64encode(f).decode()} for f in record["frames"])
    path = tmp_path / "capture.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines))
    raw, = capture_replays(path)
    rep = parse(raw, fuel=True)
    assert len(convert(rep)) == 3
    assert rep.events == [[0, {"t": "intake", "robot": 24924}]]
    assert rep.meta["autoStartTick"] == 180


def test_countdown_origin_uses_clock_not_zero_elapsed():
    record = excerpt()
    frame = bytearray(record["frames"][0])
    struct.pack_into("<ifff", frame, 4, 1, 179 / 60, 20, 0)
    record["frames"] = [bytes(frame)]
    record["metas"][0][1]["phase"] = "countdown"
    rep = parse(package(record), fuel=True)
    state, = convert(rep)
    assert rep.meta["autoStartTick"] == 180
    assert state.t == pytest.approx(-179 / 60)
    assert state.phase == Phase.COUNTDOWN


def test_empty_and_invalid_capture(tmp_path):
    path = tmp_path / "capture.jsonl"
    path.write_text('{"type":"capture"}\n')
    assert list(capture_replays(path)) == []
    path.write_text('{"type":"binary","data":"Aw=="}\n')
    with pytest.raises(ValueError, match="before match start"):
        list(capture_replays(path))
    record = excerpt()
    record["frames"] = record["frames"][1:]
    with pytest.raises(ValueError, match="keyframe"):
        package(record)


def test_spectator_node_transport():
    result = subprocess.run([find_node(), "--test", str(Path(__file__).with_name("spectate.test.mjs"))],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
