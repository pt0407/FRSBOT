"""Decoder for REBUILT replay files (RBRP container, version 2).

Container:  magic 'RBRP' | u8 version | u32 headLen | head JSON | u32 nFrames | (u32 len | frame)*
Frames are delta-encoded Int16 state arrays (robots, hubs, doors) plus a FUEL section (see fuel.py).
The recorder keeps one frame every 2 ticks (30 Hz): a key frame (type 3) every 80 ticks, packed deltas (type 4) between.
"""
import json
import struct
import zlib
from dataclasses import dataclass, field

from frsbot.fuel import FuelState

MAGIC = b"RBRP"
QUAT_SCALE = 1 / 32767
POS_SCALE = 5e-4
VEL_SCALE = 1e-3


@dataclass
class Robot:
    id: int
    team: int
    alliance: str
    name: str
    at: int
    parts: int
    net: int


@dataclass
class Replay:
    meta: dict
    start: dict
    robots: list
    robot_offsets: list
    state_len: int
    frames: list = field(default_factory=list)
    events: list = field(default_factory=list)
    fuel: list = field(default_factory=list)  # per frame (pos int16 array, where array) when parse(fuel=True)
    radius: object = None


def gunzip_if_needed(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        return d.decompress(raw) + d.flush()
    return raw


def layout(start: dict):
    """Mirror of the game's state layout: robot blocks, then hub blocks, then 2 door values."""
    at = 0
    robots = []
    for r in start["robots"]:
        robots.append({"at": at, "parts": r["partsLen"], "net": r["netLen"]})
        at += 15 + r["partsLen"] + r["netLen"]
    hubs = []
    for knots in start["hubKnots"]:
        hubs.append({"at": at, "knots": knots})
        at += 4 + knots
    doors = at
    return robots, hubs, doors, at + 2


def parse(raw: bytes, fuel: bool = False) -> Replay:
    raw = gunzip_if_needed(raw)
    if raw[:4] != MAGIC:
        raise ValueError("not an RBRP replay")
    head_len = struct.unpack_from("<I", raw, 5)[0]
    head = json.loads(raw[9:9 + head_len])
    start = head["start"]
    offsets, _, _, state_len = layout(start)
    rep = Replay(meta=head["meta"], start=start, robots=start["robots"], robot_offsets=offsets, state_len=state_len)
    rep.events = head.get("events", [])
    rep.metas = head.get("metas", [])
    p = 9 + head_len
    n_frames = struct.unpack_from("<I", raw, p)[0]
    p += 4
    state = [0] * state_len
    fs = FuelState(start["fuelCount"])
    synced = False
    for _ in range(n_frames):
        flen = struct.unpack_from("<I", raw, p)[0]
        p += 4
        f = raw[p:p + flen]
        p += flen
        ftype = f[0]
        tick = struct.unpack_from("<i", f, 4)[0]
        if ftype == 3:
            state = [0] * state_len
            synced = True
        elif not synced or ftype not in (2, 4):
            continue
        q = apply_runs(f, state, ftype)
        if fuel:
            fs.apply(f, q, ftype)
            rep.fuel.append((fs.pos.astype("int16"), fs.where.copy()))
        rep.frames.append((tick, list(state)))
    rep.radius = fs.radius
    return rep


def apply_runs(f: bytes, state: list, ftype: int):
    n_runs = struct.unpack_from("<H", f, 20)[0]
    q = 22
    idx = 0
    for _ in range(n_runs):
        if ftype == 4:
            idx += f[q]
            count = f[q + 1]
            q += 2
        else:
            idx, count = struct.unpack_from("<HH", f, q)
            q += 4
        for k in range(count):
            state[idx + k] = struct.unpack_from("<h", f, q + 2 * k)[0]
        q += 2 * count
        if ftype == 4:
            idx += count
    return q


def robot_pose(state: list, rep: Replay, i: int) -> dict:
    o = rep.robot_offsets[i]["at"]
    pos = [state[o + k] * POS_SCALE for k in range(3)]
    vel = [state[o + 7 + k] * VEL_SCALE for k in range(3)]
    quat = [state[o + 3 + k] * QUAT_SCALE for k in range(4)]
    norm = sum(c * c for c in quat) ** 0.5 or 1.0
    quat = [c / norm for c in quat]
    return {
        "pos": [round(c, 4) for c in pos],
        "vel": [round(c, 3) for c in vel],
        "quat": [round(c, 4) for c in quat],
        "drumRpm": state[o + 12],
        "hopperFuel": state[o + 13],
    }


def telemetry(rep: Replay) -> dict:
    """One record per decoded frame: tick plus every robot's pose."""
    samples = []
    for tick, state in rep.frames:
        samples.append({"tick": tick, "robots": [robot_pose(state, rep, i) for i in range(len(rep.robots))]})
    return {
        "id": rep.meta.get("id"),
        "kind": rep.meta.get("kind"),
        "at": rep.meta.get("at"),
        "durationMs": rep.meta.get("durationMs"),
        "score": rep.meta.get("score"),
        "rules": rep.start.get("rules"),
        "robots": [{"id": r["id"], "team": r["team"], "alliance": r["alliance"], "name": r["name"]} for r in rep.robots],
        "events": rep.events,
        "samples": samples,
    }
