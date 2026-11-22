"""Recover controls and state-only goals into data/imitation/<run>.npz."""
import argparse
from bisect import bisect_left
import math
import multiprocessing as mp
from multiprocessing.util import Finalize
import os
from pathlib import Path

import numpy as np

from frsbot.core.field import ALLIANCES, HUB_CENTRES
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.imitation.labels import label_segments
from frsbot.replay.convert import convert
from frsbot.replay.rbrp import POS_SCALE, parse, robot_pose
from frsbot.scrape.harvest import DATA, EXCLUDE_NAMES
from frsbot.sim.wasm_env import WasmEnv

SEGMENT_TICKS = 120
RESIDUALS = ("pose_m", "yaw_rad", "velocity_m_s", "hopper_abs", "fuel_where", "fuel_match")
_env = None


def discover(solo, multiplayer):
    return [(kind, path) for kind, directory in (("solo", solo), ("multiplayer", multiplayer))
            for path in sorted(directory.rglob("*.rbrp*"))
            if path.name.endswith((".rbrp", ".rbrp.gz"))]


def excluded(replay, source):
    if source != "solo":
        return False
    robots = replay.robots + (replay.meta.get("robots") or [])
    return any(str(r.get("name", "")).strip() in EXCLUDE_NAMES for r in robots)


def windows(ticks, states, control_ticks):
    """Use recorded boundaries; never synthesize frames or cross a capture gap."""
    start = next((i for i, state in enumerate(states) if state.t >= 0), len(ticks))
    while start < len(ticks):
        end = bisect_left(ticks, ticks[start] + SEGMENT_TICKS, lo=start + 1)
        if end == len(ticks) or states[end].t > 163:
            break
        if ticks[end] - ticks[start] <= SEGMENT_TICKS + 2 and all(
            0 < ticks[i] - ticks[i - 1] <= 12 for i in range(start + 1, end + 1)
        ):
            indices = [start]
            for i in range(start + 1, end):
                if ticks[i] - ticks[indices[-1]] >= control_ticks:
                    indices.append(i)
            yield [*indices, end]
        start = end


def _fuel(state):
    return {"pos": state.fuel.pos.ravel().tolist(), "where": state.fuel.where.tolist()}


def _bodies(replay, ticks, k):
    previous = k - 1 if k else k
    following = k + 1 if k + 1 < len(ticks) else k
    dt = (ticks[following] - ticks[previous]) / 60
    result = []
    for i in range(len(replay.robots)):
        body = robot_pose(replay.frames[k][1], replay, i)
        orientations = [robot_pose(replay.frames[j][1], replay, i)["quat"] for j in (previous, following)]
        delta = 2 * (math.atan2(orientations[1][1], orientations[1][3]) -
                     math.atan2(orientations[0][1], orientations[0][3]))
        result.append(dict(pos=body["pos"], quat=body["quat"], vel=body["vel"],
                           hopper=body["hopperFuel"], omega=math.remainder(delta, 2 * math.pi) / dt if dt else 0))
    return result


def _button_hints(states, ticks, k, goals, specs):
    near = min(bisect_left(ticks, ticks[k] + 24, lo=k), len(ticks) - 1)
    future = min(bisect_left(ticks, ticks[k] + 90, lo=k), len(ticks) - 1)
    hints = []
    for i, (robot, goal) in enumerate(zip(states[k].robots, goals)):
        spec = specs[robot.team]
        hx, hz, _ = HUB_CENTRES[robot.alliance]
        in_range = math.hypot(robot.x - hx, robot.z - hz) <= (spec.max_range or max(spec.hub_shot["distance"])) + .5
        hopper = [state.robots[i].hopper for state in states[k:future + 1]]
        hints.append([goal.kind == "Collect" or states[near].robots[i].hopper > robot.hopper,
                      in_range and robot.hopper > 0 and min(hopper) < robot.hopper and goal.kind == "Score"])
    return hints


def make_segment(replay, states, ticks, indices, specs):
    start = indices[0]
    previous = start - 1 if start and 0 < ticks[start] - ticks[start - 1] <= 12 else start
    before, now = states[previous].fuel, states[start].fuel
    vel = (now.pos - before.pos) * (60 / (ticks[start] - ticks[previous])) if previous != start else np.zeros_like(now.pos)
    vel[before.where != now.where] = 0
    # Quantized positions can jump when a recycled ball re-enters the field.
    vel[np.linalg.norm(vel, axis=1) > 25] = 0
    goals = label_segments(states[start:indices[-1] + 1], specs)
    frames = [dict(tick=ticks[k], bodies=_bodies(replay, ticks, k), fuel=_fuel(states[k]),
                   buttons=_button_hints(states, ticks, k, goals, specs)) for k in indices]
    robots = [dict(id=r["id"], team=r["team"], alliance=r["alliance"],
                   start=r.get("start") or dict(fx=0, fy=0, headingDeg=0), speed=specs[r["team"]].max_speed,
                   turnRate=specs[r["team"]].turn_rate) for r in replay.robots]
    payload = dict(robots=robots, seed=replay.start["seed"],
                   initialFuel=dict(frames[0]["fuel"], vel=vel.ravel().tolist()), frames=frames)
    return goals, payload


def _run(segment):
    global _env
    if _env is None:
        _env = WasmEnv()
        _env._spawn()
        Finalize(_env, _env.close, exitpriority=0)
        _env._request("new", config=dict(team=254, recovery=True))
    try:
        return _env._request("recover", segment=segment)
    except RuntimeError as error:
        if not str(error).startswith("no recoverable candidate:"):
            raise
        return {"rejection": str(error)}


def write_run(path, source, states, ticks, selections, goals, results):
    start = np.array([k for segment in selections for k in segment[:-1]])
    end = np.array([k for segment in selections for k in segment[1:]])
    robot_states = [states[k].robots for k in start]
    residual = np.concatenate([r["residuals"] for r in results]).astype(np.float32)
    controls = np.concatenate([r["controls"] for r in results])
    packets = np.array([[list(Inputs(x, y, turn, bool(intake), bool(score), field_relative=True).encode())
                         for x, y, turn, intake, score in row] for row in controls], dtype=np.uint8)
    segment_lengths = [len(segment) - 1 for segment in selections]
    segment_id = np.repeat(np.arange(len(selections)), segment_lengths)
    flagged = (residual[:, :, 0] > .1) | (residual[:, :, 1] > .2) | (residual[:, :, 3] > 2) | (residual[:, :, 5] < .7)
    np.savez_compressed(
        path, version=1, source=source, segment=segment_id,
        tick=np.asarray(ticks)[start], ticks=np.asarray(ticks)[end] - np.asarray(ticks)[start],
        t=[states[k].t for k in start], phase=[states[k].phase.value for k in start],
        hub_active=[[states[k].hubs[a].active for a in ALLIANCES] for k in start],
        hub_toggle=[[states[k].hubs[a].seconds_to_toggle for a in ALLIANCES] for k in start],
        robot_id=[r.id for r in states[0].robots], robot_team=[r.team for r in states[0].robots],
        robot_alliance=[r.alliance for r in states[0].robots],
        pose=np.array([[(r.x, r.z, r.a, r.vx, r.vz) for r in row] for row in robot_states], dtype=np.float32),
        hopper=np.array([[r.hopper for r in row] for row in robot_states], dtype=np.int16),
        capacity=[r.capacity for r in states[0].robots],
        fuel_pos=np.array([states[k].fuel.pos for k in start], dtype=np.float32),
        fuel_where=np.array([states[k].fuel.where for k in start], dtype=np.uint8),
        inputs=packets, axes=controls[:, :, :3], residual=residual, residual_names=RESIDUALS,
        flagged=flagged, weight=np.where(flagged, .25, 1.).astype(np.float32),
        goal_kind=np.array([[g.kind for g in row] for row in goals])[segment_id],
        goal_point=np.array([[g.point for g in row] for row in goals], dtype=np.float32)[segment_id],
        initial_hopper=[r["initialHoppers"] for r in results],
    )


def recover_run(pool, path, source, destination, control_ticks, limit):
    replay = parse(path.read_bytes(), fuel=True)
    if excluded(replay, source):
        print(f"excluded {path.name}: driver 404", flush=True)
        return
    specs = {s.team: s for s in load_specs()}
    states = convert(replay, list(specs.values()))
    ticks = [tick for tick, _ in replay.frames]
    selections = list(windows(ticks, states, control_ticks))
    if limit and len(selections) > limit:
        selections = [selections[i] for i in np.linspace(0, len(selections) - 1, limit, dtype=int)]
    if not selections:
        print(f"skipped {path.name}: no complete 2 s segments", flush=True)
        return
    goals = []

    def jobs():
        for selection in selections:
            labels, segment = make_segment(replay, states, ticks, selection, specs)
            goals.append(labels)
            yield segment

    results = list(pool.imap(_run, jobs(), chunksize=1))
    valid = []
    for i, result in enumerate(results):
        if "rejection" in result:
            print(f"rejected {path.name} at tick {ticks[selections[i][0]]}: {result['rejection']}", flush=True)
        else:
            valid.append(i)
    if not valid:
        print(f"skipped {path.name}: all segments rejected", flush=True)
        return
    selections, goals, results = ([xs[i] for i in valid] for xs in (selections, goals, results))
    write_run(destination, source, states, ticks, selections, goals, results)
    print(f"{source} {path.name}: {len(selections)} segments, {sum(len(s) - 1 for s in selections)} frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo", type=Path, default=DATA / "raw")
    parser.add_argument("--multiplayer", type=Path, default=DATA / "raw_mp")
    parser.add_argument("--output", type=Path, default=DATA / "imitation")
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
    parser.add_argument("--control-ticks", type=int, default=2)
    parser.add_argument("--segments-per-replay", type=int, default=0, help="evenly spaced subset; 0 uses all segments")
    parser.add_argument("--replay", action="append", help="process only these filename stems; repeatable")
    args = parser.parse_args()
    if min(args.workers, args.control_ticks) < 1 or args.control_ticks > SEGMENT_TICKS or args.segments_per_replay < 0:
        parser.error("workers must be positive, control ticks must be in [1, 120], and segment limit must be non-negative")
    sources = discover(args.solo, args.multiplayer)
    sources = [(source, path, path.name.removesuffix(".gz").removesuffix(".rbrp")) for source, path in sources]
    sources = [(source, path, stem) for source, path, stem in sources if not args.replay or stem in args.replay]
    if not sources:
        parser.error("no matching RBRP files; harvest with --retain-raw or decode a spectator capture")
    args.output.mkdir(parents=True, exist_ok=True)
    pool = mp.Pool(args.workers)
    try:
        for source, path, stem in sources:
            recover_run(pool, path, source, args.output / f"{source}-{stem}.npz", args.control_ticks, args.segments_per_replay)
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()


if __name__ == "__main__":
    main()
