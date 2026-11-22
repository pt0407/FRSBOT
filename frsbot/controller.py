"""Baseline driver: follow a pro's sampled route waypoint by waypoint in the sim.

Replay samples are ~1.3 s apart, so the route is a chain of straight chords, not a racing line.
Pure pursuit reversed on those chords, so this uses plain waypoint following with a 2 m arrival radius.

Usage: python -m frsbot.controller
Writes the simulated run to data/sim/ so it shows up in the viewer.
"""
import json
import math
from pathlib import Path

from frsbot.sim import DEADZONE, Feel, Robot, TICK_HZ, load_feel

ROOT = Path(__file__).resolve().parent.parent
TEL = ROOT / "data" / "telemetry"
SIM = ROOT / "data" / "sim"
ARRIVE = 2.0  # m, waypoint counts as reached inside this radius
CRUISE = 0.6  # fraction of max speed while turning hard


def unshape(f: float) -> float:
    """Inverse of the game's deadzone, so a wanted command of f reaches the sim as f."""
    return 0.0 if f <= 0 else DEADZONE + f * (1 - DEADZONE)


def pro_path(run: dict):
    """Waypoints (x, z), the pro's hopper count at each waypoint, and the start heading in drive convention."""
    pts = [(s["robots"][0]["pos"][0], s["robots"][0]["pos"][2]) for s in run["samples"]]
    hopper = [s["robots"][0]["hopperFuel"] for s in run["samples"]]
    q = run["samples"][0]["robots"][0]["quat"]
    return pts, hopper, math.pi - 2 * math.atan2(q[1], q[3])


def run(feel: Feel, waypoints, hopper_plan, start_heading, max_seconds=166.0):
    """Follow waypoints. At each one, match the pro's FUEL count: intake up to capacity, or score what's carried."""
    bot = Robot(feel, x=waypoints[0][0], z=waypoints[0][1], yaw=start_heading)
    trace, i, misses = [], 1, []
    for step in range(int(max_seconds * TICK_HZ)):
        if i >= len(waypoints):
            break
        tx, tz = waypoints[i]
        dx, dz = tx - bot.x, tz - bot.z
        d = math.hypot(dx, dz)
        if d < ARRIVE:
            want = hopper_plan[i]
            if want > bot.hopper:
                bot.intake(want - bot.hopper)
            elif want < bot.hopper:
                bot.score(bot.hopper - want)
            i += 1
            continue
        ang = math.atan2(math.sin(math.atan2(dz, dx) - bot.yaw), math.cos(math.atan2(dz, dx) - bot.yaw))
        want_w = max(-1.0, min(1.0, (2 * max(bot.v, 1.0) * math.sin(ang) / d) / feel.turn_rate))
        fwd = (CRUISE if abs(ang) < 0.5 else 0.0) * 1.0
        bot.step(unshape(fwd) if abs(ang) < 0.8 else 0.0, math.copysign(unshape(abs(want_w)), want_w))
        misses.append(d)
        trace.append((round(step / TICK_HZ, 3), bot.x, bot.z, bot.yaw, bot.v, bot.hopper))
    return trace, misses, i, bot


def to_telemetry(trace, run_id: str, team: int, capacity: int):
    """Write the sim trace in the same shape as data/telemetry, so the viewer can play it back."""
    samples = []
    for t, x, z, a, v, hopper in trace:
        yaw_quat = math.pi - a  # invert pro_path's heading convention
        samples.append({"tick": int(round(t * TICK_HZ)), "robots": [{
            "pos": [round(x, 4), 0.0, round(z, 4)],
            "vel": [round(v * math.cos(a), 3), 0.0, round(v * math.sin(a), 3)],
            "quat": [0.0, round(math.sin(yaw_quat / 2), 4), 0.0, round(math.cos(yaw_quat / 2), 4)],
            "drumRpm": 0, "hopperFuel": hopper}]})
    return {"id": run_id, "kind": "sim", "at": None, "durationMs": int(trace[-1][0] * 1000) if trace else 0,
            "score": None, "rules": None, "robots": [{"id": 0, "team": team, "alliance": "blue", "name": "ImAI baseline"}],
            "events": [], "samples": samples, "leaderboard": {"name": "ImAI baseline (sim)", "points": None}}


def main():
    spec = next(s for s in load_feel() if s["team"] == 254)  # the robot the pro run used (Overload)
    feel = Feel.from_spec(spec)
    pro = json.loads((TEL / "rutumgffbieaptw0.json").read_text())
    wps, plan, heading = pro_path(pro)
    trace, misses, reached, bot = run(feel, wps, plan, heading)
    SIM.mkdir(parents=True, exist_ok=True)
    out = to_telemetry(trace, "sim-baseline-254", 254, feel.hopper_capacity)
    (SIM / "sim-baseline-254.json").write_text(json.dumps(out, separators=(",", ":")))
    dist = sum(math.dist(trace[k][1:3], trace[k + 1][1:3]) for k in range(len(trace) - 1))
    done = reached >= len(wps)
    print(f"robot {spec['robotName'] or 'team 254'}  hopper capacity {feel.hopper_capacity}  pro peak {max(plan)}")
    print(f"waypoints reached {reached - 1}/{len(wps) - 1}  {'finished' if done else 'not finished'} "
          f"at {trace[-1][0]:.1f} s  distance {dist:.1f} m")
    print(f"max hopper {max(r[5] for r in trace)}  overflow {bot.overflow}  scored at end {bot.score()}")
    print(f"mean distance to current waypoint {sum(misses)/len(misses):.2f} m")


if __name__ == "__main__":
    main()
