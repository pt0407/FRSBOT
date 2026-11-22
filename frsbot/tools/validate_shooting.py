"""Diagnose shooting envelopes using net hopper drops, not inferred control inputs.

Rates use bursts with <=0.5 s between drops and >=1 s from first to last drop.
The first drop is excluded from the numerator to avoid an endpoint/30 Hz bias.
Concurrent intake masks shots; outtakes and passes can also produce hopper drops.
"""
import math

from frsbot.core.field import HUB_CENTRES
from frsbot.core.specs import RobotSpec, load_specs
from frsbot.replay.rbrp import POS_SCALE, VEL_SCALE, Replay, parse
from frsbot.tools.validate_schedule import RAW


def empty_stats():
    return dict(runs=0, drop_frames=0, net_balls=0, range_frames=0, speed_frames=0,
                inside_frames=0, rate_balls=0, rate_seconds=0.0, rate_bursts=0)


def add_burst(stats: dict, drops: list[tuple[float, int]]):
    if len(drops) < 2:
        return
    duration = drops[-1][0] - drops[0][0]
    if duration >= 1 - 1e-9:
        stats["rate_balls"] += sum(n for _, n in drops[1:])
        stats["rate_seconds"] += duration
        stats["rate_bursts"] += 1


def summarize_replay(rep: Replay, specs: dict[int, RobotSpec]) -> dict[int, dict]:
    if not rep.frames:
        return {}
    if any(b[0] <= a[0] for a, b in zip(rep.frames, rep.frames[1:])):
        raise ValueError("replay frame ticks must increase")
    result = {}
    for robot, offset in zip(rep.robots, rep.robot_offsets):
        spec = specs[robot["team"]]
        cx, cz, _ = HUB_CENTRES[robot["alliance"]]
        stats = result.setdefault(spec.team, empty_stats())
        stats["runs"] = 1
        o = offset["at"]
        drops = []
        previous = rep.frames[0][1][o + 13]
        for tick, raw in rep.frames[1:]:
            count = previous - raw[o + 13]
            previous = raw[o + 13]
            if count <= 0:
                continue
            t = tick / 60
            if drops and t - drops[-1][0] > .5 + 1e-9:
                add_burst(stats, drops)
                drops = []
            drops.append((t, count))
            distance = math.hypot(raw[o] * POS_SCALE - cx, raw[o + 2] * POS_SCALE - cz)
            speed = math.hypot(raw[o + 7] * VEL_SCALE, raw[o + 9] * VEL_SCALE)
            in_range = (spec.min_range is not None and spec.max_range is not None
                        and spec.min_range <= distance <= spec.max_range)
            in_speed = spec.max_shot_speed is not None and speed <= spec.max_shot_speed
            stats["drop_frames"] += 1
            stats["net_balls"] += count
            stats["range_frames"] += in_range
            stats["speed_frames"] += in_speed
            stats["inside_frames"] += in_range and in_speed
        add_burst(stats, drops)
    return result


def fraction(stats: dict, field: str, available: bool) -> str:
    if not available or not stats["drop_frames"]:
        return "N/A"
    return f'{stats[field]}/{stats["drop_frames"]} ({stats[field] / stats["drop_frames"]:.1%})'


def main():
    specs = {s.team: s for s in load_specs()}
    totals = {team: empty_stats() for team in specs}
    paths = sorted(RAW.glob("*.rbrp"))
    if not paths:
        raise FileNotFoundError(f"no cached replays in {RAW}")
    for path in paths:
        for team, stats in summarize_replay(parse(path.read_bytes()), specs).items():
            for key, value in stats.items():
                totals[team][key] += value
    print(f"runs={len(paths)} robot_runs={sum(s['runs'] for s in totals.values())} "
          f"teams_observed={sum(s['runs'] > 0 for s in totals.values())}/{len(specs)}")
    print("Rates: net balls/s in >=1 s bursts; gaps <=0.5 s; exclude first drop from numerator.")
    print("Envelope: current drop-frame position and horizontal velocity; no tolerance added.")
    print("N/A: no replay/drop frames, no sustained bursts, or missing brain bounds. Hopper drops may include passes/outtakes and mask concurrent intake.")
    print("| Team | Runs | Drop frames | Net balls | In range | In speed | Both | Measured balls/s | fireRate | Rate bursts |")
    print("|---:|---:|---:|---:|---|---|---|---:|---:|---:|")
    for team, spec in specs.items():
        s = totals[team]
        has_range = spec.min_range is not None and spec.max_range is not None
        has_speed = spec.max_shot_speed is not None
        rate = f'{s["rate_balls"] / s["rate_seconds"]:.2f}' if s["rate_seconds"] else "N/A"
        print(f"| {team} | {s['runs']} | {s['drop_frames']} | {s['net_balls']} | "
              f"{fraction(s, 'range_frames', has_range)} | {fraction(s, 'speed_frames', has_speed)} | "
              f"{fraction(s, 'inside_frames', has_range and has_speed)} | {rate} | {spec.fire_rate:g} | {s['rate_bursts']} |")


if __name__ == "__main__":
    main()
