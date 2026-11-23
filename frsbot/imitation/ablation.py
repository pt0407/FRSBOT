"""Paired pose ablations on high-error and seeded random control windows."""
import argparse
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path

import numpy as np

from frsbot.imitation.buttons import transition_buttons
from frsbot.imitation.recover import (convert, discover, load_specs, make_segment, parse, windows, _run)
from frsbot.scrape.harvest import DATA


def _alternatives(robots, specs, rank):
    def distance(a, b):
        return sum(((getattr(a, key) - getattr(b, key)) / getattr(a, key)) ** 2
                   for key in ("bumper_length", "bumper_width", "max_speed", "turn_rate", "accel"))

    result = []
    for robot in robots:
        spec = specs[robot["team"]]
        others = sorted((s for s in specs.values() if s.team != spec.team), key=lambda s: distance(spec, s))
        other = others[rank]
        result.append(dict(robot, engineTeam=other.team, speed=other.max_speed, turnRate=other.turn_rate))
    return result


def _jobs(replay, states, ticks, old, specs, count, rng):
    first = np.r_[True, old["segment"][1:] != old["segment"][:-1]]
    last = np.r_[first[1:], True]
    starts = old["tick"][first]
    errors = old["residual"][last, :, 0].max(axis=1)
    ordered = np.argsort(errors)
    high = ordered[-count:]
    remaining = ordered[:-count]
    control = rng.choice(remaining, min(len(remaining), count), replace=False)
    buttons = transition_buttons(states, ticks, replay.events)
    fine = {ticks[w[0]]: w for w in windows(ticks, states, 2)}
    coarse = {ticks[w[0]]: w for w in windows(ticks, states, 6)}
    for cohort, indexes in (("tail", high), ("control", control)):
        for t in starts[indexes]:
            _, baseline = make_segment(replay, states, ticks, fine[t], specs, buttons)
            _, period = make_segment(replay, states, ticks, coarse[t], specs, buttons)
            variants = {
                "recorded velocities / 2 ticks": baseline,
                "zero initial velocities": dict(baseline, resetVelocity="zero"),
                "position-derived linear velocity": dict(baseline, resetVelocity="estimate"),
                "6 ticks": period,
                "robot-relative": dict(baseline, robotRelative=True),
                "hood on": dict(baseline, hood=True),
                "stow on": dict(baseline, stow=True),
                "neighbor spec 1": dict(baseline, robots=_alternatives(baseline["robots"], specs, 0)),
                "neighbor spec 2": dict(baseline, robots=_alternatives(baseline["robots"], specs, 1)),
            }
            yield (cohort, int(t), variants)


def _evaluate(job):
    cohort, tick, variants = job
    poses, reset_errors = {}, None
    for name, segment in variants.items():
        result = _run(segment)
        if "rejection" not in result:
            poses[name] = [row[0] for row in result["residuals"][-1]]
            if reset_errors is None and name.startswith("recorded"):
                wanted = segment["frames"][0]["bodies"]
                actual = result["initialBodies"]
                reset_errors = [
                    max(abs(a - b) for x, y in zip(wanted, actual) for a, b in zip(x["vel"], y["vel"])),
                    max(abs(x["omega"] - y["omega"]) for x, y in zip(wanted, actual)),
                ]
    return cohort, tick, poses, reset_errors


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--previous", type=Path, required=True)
    cli.add_argument("--workers", type=int, required=True)
    cli.add_argument("--per-replay", type=int, default=2)
    args = cli.parse_args()
    specs = {s.team: s for s in load_specs()}
    rng = np.random.default_rng(2026)
    summary = defaultdict(lambda: defaultdict(list))
    reset_errors = []
    rejected = 0
    with mp.Pool(args.workers) as pool:
        for source, path in discover(DATA / "raw", DATA / "raw_mp"):
            name = path.name.removesuffix(".gz").removesuffix(".rbrp")
            old_path = args.previous / f"{source}-{name}.npz"
            if not old_path.exists():
                continue
            replay = parse(path.read_bytes(), fuel=True)
            states = convert(replay, list(specs.values()))
            ticks = [tick for tick, _ in replay.frames]
            with np.load(old_path, allow_pickle=False) as old:
                jobs = list(_jobs(replay, states, ticks, old, specs, args.per_replay, rng))
            for cohort, tick, poses, reset in pool.imap(_evaluate, jobs, chunksize=1):
                if len(poses) != len(jobs[0][2]):
                    rejected += 1
                    print(f"{name} {tick}: rejected variants, omitted from paired analysis", flush=True)
                    continue
                reset_errors.append(reset)
                for variant, errors in poses.items():
                    summary[cohort][variant].extend(errors)
                    summary["all"][variant].extend(errors)
            print(f"{name}: {len(jobs)} paired windows attempted", flush=True)
    print(f"Paired-window exclusions: {rejected}")
    print(f"Max readback error: linear/angular velocity {np.max(reset_errors, axis=0)}")
    print("| Variant | All median / p90 | Tail median / p90 | Control median / p90 |")
    print("| --- | ---: | ---: | ---: |")
    print(f"| Robot endpoints | {len(next(iter(summary['all'].values())))} | "
          f"{len(next(iter(summary['tail'].values())))} | {len(next(iter(summary['control'].values())))} |")
    for variant in summary["all"]:
        cols = []
        for cohort in ("all", "tail", "control"):
            median, p90 = np.percentile(summary[cohort][variant], [50, 90]) * 100
            cols.append(f"{median:.2f} / {p90:.2f} cm")
        print(f"| {variant} | {' | '.join(cols)} |")


if __name__ == "__main__":
    main()
