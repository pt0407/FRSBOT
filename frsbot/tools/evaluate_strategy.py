"""Paired alliance-margin evaluation against identical rule-bot lineups and seeds."""
import argparse
import concurrent.futures
import json
import math
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from frsbot.bot.imai import ImAI
from frsbot.core.specs import load_specs
from frsbot.sim.multi_env import MultiWasmEnv, RobotSlot, default_starts
from frsbot.strategy.bot import StrategicImAI
from frsbot.strategy.roles import Role
from frsbot.tools.evaluate import ROOT, TEAMS


MODES = {"1v1": 1, "2v2": 2, "3v3": 3}
ROLES = ("Scorer", "Feeder", "Defender", "Contester")
OUTPUT = ROOT / "data" / "sim" / "strategy" / "strategy_eval.json"


class BotPolicy:
    """Same act(state, info) contract as the throughput branch's ImAIPolicy."""
    def __init__(self, bot):
        self.bot = bot

    def act(self, state, info=None):
        return self.bot.act(state)


class ForcedDefender(StrategicImAI):
    def __init__(self, spec, target):
        super().__init__(spec)
        self.layer.role = lambda state: Role("Defender", target)


def composition(mode, seed):
    size = MODES[mode]
    rng = np.random.default_rng(np.random.SeedSequence([seed, size]))
    teams = rng.choice(TEAMS, 2 * size, replace=True)
    slots = [RobotSlot(int(team), "blue" if i < size else "red") for i, team in enumerate(teams)]
    starts = default_starts(slots)
    return [dict(team=s.team, alliance=s.alliance, start=start, preload=s.preload)
            for s, start in zip(slots, starts)], (0 if seed % 2 else size)


def effective_role(bot, state):
    if not isinstance(bot, StrategicImAI):
        return "Scorer"
    kind = bot.goal.kind
    if kind == "Defend":
        return "Defender"
    if kind == "Feed":
        return "Feeder"
    if kind == "Contest":
        return "Feeder" if bot.layer.role(state).kind == "Feeder" else "Contester"
    return "Scorer"


def play(env, lineup, ego, seed, strategic, specs, opponent_defender=False, duration=160.):
    if not 0 < duration <= 160:
        raise ValueError("duration must be in (0, 160]")
    alliance = lineup[ego]["alliance"]
    other = "red" if alliance == "blue" else "blue"
    defender = next(i for i, slot in enumerate(lineup) if slot["alliance"] == other) if opponent_defender else None
    ours = (StrategicImAI if strategic else ImAI)(specs[lineup[ego]["team"]])
    slots = []
    for i, slot in enumerate(lineup):
        bot = ForcedDefender(specs[slot["team"]], ego) if i == defender else ImAI(specs[slot["team"]])
        slots.append(RobotSlot(**slot, policy=None if i == ego else BotPolicy(bot)))
    state = env.reset(robots=slots, ego=ego, seed=seed, auto="none", event="event", hp=True, unload=True)
    our_id = state.robots[ego].id
    if defender is not None:
        slots[defender].policy.bot.layer.role = lambda observation: Role("Defender", our_id)
    roles, assigned_roles, goals = Counter(), Counter(), Counter()
    timelines, events, conceded = [], [], 0
    score_seconds = feed_releases = retreat_seconds = 0.
    started = time.perf_counter()
    while state.t < duration - 1e-7:
        action = ours.act(state)
        role = effective_role(ours, state)
        assigned = ours.layer.role(state).kind if strategic else "Scorer"
        goal = ours.goal.kind
        hopper = state.robots[ego].hopper
        ticks = min(3, max(1, round((duration - state.t) * 60)))
        result = env.step({ego: action}, ticks=ticks)
        dt = result.t - state.t
        roles[role] += dt
        assigned_roles[assigned] += dt
        goals[goal] += dt
        score_seconds += dt * action.score
        retreat_seconds += dt * (strategic and ours.driver.retreat_target is not None)
        if goal == "Feed":
            feed_releases += max(0, hopper - result.views[ego].robots[ego].hopper)
        index = int((state.t + 1e-6) // 5)
        if index == len(timelines):
            timelines.append(dict(start=state.t, end=result.t, roles={}, goals={}, score_seconds=0.,
                                  hopper_start=hopper, hopper_end=hopper, our_score=0, opponent_score=0, robot_fuel=0))
        window = timelines[index]
        window["end"] = result.t
        window["roles"][role] = window["roles"].get(role, 0.) + dt
        window["goals"][goal] = window["goals"].get(goal, 0.) + dt
        window["score_seconds"] += dt * action.score
        window["hopper_end"] = result.views[ego].robots[ego].hopper
        window["our_score"], window["opponent_score"] = result.scores[alliance], result.scores[other]
        window["robot_fuel"] = int(result.observation.driver[ego]["scored"])
        if not hasattr(result, "events"):
            raise RuntimeError("paired evaluation requires foul events on StepResult")
        for event in result.events:
            if event["t"] != "foul":
                continue
            events.append(dict(event, observed_at=result.t))
            if event["alliance"] == other:
                conceded += event["points"]
        state = result.views[ego]
        if result.done:
            break
    return dict(variant="strategic" if strategic else "pure", alliance=alliance, ego=ego,
                our_score=result.scores[alliance], opponent_score=result.scores[other],
                margin=result.scores[alliance] - result.scores[other],
                robot_fuel=int(result.observation.driver[ego]["scored"]),
                our_fouls=[e for e in events if e["by"] == our_id], foul_points_conceded=conceded,
                foul_events=events, role_fractions={r: roles[r] / state.t for r in ROLES},
                assigned_role_fractions={r: assigned_roles[r] / state.t for r in ROLES},
                goal_seconds=dict(goals), score_command_seconds=score_seconds, feed_hopper_drops=feed_releases,
                retreat_seconds=retreat_seconds, timeline=timelines, duration=state.t,
                wall_seconds=time.perf_counter() - started)


def run_pair(mode, seed, opponent_defender=False, duration=160.):
    lineup, ego = composition(mode, seed)
    specs = {s.team: s for s in load_specs()}
    with MultiWasmEnv() as env:
        pure = play(env, lineup, ego, seed, False, specs, opponent_defender, duration)
        strategic = play(env, lineup, ego, seed, True, specs, opponent_defender, duration)
    return dict(mode=mode, seed=seed, lineup=lineup, ego=ego, opponent_defender=opponent_defender,
                pure=pure, strategic=strategic)


def summarize(pairs):
    summaries = []
    for mode in MODES:
        group = [p for p in pairs if p["mode"] == mode]
        if not group:
            continue
        pure = np.array([p["pure"]["margin"] for p in group], float)
        strategic = np.array([p["strategic"]["margin"] for p in group], float)
        differences = strategic - pure
        summaries.append(dict(mode=mode, n=len(group), mean_margin_pure=float(pure.mean()),
                              mean_margin_strategic=float(strategic.mean()), paired_diff=float(differences.mean()),
                              paired_se=float(differences.std(ddof=1) / math.sqrt(len(group))) if len(group) > 1 else None,
                              win_rate_pure=float((pure > 0).mean()), win_rate_strategic=float((strategic > 0).mean()),
                              foul_points_pure=float(np.mean([p["pure"]["foul_points_conceded"] for p in group])),
                              foul_points_strategic=float(np.mean([p["strategic"]["foul_points_conceded"] for p in group])),
                              role_fractions={r: float(np.mean([p["strategic"]["role_fractions"][r] for p in group])) for r in ROLES}))
    return summaries


def markdown(summaries):
    lines = ["| Mode | n pairs | Margin pure | Margin strategic | Paired diff ± SE | Win pure/strategic | Foul pts pure/strategic | Roles S/F/D/C |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for s in summaries:
        se = f'{s["paired_se"]:.1f}' if s["paired_se"] is not None else "n/a"
        roles = "/".join(f'{100*s["role_fractions"][r]:.1f}%' for r in ROLES)
        lines.append(f'| {s["mode"]} | {s["n"]} | {s["mean_margin_pure"]:.1f} | {s["mean_margin_strategic"]:.1f} | '
                     f'{s["paired_diff"]:+.1f} ± {se} | {s["win_rate_pure"]:.0%}/{s["win_rate_strategic"]:.0%} | '
                     f'{s["foul_points_pure"]:.1f}/{s["foul_points_strategic"]:.1f} | {roles} |')
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", type=int, default=12, help="paired seeds 1..N per mode")
    parser.add_argument("--jobs", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--opponent-defender", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    start = time.perf_counter()
    pairs = []
    metadata = dict(created=datetime.now(timezone.utc).isoformat(), seeds=args.seeds, modes=args.modes,
                    opponent_defender=args.opponent_defender, jobs=args.jobs, duration=160, ticks=3,
                    auto="none", event="event", hp=True, teams=list(TEAMS),
                    revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    win_definition="margin > 0; draws are not wins", role_order=list(ROLES))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_pair, mode, seed, args.opponent_defender)
                   for mode in dict.fromkeys(args.modes) for seed in range(1, args.seeds + 1)]
        for future in concurrent.futures.as_completed(futures):
            pair = future.result()
            pairs.append(pair)
            pairs.sort(key=lambda p: (p["mode"], p["seed"]))
            elapsed = time.perf_counter() - start
            args.output.write_text(json.dumps(dict(metadata=metadata, pairs=pairs, summaries=summarize(pairs),
                                                   wall_seconds=elapsed), indent=2) + "\n")
            print(f'{pair["mode"]} seed={pair["seed"]} pure={pair["pure"]["margin"]} '
                  f'strategic={pair["strategic"]["margin"]} diff={pair["strategic"]["margin"]-pair["pure"]["margin"]:+d}', flush=True)
    table = markdown(summarize(pairs))
    args.output.with_suffix(".md").write_text(table + f"\n\nWall time: {elapsed:.2f}s\n")
    print(table)
    print(f"Wall time: {elapsed:.2f}s; wrote {args.output}")


if __name__ == "__main__":
    main()
