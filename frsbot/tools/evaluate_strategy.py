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
from frsbot.sim.vec_env import RobotSlot, VecWasmEnv, default_starts
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


class TrackedPolicy(BotPolicy):
    """Accumulate control-time diagnostics inside a local VecWasmEnv rollout."""
    def __init__(self, bot):
        super().__init__(bot)
        self.roles, self.assigned_roles, self.goals = Counter(), Counter(), Counter()
        self.timeline = []
        self.score_seconds = self.feed_releases = self.retreat_seconds = 0.
        self.pending = None

    def observe(self, state, info):
        if self.pending is None:
            return
        t, hopper, role, assigned, goal, scoring, retreat, position = self.pending
        dt = state.t - t
        self.roles[role] += dt
        self.assigned_roles[assigned] += dt
        self.goals[goal] += dt
        self.score_seconds += dt * scoring
        self.retreat_seconds += dt * retreat
        me = state.robots[state.me]
        if goal == "Feed":
            self.feed_releases += max(0, hopper - me.hopper)
        index = int((t + 1e-6) // 5)
        if index == len(self.timeline):
            self.timeline.append(dict(start=t, end=state.t, roles={}, goals={}, score_seconds=0.,
                                      hopper_start=hopper, position_start=position))
        window = self.timeline[index]
        window["end"] = state.t
        window["roles"][role] = window["roles"].get(role, 0.) + dt
        window["goals"][goal] = window["goals"].get(goal, 0.) + dt
        window["score_seconds"] += dt * scoring
        window.update(hopper_end=me.hopper, robot_fuel=info["scored"], our_fouls=info["fouls"],
                      position_end=[me.x, me.z], goal_point_end=list(self.bot.goal.point))
        self.pending = None

    def act(self, state, info=None):
        self.observe(state, info)
        action = self.bot.act(state)
        strategic = isinstance(self.bot, StrategicImAI)
        me = state.robots[state.me]
        self.pending = (state.t, me.hopper, effective_role(self.bot, state),
                        self.bot.layer.role(state).kind if strategic else "Scorer", self.bot.goal.kind,
                        action.score, strategic and self.bot.driver.retreat_target is not None, [me.x, me.z])
        return action


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
    ours = TrackedPolicy((StrategicImAI if strategic else ImAI)(specs[lineup[ego]["team"]]))
    slots = []
    for i, slot in enumerate(lineup):
        bot = ForcedDefender(specs[slot["team"]], ego) if i == defender else ImAI(specs[slot["team"]])
        slots.append(RobotSlot(**slot, policy=ours if i == ego else BotPolicy(bot)))
    state = env.reset([dict(robots=slots, ego=ego, seed=seed, auto="none", event="event", hp=True, unload=True)])[0]
    if defender is not None:
        slots[defender].policy.bot.layer.role = lambda observation: Role("Defender", state.robots[ego].id)
    started = time.perf_counter()
    while state.t < duration - 1e-7:
        result = env.rollout(min(duration, state.t + 5), ticks=3)[0]
        ours.observe(result.views[ego], result.observation.driver[ego])
        ours.timeline[-1].update(our_score=result.scores[alliance], opponent_score=result.scores[other])
        state = result.views[ego]
        if result.done:
            break
    return dict(variant="strategic" if strategic else "pure", alliance=alliance, ego=ego,
                our_score=result.scores[alliance], opponent_score=result.scores[other],
                margin=result.scores[alliance] - result.scores[other],
                robot_fuel=int(result.observation.driver[ego]["scored"]),
                our_fouls=int(result.observation.driver[ego]["fouls"]),
                foul_points_conceded=result.breakdowns[other]["foulPoints"],
                role_fractions={r: ours.roles[r] / state.t for r in ROLES},
                assigned_role_fractions={r: ours.assigned_roles[r] / state.t for r in ROLES},
                goal_seconds=dict(ours.goals), score_command_seconds=ours.score_seconds, feed_hopper_drops=ours.feed_releases,
                retreat_seconds=ours.retreat_seconds, timeline=ours.timeline, duration=state.t,
                wall_seconds=time.perf_counter() - started)


def run_pair(mode, seed, opponent_defender=False, duration=160.):
    lineup, ego = composition(mode, seed)
    specs = {s.team: s for s in load_specs()}
    with VecWasmEnv(n_workers=1, processes=False) as env:
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
                    auto="none", event="event", hp=True, teams=list(TEAMS), backend="VecWasmEnv.rollout",
                    foul_source="alliance points: opponent breakdown.foulPoints; robot count: native foul events",
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
