"""Evaluate ImAI in full real-engine solo matches and write viewer-compatible traces."""
import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path

from frsbot.bot.imai import ImAI
from frsbot.core.specs import load_specs
from frsbot.sim.wasm_env import WasmEnv

ROOT = Path(__file__).resolve().parents[2]
TEAMS = (254, 2910, 1114, 4414, 60026)


def leaderboard(team):
    entries = json.loads((ROOT / "data" / "board.json").read_text())["entries"]
    scores = [(e["points"], "board", e["name"]) for e in entries if e["team"] == team and e["name"] != "404"]
    for path in (ROOT / "data" / "telemetry").glob("*.json"):
        run = json.loads(path.read_text())
        board = run.get("leaderboard") or {}
        if run["robots"][0]["team"] == team and board.get("points") is not None and board.get("name") != "404":
            scores.append((board["points"], "telemetry", board.get("name", "unknown")))
    return max(scores, default=(None, "unknown", "unknown"))


def evaluate_run(spec, alliance, seed, output, ticks=3, duration=166, moving_score=None):
    if not 0 < duration <= 166 or ticks not in (1, 2, 3, 6):
        raise ValueError("duration must be in (0, 166] and ticks must be 1, 2, 3 or 6")
    bot = ImAI(spec)
    if moving_score is not None:
        bot.driver.shoot_while_collecting = moving_score
    run_id = f"imai-{spec.team}-{alliance}-{seed}"
    goals, blocked, attempts = Counter(), Counter(), 0
    samples, maximum, low_hopper_seconds, carried_at_close, collect_score_seconds = [], 0, 0, None, 0
    with WasmEnv() as env:
        state = env.reset(spec.team, alliance=alliance, seed=seed, auto="none", event="event", hp=True, unload=True)
        while True:
            inputs = bot.act(state)
            robot = state.robots[state.me]
            maximum = max(maximum, robot.hopper)
            if state.t >= 163 and carried_at_close is None:
                carried_at_close = robot.hopper
            if not samples or state.t * 60 - samples[-1]["tick"] >= 6 - 1e-6 or state.t >= duration:
                raw = env.view["robots"][state.me]
                observed = {k: raw[k] for k in ("pos", "quat", "vel", "hopperFuel", "drumRpm", "gate", "state")}
                samples.append(dict(tick=round(state.t * 60), robots=[observed],
                                    goal=dict(kind=bot.goal.kind, point=bot.goal.point), phase=state.phase.value,
                                    score=env.view["score"], recovery=bot.driver.recovering,
                                    recovery_stage=bot.driver.recovery_stage,
                                    inputs=dict(score=inputs.score, intake=inputs.intake, turn=inputs.turn)))
            if state.t >= duration:
                break
            n = min(ticks, round((duration - state.t) * 60))
            goals[bot.goal.kind] += n / 60
            if bot.goal.kind == "Score" and robot.hopper <= 5:
                low_hopper_seconds += n / 60
            if bot.goal.kind == "Collect" and inputs.score:
                collect_score_seconds += n / 60
            if inputs.score:
                attempts += n
                if robot.gate:
                    for reason, value in robot.gate.items():
                        if reason != "open" and value:
                            blocked[reason] += n
            state = env.step(inputs, n)
        score = env.view["score"]
        tally = score[alliance]
        result = dict(team=spec.team, alliance=alliance, seed=seed, auto="none", score=tally["total"],
                      fuel=tally["autoFuel"] + tally["teleFuel"], inactive_fuel=tally["inactiveFuel"],
                      max_hopper=maximum, goal_seconds={k: round(v, 2) for k, v in goals.items()},
                      low_hopper_score_seconds=round(low_hopper_seconds, 2),
                      carried_at_close=carried_at_close,
                      collect_score_seconds=round(collect_score_seconds, 2),
                      moving_score=bot.driver.shoot_while_collecting,
                      robot_fuel=sum(d["fuel"] for d in env.view.get("drivers", []) if d["robot"] == robot.id),
                      score_command_ticks=attempts, blocked_gate_ticks=dict(blocked),
                      blocked_shot_opportunities={k: round(v / 60 * spec.fire_rate, 1) for k, v in blocked.items()})
    trace = dict(id=run_id, kind="sim", durationMs=round(duration * 1000), score=score,
                 robots=[dict(id=0, team=spec.team, alliance=alliance, name="ImAI")], samples=samples,
                 leaderboard=dict(name="ImAI (real engine)", points=None), evaluation=result)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{run_id}.json").write_text(json.dumps(trace, separators=(",", ":")))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", nargs="+", type=int, default=TEAMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--alliances", nargs="+", choices=["blue", "red"], default=["blue", "red"])
    parser.add_argument("--ticks", type=int, choices=[1, 2, 3, 6], default=3)
    parser.add_argument("--duration", type=float, default=166)
    parser.add_argument("--moving-score", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "sim" / "imai")
    args = parser.parse_args()
    specs = {s.team: s for s in load_specs()}
    if not 0 < args.duration <= 166 or any(t not in specs for t in args.teams):
        parser.error("duration must be in (0, 166] and all teams must have a robot spec")
    results = []
    print("team alliance seed score fuel max_hopper goal_seconds blocked_gate_ticks", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(evaluate_run, specs[team], alliance, seed, args.output, args.ticks, args.duration, args.moving_score)
                   for team in args.teams for alliance in args.alliances for seed in args.seeds]
        for future in futures:
            r = future.result()
            results.append(r)
            print(r["team"], r["alliance"], r["seed"], r["score"], r["fuel"], r["max_hopper"], r["goal_seconds"], r["blocked_gate_ticks"], flush=True)
    comparisons = []
    print("team mean_score best_bot leaderboard_best bot_percent source driver", flush=True)
    for team in args.teams:
        own = [r["score"] for r in results if r["team"] == team]
        best, source, driver = leaderboard(team)
        mean = sum(own) / len(own)
        comparisons.append(dict(team=team, mean=mean, best_bot=max(own), leaderboard=best, source=source, driver=driver))
        print(team, round(mean, 1), max(own), best, round(100 * mean / best, 1) if best else None, source, driver, flush=True)
    (args.output / "summary.json").write_text(json.dumps(dict(runs=results, leaderboard=comparisons), indent=2) + "\n")
    print("Gate counts are overlapping blocked command-ticks; summary also estimates blocked fire opportunities at configured fireRate, not actual lost shots. Scores include enabled human players.")


if __name__ == "__main__":
    main()
