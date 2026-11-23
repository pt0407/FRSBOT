"""Engine throughput: straight-drive traces, raw ticks/s, and matches/minute for each docs/PERF.md step.

    python3 -m frsbot.tools.benchmark_engine                       # traces + raw ticks/s (original output)
    python3 -m frsbot.tools.benchmark_engine --step all --workers 8 --matches 16

Steps (cumulative): baseline = Python ImAI over JSON lines (server.mjs), acting every 3 ticks like evaluate.py;
1 = JS driver every tick in the engine process, Python planner every --decide ticks, still JSON;
2 = same over the binary protocol (vec_server.mjs); 3 = VecWasmEnv, --workers Node workers x --per-worker matches,
slot policies run in each worker's own Python process via VecWasmEnv.rollout (--lockstep: VecWasmEnv.step, one
barrier per decision as with central batched inference); 3x6 = step 3 with 6 robots per match (all rule ImAI).
Step 4 (decode cache) is timed by `python3 -m frsbot.replay.cache --benchmark`.
"""
import argparse
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.sim.ipc import driver_config
from frsbot.sim.wasm_env import JsonWasmEnv, WasmEnv

TEAMS = (254, 2910, 1114, 4414, 60026, 1678)


def traces(args):
    with WasmEnv() as env:
        for team in args.teams:
            state = env.reset(team, start=dict(fx=0, fy=0, headingDeg=0), seed=1)
            print(f"team={team} engine={env.engine} start={{fx:0,fy:0,headingDeg:0}} seed=1")
            print("t_s x_m z_m speed_m_s hopper")
            for second in range(6):
                if second:
                    state = env.step(Inputs(y=1, field_relative=True), ticks=60)
                robot = state.robots[state.me]
                print(f"{state.t:.1f} {robot.x:.4f} {robot.z:.4f} {math.hypot(robot.vx, robot.vz):.4f} {robot.hopper}")
    for name, cls in (("json", JsonWasmEnv), ("binary", WasmEnv)):
        with cls() as env:
            for team in args.teams:
                env.reset(team, seed=1)
                before = time.perf_counter()
                env.step(Inputs(), ticks=args.ticks)
                elapsed = time.perf_counter() - before
                print(f"throughput {name} team={team} ticks={args.ticks} wall_s={elapsed:.4f} ticks_per_wall_s={args.ticks / elapsed:.1f} (one batched step; excludes reset)")


def run_match(step, team, seed, seconds, decide):
    """One single-robot match in its own process; returns the blue score."""
    from frsbot.bot.imai import ImAI
    from frsbot.sim.policies import GoalImAIPolicy
    spec = next(s for s in load_specs() if s.team == team)
    if step == "baseline":
        bot = ImAI(spec)
        with JsonWasmEnv() as env:
            state = env.reset(team, seed=seed)
            while state.t < seconds:
                state = env.step(bot.act(state), min(3, round((seconds - state.t) * 60)))
            return env.view["score"]["blue"]["total"]
    policy = GoalImAIPolicy(spec)
    env_cls = JsonWasmEnv if step == "1" else WasmEnv
    with env_cls() as env:
        state = env.reset(team, seed=seed, driver=driver_config(spec)) if step == "1" else env.reset(team, seed=seed)
        while state.t < seconds:
            info = env.driver.get("0") if step == "1" else env.observation.driver[0]
            state = env.step(policy.act(state, info), min(decide, round((seconds - state.t) * 60)))
        return env.view["score"]["blue"]["total"]


def per_process(step, args):
    jobs = [(TEAMS[i % len(TEAMS)], 1 + i // len(TEAMS)) for i in range(args.matches)]
    before = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        scores = list(pool.map(run_match, [step] * len(jobs), *zip(*jobs), [args.seconds] * len(jobs), [args.decide] * len(jobs)))
    return time.perf_counter() - before, scores, len(jobs)


def vectorised(robots_per_match, args):
    from frsbot.sim.policies import GoalImAIPolicy
    from frsbot.sim.vec_env import RobotSlot, VecWasmEnv
    specs = {s.team: s for s in load_specs()}
    alliances = ("blue", "red", "blue", "red", "blue", "red")
    total = args.workers * args.per_worker
    rounds = max(1, math.ceil(args.matches / total))
    scores = []
    before = time.perf_counter()
    with VecWasmEnv(args.workers, args.per_worker) as env:
        for r in range(rounds):
            configs = []
            for m in range(total):
                teams = [TEAMS[(m + r * total + k) % len(TEAMS)] for k in range(robots_per_match)]
                configs.append(dict(robots=[RobotSlot(t, alliances[k], policy=GoalImAIPolicy(specs[t])) for k, t in enumerate(teams)],
                                    seed=1 + r))
            states = env.reset(configs)
            if args.lockstep:
                t = states[0].t
                while t < args.seconds:
                    results = env.step(None, min(args.decide, round((args.seconds - t) * 60)))
                    t = results[0].t
            else:
                results = env.rollout(args.seconds, args.decide)
            scores += [res.scores["blue"] for res in results]
    return time.perf_counter() - before, scores, rounds * total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teams", type=int, nargs="+", default=[2910, 254])
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--step", choices=["baseline", "1", "2", "3", "3x6", "all"])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--per-worker", type=int, default=4, help="matches per Node worker (step 3)")
    parser.add_argument("--matches", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=166)
    parser.add_argument("--decide", type=int, default=6, help="ticks between Python decisions once the JS driver runs every tick")
    parser.add_argument("--lockstep", action="store_true", help="step 3 via VecWasmEnv.step (barrier every decision) "
                        "instead of VecWasmEnv.rollout")
    args = parser.parse_args()
    if args.step is None:
        return traces(args)
    steps = ["baseline", "1", "2", "3", "3x6"] if args.step == "all" else [args.step]
    print(f"cores={os.cpu_count()} workers={args.workers} seconds={args.seconds} decide_ticks={args.decide} per_worker={args.per_worker}")
    print("step matches robots_per_match wall_s matches_per_min robot_ticks_per_s mean_blue_score")
    for step in steps:
        robots = 6 if step == "3x6" else 1
        wall, scores, n = vectorised(robots, args) if step.startswith("3") else per_process(step, args)
        ticks = n * robots * args.seconds * 60
        print(f"{step} {n} {robots} {wall:.1f} {60 * n / wall:.1f} {ticks / wall:.0f} {statistics.mean(scores):.1f}", flush=True)


if __name__ == "__main__":
    main()
