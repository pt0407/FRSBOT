"""DAgger rollout: drive the blended learned+rule policy in the real engine and
label every state it visits with the rule driver's sticks (the programmatic
expert). Merged into training, this teaches the net to servo on the states it
actually reaches instead of drifting off the human distribution.
Usage: python -m frsbot.imitation.dagger_rollout --teams 254 4414 --seeds 1 2
"""
import argparse
from pathlib import Path

import numpy as np

from frsbot.bot.imai import ImAI
from frsbot.bot.planner import Goal
from frsbot.bot.tuning import PARAMS
from frsbot.core.specs import load_specs
from frsbot.imitation.train_drive import features
from frsbot.sim.wasm_env import WasmEnv

ROOT = Path(__file__).resolve().parents[2]
SPECS = {s.team: s for s in load_specs()}


def rollout(spec, alliance, seed, duration, ticks):
    bot = ImAI(spec)
    X, Y = [], []
    with WasmEnv() as env:
        state = env.reset(spec.team, alliance=alliance, seed=seed, auto="none", event="event", hp=True, unload=True)
        while state.t < duration:
            inputs = bot.act(state)
            steer = bot.driver.last_steer
            goal = Goal(steer["kind"], tuple(map(float, steer["point"])))
            X.append(features(state, spec, goal))
            Y.append([*steer["rule"], steer.get("turn", 0.)])
            state = env.step(inputs, ticks)
    return np.array(X, np.float32), np.array(Y, np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", nargs="+", type=int, default=[254, 4414, 1678])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--alliances", nargs="+", default=["blue"])
    parser.add_argument("--blend", type=float, default=.35)
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--duration", type=float, default=166)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "imitation")
    args = parser.parse_args()
    PARAMS["learned_drive"] = 1
    PARAMS["learned_blend"] = args.blend
    PARAMS["learned_turn_blend"] = 0
    args.output.mkdir(parents=True, exist_ok=True)
    for team in args.teams:
        for alliance in args.alliances:
            for seed in args.seeds:
                X, Y = rollout(SPECS[team], alliance, seed, args.duration, args.ticks)
                path = args.output / f"dagger-{team}-{alliance}-{seed}.npz"
                np.savez_compressed(path, X=X, Y=Y, W=np.ones(len(X), np.float32))
                print(f"{path.name}: {len(X)} states", flush=True)


if __name__ == "__main__":
    main()
