"""Measure score-button startup and net hopper drain with HP/intake disabled."""
import argparse
import concurrent.futures
import json
from pathlib import Path

from frsbot.bot.imai import ImAI
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.sim.wasm_env import WasmEnv


def measure(spec, seed):
    with WasmEnv() as env:
        bot = ImAI(spec)
        state = env.reset(spec.team, seed=seed, hp=False, unload=False)
        while state.t < 40:
            inputs = bot.act(state)
            if inputs.score and state.robots[state.me].hopper >= .5 * state.robots[state.me].capacity:
                break
            state = env.step(inputs, 3)
        else:
            return dict(team=spec.team, seed=seed, error="no loaded scoring attempt within 40 s")
        pressed = state.t
        previous = state.robots[state.me].hopper
        drops = []
        while state.t < min(pressed + 12, 20 if pressed < 20 else 33):
            state = env.step(Inputs(score=True, field_relative=True), 1)
            current = state.robots[state.me].hopper
            if current < previous:
                drops.append((state.t, previous - current))
            previous = current
            if current <= 5:
                break
        if len(drops) < 2:
            return dict(team=spec.team, seed=seed, error="not enough drops", gate=state.robots[state.me].gate)
        rate = sum(count for _, count in drops[1:]) / (drops[-1][0] - drops[0][0])
        return dict(team=spec.team, seed=seed, startup_seconds=drops[0][0] - pressed,
                    drain_rate=rate, pressed=pressed, last_drop=drops[-1][0], remaining=previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = [s for s in load_specs() if s.team in (254, 2910, 1114, 4414, 60026)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(measure, spec, seed) for spec in specs for seed in (1, 2, 3)]
        rows = [f.result() for f in futures]
    measured = {}
    for spec in specs:
        valid = [r for r in rows if r["team"] == spec.team and "error" not in r]
        if not valid:
            raise RuntimeError(f"no shooting measurements for {spec.team}")
        measured[str(spec.team)] = dict(startup_seconds=round(max(r["startup_seconds"] for r in valid) + .2, 3),
                                        drain_rate=round(min(r["drain_rate"] for r in valid), 3))
    args.output.write_text(json.dumps(dict(measurements=rows, profiles=measured), indent=2) + "\n")
    print(json.dumps(measured, indent=2))
    for row in rows:
        if "error" in row:
            print("FAILED", row)


if __name__ == "__main__":
    main()
