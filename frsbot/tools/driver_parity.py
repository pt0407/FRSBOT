"""Compare the JS driver port (frsbot/sim/engine/driver.mjs) with the Python reference on recorded states."""
import argparse
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from frsbot.bot.calibration import profile
from frsbot.bot.driver import Driver
from frsbot.bot.imai import ImAI
from frsbot.bot.planner import Goal
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.sim.wasm_env import WasmEnv, find_node

DRIVER_JS = Path(__file__).resolve().parents[1] / "sim" / "engine" / "driver.mjs"
AXES = ("x", "y", "turn")
BUTTONS = ("intake", "score", "field_relative")


def state_json(state) -> dict:
    return dict(t=state.t, me=state.me,
                hubs={a: dict(active=h.active, seconds_to_toggle=None if math.isinf(h.seconds_to_toggle) else h.seconds_to_toggle)
                      for a, h in state.hubs.items()},
                robots=[dict(id=r.id, team=r.team, alliance=r.alliance, x=r.x, z=r.z, a=r.a, vx=r.vx, vz=r.vz,
                             hopper=r.hopper, capacity=r.capacity, gate=r.gate) for r in state.robots],
                fuel_pos=state.fuel.pos.astype(np.float64).ravel().tolist(), fuel_where=state.fuel.where.tolist())


def driver_config(spec, shoot_while_collecting=None) -> dict:
    config = dict(spec=asdict(spec), profile=profile(spec.team))
    if shoot_while_collecting is not None:
        config["shootWhileCollecting"] = shoot_while_collecting
    return config


def record(spec, alliance="blue", seed=1, seconds=166, ticks=3):
    """Run Python ImAI in the real engine and keep every (goal, state) the driver saw."""
    bot, calls = ImAI(spec), []
    act = bot.driver.act

    def recording(goal, state):
        calls.append((goal, state))
        return act(goal, state)
    bot.driver.act = recording
    with WasmEnv() as env:
        state = env.reset(spec.team, alliance=alliance, seed=seed)
        while state.t < seconds:
            state = env.step(bot.act(state), ticks)
    return calls


def compare(spec, calls, shoot_while_collecting=None, node=None):
    """Feed one sequence to a fresh Python Driver and a fresh JS driver; return per-call differences."""
    config = driver_config(spec, shoot_while_collecting)
    python = Driver(spec)
    if shoot_while_collecting is not None:
        python.shoot_while_collecting = shoot_while_collecting
    lines = [json.dumps(config)] + [json.dumps(dict(goal=dict(kind=g.kind, point=list(g.point)), state=state_json(s)))
                                    for g, s in calls]
    result = subprocess.run([node or find_node(), str(DRIVER_JS)], input="\n".join(lines) + "\n",
                            capture_output=True, text=True, check=True, timeout=600)
    outputs = [json.loads(line) for line in result.stdout.splitlines()[1:]]
    if len(outputs) != len(calls):
        raise RuntimeError(f"JS driver returned {len(outputs)} of {len(calls)} results: {result.stderr[-2000:]}")
    axis_error, button_mismatch, flag_mismatch = 0.0, 0, 0
    for (goal, state), js in zip(calls, outputs):
        expected = python.act(goal, state)
        axis_error = max(axis_error, *(abs(getattr(expected, a) - js[a]) for a in AXES))
        button_mismatch += any(getattr(expected, b) != js["fieldRelative" if b == "field_relative" else b] for b in BUTTONS)
        flag_mismatch += (python.abandon != js["abandon"] or python.recovering != js["recovering"]
                          or python.recovery_stage != js["stage"])
    return dict(states=len(calls), max_axis_error=axis_error, button_mismatches=button_mismatch, flag_mismatches=flag_mismatch)


def synthetic(spec, n=400, seed=0):
    """Random but internally consistent states that exercise routing, recovery and moving shots offline."""
    from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState
    rng = np.random.default_rng(seed)
    calls, t, hopper = [], 20.0, 30
    x, z, a = -6.0, 1.0, 0.0
    goal = Goal("Score", (-6.0, 0.5))
    for i in range(n):
        t += 1 / 20
        if i % 40 == 0:
            kind = ["Collect", "Score", "Stage"][rng.integers(3)]
            goal = Goal(kind, (float(rng.uniform(-7.5, 7.5)), float(rng.uniform(-3.5, 3.5))))
        x = float(np.clip(x + rng.normal(0, .15), -8, 8))
        z = float(np.clip(z + rng.normal(0, .15), -3.9, 3.9))
        a = float(a + rng.normal(0, .2))
        if rng.random() < .1:
            hopper = int(max(0, hopper - rng.integers(0, 4)))
        pos = rng.uniform([-8, 0, -4], [8, .3, 4], size=(504, 3)).astype(np.float32)
        robot = RobotState(0, spec.team, "blue" if i < n // 2 else "red", x, z, a, float(rng.normal(0, .5)),
                           float(rng.normal(0, .5)), hopper, spec.hopper_capacity, gate=dict(open=bool(rng.random() < .5)))
        hubs = {al: HubState(bool(rng.random() < .7), float(rng.uniform(0, 25))) for al in ("blue", "red")}
        calls.append((goal, GameState(t, Phase.SHIFT1, hubs, [robot], FuelState(pos, rng.integers(0, 4, 504).astype(np.uint8)), 0)))
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", type=int, nargs="+", default=[254, 2910, 1114, 4414, 60026])
    parser.add_argument("--alliances", nargs="+", default=["blue", "red"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=166)
    args = parser.parse_args()
    specs = {s.team: s for s in load_specs()}
    print("source team alliance states max_axis_error button_mismatches flag_mismatches")
    for team in args.teams:
        r = compare(specs[team], synthetic(specs[team]))
        print("synthetic", team, "-", r["states"], f"{r['max_axis_error']:.3g}", r["button_mismatches"], r["flag_mismatches"], flush=True)
        for alliance in args.alliances:
            r = compare(specs[team], record(specs[team], alliance, args.seed, args.seconds))
            print("engine", team, alliance, r["states"], f"{r['max_axis_error']:.3g}", r["button_mismatches"], r["flag_mismatches"], flush=True)


if __name__ == "__main__":
    main()
