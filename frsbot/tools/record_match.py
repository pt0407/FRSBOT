"""Record one headless 2v2 match from the WASM engine (rule bots only) into viewer/match.js.

Usage: python -m frsbot.tools.record_match --seed 1 --step 0.5
Open viewer/match.html to watch it.
"""
import argparse
import json
from pathlib import Path

from frsbot.core.specs import load_specs
from frsbot.sim.policies import ImAIPolicy
from frsbot.sim.vec_env import RobotSlot, VecWasmEnv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "viewer" / "match.js"
LINEUP = [(254, "blue"), (1114, "blue"), (2910, "red"), (4414, "red")]
MATCH_SECONDS = 166.0


def frame(res) -> dict:
    view = res.views[0]
    return {
        "t": round(res.t, 2),
        "phase": view.phase.value,
        "scores": {a: round(s, 1) for a, s in res.scores.items()},
        "hubs": {a: bool(h.active) for a, h in view.hubs.items()},
        "robots": [{"team": r.team, "alliance": r.alliance, "x": round(r.x, 3), "z": round(r.z, 3),
                    "a": round(r.a, 4), "vx": round(r.vx, 3), "vz": round(r.vz, 3), "hopper": int(r.hopper)}
                   for r in view.robots],
        "fuel": [[round(float(x), 2), round(float(z), 2)] for x, _, z in view.fuel.pos.tolist()],
    }


def record(seed: int, step: float) -> dict:
    specs = {s.team: s for s in load_specs()}
    robots = [RobotSlot(team, alliance, policy=ImAIPolicy(specs[team])) for team, alliance in LINEUP]
    frames = []
    with VecWasmEnv(n_workers=1, matches_per_worker=1) as env:
        env.reset([dict(robots=robots, seed=seed, auto="none", event="event", hp=True, unload=True)])
        t = 0.0
        while True:
            res = env.rollout(t, ticks=3)[0]
            frames.append(frame(res))
            if res.done or t >= MATCH_SECONDS:
                break
            t = min(MATCH_SECONDS, t + step)
    return {"seed": seed, "lineup": [{"team": team, "alliance": alliance} for team, alliance in LINEUP],
            "frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step", type=float, default=0.5, help="seconds between recorded frames")
    args = parser.parse_args()
    match = record(args.seed, args.step)
    OUT.write_text("window.MATCH = " + json.dumps(match, separators=(",", ":")) + ";\n")
    last = match["frames"][-1]
    print(f"recorded {len(match['frames'])} frames to {OUT}; final scores {last['scores']} at t={last['t']}")


if __name__ == "__main__":
    main()
