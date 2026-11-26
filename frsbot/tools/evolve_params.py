"""CMA-ES evolution of the rule driver's tuning params in the vectorised real engine.

This is the game-appropriate optimiser: the engine is a perfect model at ~140x
realtime and the decision layer is ~17 interacting scalars, so direct search on
true match score beats gradient methods with credit-assignment noise. Candidates
are evaluated as full solo matches, one per (team, seed) cell, inside the worker
processes so 8 matches run in parallel. Fitness is the alliance total — the
number the live scoreboard shows.
Usage: python -m frsbot.tools.evolve_params [--generations 300] [--parallel 8]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from frsbot.bot.tuning import DEFAULTS
from frsbot.sim.vec_env import RobotSlot, VecWasmEnv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "rl"
BEST = OUT / "evolved_params.json"

# (name, lo, hi) — genome order is fixed; values normalised to [0, 1] for CMA.
GENES = [
    ("collect_frac", 0.1, 0.9),
    ("active_score_frac", 0.5, 1.0),
    ("stage_full_frac", 0.6, 1.0),
    ("drive_dist_gain", 0.5, 4.0),
    ("drive_brake_frac", 0.1, 1.0),
    ("collect_near_frac", 0.2, 1.0),
    ("collect_cos_floor", 0.01, 0.7),
    ("cluster_bias", 0.1, 2.0),
    ("goal_commit", 1.0, 2.5),
    ("hub_near_scale", 0.3, 1.0),
    ("pre_fire_s", 0.5, 4.0),
    ("post_fire_s", 0.0, 1.0),
    ("score_radius_frac", 0.2, 0.9),
    ("sweep_scale", 0.5, 2.0),
    ("active_hub_bias", 0.0, 0.5),
    ("collect_stall_s", 0.5, 4.0),
    ("arrive_margin_s", -1.0, 2.0),
    ("collect_cruise_frac", 0.3, 1.0),
]
TEAMS = (254, 4414, 1678)


class ParamPolicy:
    """Worker-side ImAI wrapper applying one candidate's params around each act."""

    def __init__(self, team, params):
        self.team, self.params, self.bot = team, dict(params), None

    def act(self, state, info=None):
        if self.bot is None:
            from frsbot.bot.imai import ImAI
            from frsbot.core.specs import load_specs
            specs = {s.team: s for s in load_specs()}
            self.bot = ImAI(specs[self.team])
        from frsbot.bot.tuning import PARAMS
        old = {k: PARAMS.get(k) for k in self.params}
        PARAMS.update(self.params)
        try:
            return self.bot.act(state)
        finally:
            PARAMS.update(old)


def denormalise(x):
    return {name: lo + float(np.clip(v, 0, 1)) * (hi - lo) for (name, lo, hi), v in zip(GENES, x)}


def normalise_defaults():
    return np.array([(DEFAULTS[name] - lo) / (hi - lo) for name, lo, hi in GENES])


def evaluate(env, x, teams, seed, slots):
    """Run one candidate as full solo matches in the given slots; mean alliance total."""
    params = denormalise(x)
    configs = [dict(robots=[RobotSlot(team=team, policy=ParamPolicy(team, params))], seed=seed) for team in teams]
    env.reset(configs, matches=list(slots))
    env.rollout(163.0, 3, matches=list(slots))
    return sum(float(env.observations[slot].breakdowns["blue"]["total"]) for slot in slots) / len(slots)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=9)
    ap.add_argument("--teams", nargs="+", type=int, default=list(TEAMS))
    ap.add_argument("--sigma", type=float, default=.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import cma
    OUT.mkdir(parents=True, exist_ok=True)
    x0 = normalise_defaults()
    popsize = 4 + int(3 * np.log(len(GENES)))
    es = cma.CMAEvolutionStrategy(x0, args.sigma, {"bounds": [0, 1], "popsize": popsize, "seed": args.seed})
    print(f"CMA-ES: {len(GENES)} genes, popsize {popsize}, teams {args.teams}", flush=True)

    workers = max(1, int(np.sqrt(args.parallel)))
    per = max(1, args.parallel // workers)
    env = VecWasmEnv(n_workers=workers, matches_per_worker=per)
    slots_per_wave = workers * per
    print(f"{slots_per_wave} parallel slots", flush=True)

    best_x, best_f = x0.copy(), None
    log_path = OUT / "evolve.log"
    t_start = time.time()
    for generation in range(1, args.generations + 1):
        xs = es.ask()
        seed = generation % 3 + 1
        fitness = [0.] * len(xs)
        # Evaluate candidates in waves of parallel slots; each candidate needs len(teams) slots.
        jobs = [(i, x) for i, x in enumerate(xs)]
        while jobs:
            wave, jobs = jobs[:slots_per_wave // len(args.teams)], jobs[slots_per_wave // len(args.teams):]
            for k, (i, x) in enumerate(wave):
                slots = list(range(k * len(args.teams), (k + 1) * len(args.teams)))
                try:
                    fitness[i] = evaluate(env, x, args.teams, seed, slots)
                except Exception as exc:
                    print(f"  candidate {i} crashed ({exc}); fitness 0", flush=True)
                    fitness[i] = 0.
        es.tell(xs, [-f for f in fitness])
        order = np.argsort(fitness)[::-1]
        top = fitness[order[0]]
        if best_f is None or top > best_f:
            best_f, best_x = top, xs[order[0]].copy()
            BEST.write_text(json.dumps(dict(fitness=best_f, params=denormalise(best_x), generation=generation), indent=2))
        mean_f = float(np.mean(fitness))
        print(f"gen {generation}: best {top:.0f} mean {mean_f:.0f} ever {best_f:.0f} "
              f"sigma {es.sigma:.3f} elapsed {(time.time() - t_start) / 60:.0f}m", flush=True)
        with log_path.open("a") as log:
            log.write(json.dumps(dict(generation=generation, seed=seed, best=top, mean=mean_f,
                                      ever=best_f, params=denormalise(xs[order[0]]))) + "\n")
        if es.stop():
            print("CMA converged", flush=True)
            break
    print(f"best fitness {best_f:.0f}: {json.dumps(denormalise(best_x))}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
