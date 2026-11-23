"""Auto mimic test: replay the recovered pro controls for the first 20 s, then hand off to ImAI.

Team 2910, CLHNB666 replay. Blue uses the pro start; red uses the 180-degree mirror.
Usage: PYTHONPATH=. python -m frsbot.tools.auto_mimic  (prints final and 20 s scores; no bot defaults change)
"""
import bisect
import json
import math
import statistics as st

from frsbot.core.specs import load_specs
from frsbot.core.inputs import Inputs
from frsbot.imitation.dataset import examples
from frsbot.sim.policies import ImAIPolicy
from frsbot.sim.vec_env import RobotSlot, VecWasmEnv
AUTO_END = 20.0
F, Y = 16.540988, 8.069326   # field length and width (engine frame), mirror pivot
spec = {s.team: s for s in load_specs()}[2910]
rec0 = json.load(open("data/telemetry/ljqj1ukg3mnpcxvl.json"))["samples"][0]["robots"][0]
blue_start = dict(fx=rec0["pos"][0] + 8.27, fy=4.03 - rec0["pos"][2],
                  headingDeg=180 - math.degrees(math.pi - 2 * math.atan2(rec0["quat"][1], rec0["quat"][3])))
def mirror(s): return dict(fx=F - s["fx"], fy=Y - s["fy"], headingDeg=s["headingDeg"] + 180)
TABLE = sorted((st_.t, float(i.x), float(i.y), float(i.turn), bool(i.intake), bool(i.score))
               for st_, i, _, _ in examples("data/imitation/solo-ljqj1ukg3mnpcxvl.npz")
               if st_.robots[st_.me].team == 2910 and st_.t < AUTO_END)
TS = [r[0] for r in TABLE]
class Mimic(ImAIPolicy):
    def __init__(self, spec, sign):
        super().__init__(spec); self.sign = sign
    def act(self, state, info=None):
        if state.t < AUTO_END:
            k = min(max(bisect.bisect_left(TS, state.t), 0), len(TABLE) - 1)
            _, x, y, turn, it, sc = TABLE[k]
            return Inputs(x=x, y=y, turn=turn, intake=it, score=sc, field_relative=True)
        return super().act(state, info)
rows = []
for seed in (1, 2, 3):
    for alliance in ("blue", "red"):
        sign = 1 if alliance == "blue" else -1
        start = blue_start if alliance == "blue" else mirror(blue_start)
        with VecWasmEnv(n_workers=1, matches_per_worker=1) as env:
            env.reset([dict(robots=[RobotSlot(2910, alliance, start=start, policy=Mimic(spec, sign))], seed=seed, auto="none", event="event", hp=True, unload=True)])
            at20 = env.rollout(AUTO_END, ticks=3)[0].scores[alliance]
            final = env.rollout(166.0, ticks=3)[0].scores[alliance]
        rows.append((alliance, seed, at20, final))
print(f"mimic auto, both alliances: final mean {st.mean(r[3] for r in rows):.1f}, @20s mean {st.mean(r[2] for r in rows):.1f}", flush=True)
print("per case (alliance, seed, @20s, final):", rows, flush=True)
