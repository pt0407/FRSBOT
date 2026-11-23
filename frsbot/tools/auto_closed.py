"""Closed-loop auto test: the driver follows the aligned pro route for 0-20 s, with a Score goal at the scoring spot when the pro sheds FUEL, then hands off to ImAI.

Team 2910, CLHNB666 replay. Blue uses the pro start; red uses the 180-degree mirror.
Usage: PYTHONPATH=. python -m frsbot.tools.auto_closed
"""
from frsbot.bot.driver import Driver
from frsbot.bot.planner import Goal, Planner
from frsbot.core.specs import load_specs
from frsbot.sim.policies import ImAIPolicy
from frsbot.sim.vec_env import RobotSlot, VecWasmEnv
AUTO_END = 20.0
F, Y = 16.540988, 8.069326
spec = {s.team: s for s in load_specs()}[2910]
S = json.load(open("data/telemetry/ljqj1ukg3mnpcxvl.json"))["samples"]
P = [(s["tick"] / 60, s["robots"][0]["pos"][0], s["robots"][0]["pos"][2]) for s in S if s["tick"] / 60 <= AUTO_END + 1]
HOP = [(s["tick"] / 60, s["robots"][0]["hopperFuel"]) for s in S]
r0 = S[0]["robots"][0]
blue = dict(fx=r0["pos"][0] + 8.27, fy=4.03 - r0["pos"][2], headingDeg=180 - math.degrees(math.pi - 2*math.atan2(r0["quat"][1], r0["quat"][3])))
red = dict(fx=F - blue["fx"], fy=Y - blue["fy"], headingDeg=blue["headingDeg"] + 180)
class Closed(ImAIPolicy):
    def __init__(self, spec):
        super().__init__(spec); self.driver = Driver(spec); self.planner = Planner(spec)
    def act(self, state, info=None):
        if state.t >= AUTO_END:
            return super().act(state, info)
        robot = state.robots[state.me]
        j = min(range(len(HOP)), key=lambda k: abs(HOP[k][0] - state.t))
        d = HOP[min(len(HOP) - 1, j + 30)][1] - HOP[j][1]
        if d <= -2 and robot.hopper > 0:
            goal = Goal("Score", self.planner.score_spot(state))
        else:
            i = min(range(len(P)), key=lambda k: abs(P[k][0] - state.t))
            x, z = P[i][1], P[i][2]
            if robot.alliance == "red":
                x, z = F - x, Y - z
            goal = Goal("Stage", (x, z))
        inp = self.driver.act(goal, state)
        inp.intake = d >= 2
        inp.score = goal.kind == "Score"
        return inp
rows = []
for seed in (1, 2, 3):
    for alliance in ("blue", "red"):
        start = blue if alliance == "blue" else red
        pol = Closed(spec)
        with VecWasmEnv(n_workers=1, matches_per_worker=1) as env:
            env.reset([dict(robots=[RobotSlot(2910, alliance, start=start, policy=pol)], seed=seed, auto="none", event="event", hp=True, unload=True)])
            at20 = env.rollout(AUTO_END, ticks=3)[0].scores[alliance]
            final = env.rollout(166.0, ticks=3)[0].scores[alliance]
        rows.append((alliance, seed, at20, final))
print(f"closed-loop auto with scoring: final mean {st.mean(r[3] for r in rows):.1f}, @20s mean {st.mean(r[2] for r in rows):.1f}", flush=True)
print("per case (alliance, seed, @20s, final):", rows, flush=True)
