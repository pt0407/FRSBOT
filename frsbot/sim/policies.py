"""Pluggable policies for robots that get no explicit action in a WasmEnv/VecWasmEnv step.

`act(state, info)` returns `Inputs` (applied raw), a planner `Goal` (the JS driver in the engine process drives it
every tick) or `None` (idle). `info` holds the JS driver's flags for that robot (abandoned, recovering, stage).
A policy may also define `act_batch(states, infos)`; VecWasmEnv then calls it once per step for all of its robots.
"""
import json
import math
from pathlib import Path

from frsbot.bot.imai import ImAI
from frsbot.bot.planner import Goal


class IdlePolicy:
    def act(self, state, info=None):
        return None


class ImAIPolicy:
    """Rule ImAI with the Python driver: exact reference behaviour, one Python act per env step."""

    def __init__(self, spec, shoot_while_collecting=None):
        self.bot = ImAI(spec)
        if shoot_while_collecting is not None:
            self.bot.driver.shoot_while_collecting = shoot_while_collecting

    def act(self, state, info=None):
        return self.bot.act(state)


class _GoalDriver:
    """Stands in for ImAI.driver: returns the goal for the JS driver; `abandon` comes from the engine's flags."""

    def __init__(self):
        self.abandon = False
        self.recovering = False
        self.recovery_stage = 0

    def act(self, goal, state):
        return goal


class GoalImAIPolicy:
    """Rule ImAI with the planner in Python and the driver in the engine process (PERF.md step 1).

    The JS driver runs every tick, so decisions can be sparse (e.g. every 6 ticks) without coarser control."""

    def __init__(self, spec):
        self.bot = ImAI(spec)
        self.bot.driver = _GoalDriver()

    def act(self, state, info=None):
        if info:
            self.bot.driver.abandon = info["abandoned"]
            self.bot.driver.recovering = info["recovering"]
            self.bot.driver.recovery_stage = info["stage"]
        return self.bot.act(state)


class ReplayPolicy:
    """Follows a recorded robot's poses (data/telemetry/*.json) through the JS driver.

    Collects while the recording's hopper is rising, scores while it falls, otherwise stages at the next pose."""

    def __init__(self, recording, robot: int = 0, mirror: bool = False):
        if isinstance(recording, (str, Path)):
            recording = json.loads(Path(recording).read_text())
        self.samples = [(s["tick"] / 60, s["robots"][robot]) for s in recording["samples"] if len(s["robots"]) > robot]
        if not self.samples:
            raise ValueError(f"recording has no samples for robot {robot}")
        self.sign = -1 if mirror else 1
        self.index = 0

    def act(self, state, info=None):
        while self.index + 1 < len(self.samples) and self.samples[self.index][0] <= state.t:
            self.index += 1
        t, target = self.samples[self.index]
        previous = self.samples[max(0, self.index - 1)][1]
        point = (self.sign * float(target["pos"][0]), self.sign * float(target["pos"][2]))
        if not all(math.isfinite(v) for v in point):
            return None
        rising, falling = target["hopperFuel"] > previous["hopperFuel"], target["hopperFuel"] < previous["hopperFuel"]
        return Goal("Collect" if rising else "Score" if falling else "Stage", point)
