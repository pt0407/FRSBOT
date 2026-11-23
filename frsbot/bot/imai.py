"""ImAI's GameState -> Inputs interface; no backend dependencies."""
import math

from frsbot.bot.driver import Driver
from frsbot.bot.tuning import PARAMS
from frsbot.bot.planner import Goal, Planner
from frsbot.core.inputs import Inputs
from frsbot.core.state import GameState, Phase


class ImAI:
    def __init__(self, spec):
        self.planner = Planner(spec)
        self.driver = Driver(spec)
        self.goal = None
        self.next_plan = -math.inf
        self.collect_until = -math.inf
        self.collect_deadline = -math.inf
        self.collect_target = 0

    def act(self, state: GameState) -> Inputs:
        if state.phase in (Phase.COUNTDOWN, Phase.SETTLE, Phase.DONE):
            self.goal = self.planner.plan(state)
            return Inputs(field_relative=True)
        robot = state.robots[state.me]
        if self.collect_target:
            collecting = state.t < self.collect_until or (robot.hopper < self.collect_target and state.t < self.collect_deadline)
            if collecting and state.t + self.planner.score_time(state) + 5 < 163:
                clusters = self.planner.clusters(state)
                if clusters:
                    self.goal = Goal("Collect", clusters[0][1])
                    self.planner.goal = self.goal
                    return self.driver.act(self.goal, state)
            self.collect_target = 0
        arrived = self.goal is not None and self.goal.kind == "Collect" and math.hypot(robot.x - self.goal.point[0], robot.z - self.goal.point[1]) < .3
        if state.t >= self.next_plan or arrived or not self.planner.valid(self.goal, state):
            self.goal = self.planner.plan(state)
            self.next_plan = state.t + PARAMS["replan_s"]
        if state.t >= 160 and self.goal.kind != "Score":
            return Inputs(field_relative=True)
        inputs = self.driver.act(self.goal, state)
        if self.driver.abandon and state.t + self.planner.score_time(state) + 5 < 163:
            self.collect_until = state.t + 2
            self.collect_deadline = state.t + 8
            self.collect_target = min(self.planner.capacity(state), max(PARAMS["collect_min"], PARAMS["collect_frac"] * self.planner.capacity(state)))
            self.planner.goal = None
            self.next_plan = -math.inf
        return inputs
