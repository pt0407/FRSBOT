"""ImAI's GameState -> Inputs interface; no backend dependencies."""
import math

from frsbot.bot.driver import Driver
from frsbot.bot.planner import Planner
from frsbot.core.inputs import Inputs
from frsbot.core.state import GameState, Phase


class ImAI:
    def __init__(self, spec):
        self.planner = Planner(spec)
        self.driver = Driver(spec)
        self.goal = None
        self.next_plan = -math.inf

    def act(self, state: GameState) -> Inputs:
        if state.phase in (Phase.COUNTDOWN, Phase.SETTLE, Phase.DONE):
            self.goal = self.planner.plan(state)
            return Inputs(field_relative=True)
        robot = state.robots[state.me]
        arrived = self.goal is not None and self.goal.kind == "Collect" and math.hypot(robot.x - self.goal.point[0], robot.z - self.goal.point[1]) < .3
        if state.t >= self.next_plan or arrived or not self.planner.valid(self.goal, state):
            self.goal = self.planner.plan(state)
            self.next_plan = state.t + .5
        if state.t >= 160 and self.goal.kind != "Score":
            return Inputs(field_relative=True)
        return self.driver.act(self.goal, state)
