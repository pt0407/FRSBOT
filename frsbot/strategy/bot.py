"""Best-response multiplayer wrapper; the solo ImAI policy remains untouched."""
import math
from dataclasses import replace

import numpy as np

from frsbot.bot.imai import ImAI
from frsbot.bot.planner import Goal, Planner, angle_error
from frsbot.core.field import HUB_CENTRES
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.core.state import Phase
from frsbot.strategy import fouls
from frsbot.strategy.driver import StrategicDriver, safe_point
from frsbot.strategy.intent import ALLIANCE_ZONE_EDGE
from frsbot.strategy.roles import Role, StrategyLayer


NEW_GOAL_KINDS = ("Defend", "Contest", "Feed")


class StrategicImAI:
    def __init__(self, spec):
        self.spec = spec
        self.specs = {s.team: s for s in load_specs()}
        self.specs[spec.team] = spec
        self.scorer = ImAI(spec)
        self.layer = StrategyLayer(spec, self.specs)
        self.driver = StrategicDriver(spec, self.specs)
        self.planners = {team: Planner(s) for team, s in self.specs.items()}
        self.goal = None
        self._feeding = False

    def _cluster(self, state, robot, neutral=True):
        index = next(i for i, r in enumerate(state.robots) if r.id == robot.id)
        planner = self.scorer.planner if index == state.me else self.planners[robot.team]
        clusters = planner.clusters(replace(state, me=index))
        points = [p for _, p in clusters if not neutral or abs(p[0]) < ALLIANCE_ZONE_EDGE]
        if not points:
            return None
        speed = math.hypot(robot.vx, robot.vz)
        if speed > .1:
            direction = np.array([robot.vx, robot.vz]) / speed
            rays = []
            for point in points:
                delta = np.array(point) - [robot.x, robot.z]
                along = delta @ direction
                across = abs(delta[0] * direction[1] - delta[1] * direction[0])
                if 0 < along <= 3 and across <= .8:
                    rays.append((along, across, point))
            if rays:
                return min(rays)[2]
        return min(points, key=lambda p: math.dist(p, (robot.x, robot.z)))

    def screen_point(self, state, target):
        loaded = target.hopper > 0 or self.layer._probability(target, "Score") > .5
        if loaded:
            index = next(i for i, r in enumerate(state.robots) if r.id == target.id)
            destination = self.planners[target.team].score_spot(replace(state, me=index))
        else:
            destination = self._cluster(state, target, neutral=False)
        if destination is None or math.dist(destination, (target.x, target.z)) < .1:
            destination = HUB_CENTRES[target.alliance][:2]
        delta = np.array(destination) - [target.x, target.z]
        point = np.array([target.x, target.z]) + delta / max(np.linalg.norm(delta), 1e-6)
        return safe_point(point, self.spec)

    def _feed_goal(self, state, immediate=False):
        me = state.robots[state.me]
        if me.hopper == 0:
            self._feeding = False
        cluster = self._cluster(state, me)
        if immediate or me.hopper >= .6 * self.scorer.planner.capacity(state) or (me.hopper and cluster is None):
            self._feeding = True
        if self._feeding and me.hopper:
            sign = -1 if me.alliance == "blue" else 1
            point = safe_point((sign * (ALLIANCE_ZONE_EDGE - .9), math.copysign(2.5, me.z)), self.spec)
            return Goal("Feed", point)
        return Goal("Contest", cluster) if cluster is not None else None

    def _score(self, state):
        planner = self.scorer.planner
        goal = self.scorer.goal
        if goal is not None:
            if goal.kind == "Collect" and planner.avoid_points and any(math.dist(goal.point, p) < .8 for p in planner.avoid_points):
                clusters = planner.clusters(state)
                if clusters and clusters[0][0] >= 2 and math.dist(clusters[0][1], goal.point) >= .8:
                    self.scorer.next_plan = -math.inf
            elif goal.kind in ("Score", "Stage") and (planner.avoid_slots or planner.score_offset):
                point = planner.score_spot(state)
                if math.dist(point, goal.point) > .1:
                    self.scorer.goal = planner.goal = Goal(goal.kind, point)
                    self.scorer.driver.recovery_point = None
                    self.scorer.driver.crossing = None
        inputs = self.scorer.act(state)
        self.goal = self.scorer.goal
        return self.driver.guard(inputs, self.goal, state)

    def act(self, state) -> Inputs:
        me = state.robots[state.me]
        if not any(r.id != me.id and r.alliance == me.alliance for r in state.robots):
            inputs = self.scorer.act(state)
            self.goal = self.scorer.goal
            return inputs
        self.layer.observe(state)
        planner = self.scorer.planner
        planner.avoid_points = self.layer.teammate_targets(state)
        planner.avoid_slots = self.layer.teammate_slots(state)
        planner.score_offset = 0.
        if state.phase in (Phase.COUNTDOWN, Phase.SETTLE, Phase.DONE):
            return self._score(state)
        role = self.layer.role(state)
        target = next((r for r in state.robots if r.id == role.target and r.alliance != me.alliance), None)
        if role.kind == "Defender" and (state.phase == Phase.AUTO or state.t >= 160 or target is None or
                (state.phase == Phase.ENDGAME and math.dist((target.x, target.z), fouls.TOWER_CENTRES[target.alliance]) < fouls.TOWER_AVOID_DISTANCE)):
            role = Role("Scorer")
        if role.kind == "Contester" and state.phase == Phase.AUTO:
            role = Role("Scorer")
        goal = None
        if role.kind == "Defender":
            goal = Goal("Defend", self.screen_point(state, target))
        elif role.kind == "Contester" and me.hopper < self.scorer.planner.capacity(state):
            opponents = [r for r in state.robots if r.alliance != me.alliance]
            if opponents:
                collector = max(opponents, key=lambda r: (self.layer._probability(r, "Collect"), -r.id))
                point = self._cluster(state, collector)
                if point is not None:
                    goal = Goal("Contest", point)
        elif role.kind == "Feeder":
            goal = self._feed_goal(state)
        if goal is None and role.kind == "Scorer":
            defender_id = self.layer.defended_by(state)
            if defender_id is not None:
                evasion = self.layer.evasion(state)
                if evasion == "feed":
                    goal = self._feed_goal(state, immediate=me.hopper > 0)
                elif evasion == "alternate_slot":
                    defender = next(r for r in state.robots if r.id == defender_id)
                    hub = planner.hub(state)
                    my_angle = math.atan2(me.z - hub[1], me.x - hub[0])
                    their_angle = math.atan2(defender.z - hub[1], defender.x - hub[0])
                    planner.score_offset = math.copysign(math.pi / 6, angle_error(my_angle, their_angle))
        if goal is None or (state.phase == Phase.AUTO and goal.kind in ("Defend", "Contest")):
            return self._score(state)
        self.goal = goal
        return self.driver.act(goal, state, target if goal.kind == "Defend" else None)
