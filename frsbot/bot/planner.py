"""Window-aware collection, scoring and staging goals."""
import math
from dataclasses import dataclass

import numpy as np

from frsbot.core.field import HUB_CENTRES
from frsbot.bot.tuning import PARAMS
from frsbot.core.specs import RobotSpec
from frsbot.core.state import GameState, Phase
from frsbot.bot.calibration import profile

FIELD_X, FIELD_Z = 8.270494, 4.034663
SCORING_X = 4.3  # Own-zone line is |x|=4.2418 m; keep the centre 58 mm inside it.


def angle_error(target, actual):
    return (target - actual + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Goal:
    kind: str
    point: tuple[float, float]


class Planner:
    def __init__(self, spec: RobotSpec):
        self.spec = spec
        self.goal = None
        self.profile = profile(spec.team)
        self.avoid_points = []
        self.avoid_slots = []
        self.score_offset = 0.

    def capacity(self, state):
        # hopperFuel/hopperCapacity are storage counts, not the UI's feed-inclusive count.
        if PARAMS["use_total_capacity"]:
            return self.spec.total_capacity
        return state.robots[state.me].capacity or self.spec.total_capacity

    def bounds(self):
        return self.spec.min_range or min(self.spec.hub_shot["distance"]), self.spec.max_range or max(self.spec.hub_shot["distance"])

    def hub(self, state):
        return HUB_CENTRES[state.robots[state.me].alliance][:2]

    def score_spot(self, state, angle_offset=0):
        robot = state.robots[state.me]
        cx, cz = self.hub(state)
        low, high = self.bounds()
        radius = max(low + .3, min(high - .5, (low + high) / 2))
        heading = math.atan2(robot.z - cz, robot.x - cx) + angle_offset + self.score_offset
        margin = math.hypot(self.spec.bumper_length, self.spec.bumper_width) / 2 + .1
        candidates = []
        for delta in [0, *[s * i * math.pi / 36 for i in range(1, 37) for s in (-1, 1)]]:
            x, z = cx + radius * math.cos(heading + delta), cz + radius * math.sin(heading + delta)
            own_side = x < -SCORING_X if robot.alliance == "blue" else x > SCORING_X
            if own_side and abs(x) <= FIELD_X - margin and abs(z) <= FIELD_Z - margin:
                candidates.append((x, z))
        if self.profile.get("score_arc_degrees"):
            back = math.pi if robot.alliance == "blue" else 0
            candidates = [(cx + radius * math.cos(back + math.radians(deg)),
                           cz + radius * math.sin(back + math.radians(deg)))
                          for deg in self.profile["score_arc_degrees"]]
        preferred = (cx + radius * math.cos(heading), cz + radius * math.sin(heading))
        if self.avoid_slots:
            free = [p for p in candidates if all(math.dist(p, slot) >= PARAMS["avoid_slot_m"] for slot in self.avoid_slots)]
            if free:
                candidates = free
        return min(candidates, key=lambda p: math.dist(p, preferred))

    def fuel_points(self, state):
        pos = state.fuel.pos
        return pos[(state.fuel.where == 0) & (pos[:, 1] < .2)][:, [0, 2]]

    def score_time(self, state):
        robot = state.robots[state.me]
        target = self.score_spot(state)
        position = (robot.x, robot.z)
        distance = math.dist(position, target)
        for cx, _, _ in HUB_CENTRES.values():
            if (position[0] - cx) * (target[0] - cx) < 0:
                side = -1 if position[0] < cx else 1
                lane = min((-1.8, 1.8), key=lambda z: abs(position[1] - z) + abs(target[1] - z))
                entry, exit = (cx + side * 1.65, lane), (cx - side * 1.65, lane)
                distance = math.dist(position, entry) + math.dist(entry, exit) + math.dist(exit, target)
        travel = distance / self.spec.max_speed
        if distance > .3:
            travel += .5
        return travel + self.profile.get("startup_seconds", 1.5) + robot.hopper / self.profile.get("drain_rate", self.spec.fire_rate)

    def clusters(self, state):
        points = self.fuel_points(state)
        if not len(points):
            return []
        robot = state.robots[state.me]
        cells, inverse = np.unique(np.floor(points / .5).astype(int), axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        centres = np.column_stack([np.bincount(inverse, weights=points[:, i]) / counts for i in (0, 1)])
        sweep = max(PARAMS["sweep_min_m"], self.spec.bumper_width * PARAMS["sweep_scale"])
        balls = (np.sum((centres[:, None, :] - points[None, :, :]) ** 2, axis=2) <= sweep ** 2).sum(axis=1)
        dx, dz = centres[:, 0] - robot.x, centres[:, 1] - robot.z
        travel = np.hypot(dx, dz) / self.spec.max_speed
        turn = np.abs((np.arctan2(dz, dx) - robot.a + np.pi) % (2 * np.pi) - np.pi) / self.spec.turn_rate
        values = balls / (travel + turn + PARAMS["cluster_bias"])
        own = self.score_spot(state)
        if state.hubs[robot.alliance].active:
            values /= 1 + .08 * np.hypot(centres[:, 0] - own[0], centres[:, 1] - own[1])
        margin = math.hypot(self.spec.bumper_length, self.spec.bumper_width) / 2 + .1
        result = []
        for point, value in zip(centres, values):
            x, z = map(float, point)
            if abs(x) > FIELD_X - margin or abs(z) > FIELD_Z - margin:
                continue
            clearance = min(FIELD_X - abs(x), FIELD_Z - abs(z))
            value /= 1 + .15 / max(.1, clearance)
            if any(abs(x - cx) < 1.3 and abs(z - cz) < 1.3 for cx, cz, _ in HUB_CENTRES.values()):
                continue
            if any(math.hypot(x - cx, z - cz) < 2 for cx, cz, _ in HUB_CENTRES.values()):
                value *= PARAMS["hub_near_scale"]
            if self.goal and self.goal.kind == "Collect" and math.hypot(x - self.goal.point[0], z - self.goal.point[1]) < .8:
                value *= PARAMS["goal_commit"]
            result.append((value, (x, z)))
        if self.avoid_points:
            free = [(value, point) for value, point in result
                    if all(math.dist(point, other) >= PARAMS["avoid_point_m"] for other in self.avoid_points)]
            if any(value >= 2 for value, _ in free):
                result = free
        return sorted(result, reverse=True)

    def valid(self, goal, state):
        if goal is None:
            return False
        robot = state.robots[state.me]
        if goal.kind == "Collect":
            points = self.fuel_points(state)
            return robot.hopper < self.capacity(state) and bool(len(points) and np.any(np.linalg.norm(points - goal.point, axis=1) < .85))
        if goal.kind == "Score":
            return robot.hopper > 0 and state.hubs[robot.alliance].active
        return robot.hopper > 0 and not state.hubs[robot.alliance].active

    def plan(self, state: GameState) -> Goal:
        robot = state.robots[state.me]
        hub = state.hubs[robot.alliance]
        spot = self.score_spot(state)
        if state.phase in (Phase.COUNTDOWN, Phase.SETTLE, Phase.DONE):
            return Goal("Stage", (robot.x, robot.z))
        low, high = self.bounds()
        distance = math.hypot(robot.x - self.hub(state)[0], robot.z - self.hub(state)[1])
        own_side = robot.x < -SCORING_X if robot.alliance == "blue" else robot.x > SCORING_X
        score_time = self.score_time(state)
        if hub.active and robot.hopper and 163 - state.t <= score_time + 3:
            if state.t >= 160 and own_side and low <= distance <= high:
                spot = (robot.x, robot.z)
            return Goal("Score", spot)
        if robot.hopper == 0 and 163 - state.t <= score_time + 2:
            return Goal("Stage", (robot.x, robot.z))
        if state.t >= 160:
            return Goal("Score", (robot.x, robot.z)) if robot.hopper and hub.active and own_side and low <= distance <= high else Goal("Stage", (robot.x, robot.z))
        clusters = self.clusters(state)
        if hub.active and robot.hopper and (state.t < PARAMS["early_score_until"] or hub.seconds_to_toggle <= score_time + 1 or
                robot.hopper >= PARAMS["active_score_frac"] * self.capacity(state) or not clusters or clusters[0][0] < 2):
            goal = Goal("Score", spot)
        elif not hub.active and robot.hopper and (robot.hopper >= PARAMS["stage_full_frac"] * self.capacity(state) or hub.seconds_to_toggle < score_time + PARAMS["arrive_margin_s"]):
            goal = Goal("Stage", spot)
        elif clusters:
            goal = Goal("Collect", clusters[0][1])
        else:
            goal = Goal("Stage", spot if robot.hopper else (robot.x, robot.z))
        if self.goal and self.goal.kind == "Stage" and robot.hopper and hub.active:
            goal = Goal("Score", spot)
        # Finish unloading instead of changing targets as soon as one ball leaves.
        if self.goal and self.goal.kind == "Score" and self.valid(self.goal, state):
            goal = self.goal
        self.goal = goal
        return goal
