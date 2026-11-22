"""Obstacle-aware, braking-limited field-relative pursuit."""
import heapq
import math

import numpy as np

from frsbot.bot.planner import FIELD_X, FIELD_Z, SCORING_X, Goal, Planner, angle_error
from frsbot.core.field import HUB_CENTRES
from frsbot.core.inputs import Inputs


def inverse_deadzone(value):
    return math.copysign(.12 + .88 * min(1, abs(value)), value) if value else 0.0


def intersects(a, b, box):
    lower, upper = 0.0, 1.0
    for p, q, lo, hi in zip(a, b, (box[0], box[2]), (box[1], box[3])):
        d = q - p
        if abs(d) < 1e-9:
            if not lo < p < hi:
                return False
        else:
            x, y = sorted(((lo - p) / d, (hi - p) / d))
            lower, upper = max(lower, x), min(upper, y)
    return lower < upper and lower < 1 and upper > 0


class Driver:
    def __init__(self, spec):
        self.spec = spec
        self.planner = Planner(spec)
        self.closed_since = None
        self.recovering = False
        self.crossing = None

    def waypoint(self, start, target):
        if self.crossing is None:
            for cx, _, _ in HUB_CENTRES.values():
                if (start[0] - cx) * (target[0] - cx) < 0:
                    side = -1 if start[0] < cx else 1
                    lane = min((-1.8, 1.8), key=lambda z: abs(start[1] - z) + abs(target[1] - z))
                    self.crossing = [(cx + side * 1.65, lane), (cx - side * 1.65, lane)]
                    break
        if self.crossing:
            if math.dist(start, self.crossing[0]) < .4:
                self.crossing.pop(0)
            if self.crossing:
                return self.crossing[0]
            self.crossing = None
        pad = math.hypot(self.spec.bumper_length, self.spec.bumper_width) / 2 + .12
        boxes = [(cx - .597 - pad, cx + .597 + pad, cz - .597 - pad, cz + .597 + pad)
                 for cx, cz, _ in HUB_CENTRES.values()]
        for left, right, bottom, top in boxes:
            if left < start[0] < right and bottom < start[1] < top:
                return min([(left - .08, start[1]), (right + .08, start[1]),
                            (start[0], bottom - .08), (start[0], top + .08)], key=lambda p: math.dist(start, p))
        nodes = [start, target]
        for left, right, bottom, top in boxes:
            nodes.extend([(x, z) for x in (left - .05, right + .05) for z in (bottom - .05, top + .05)])
        queue, best = [(0, 0, [])], {0: 0}
        while queue:
            cost, index, route = heapq.heappop(queue)
            if index == 1:
                return nodes[route[0]] if route else target
            if cost > best.get(index, math.inf):
                continue
            for j, point in enumerate(nodes):
                if j == index or any(intersects(nodes[index], point, box) for box in boxes):
                    continue
                new_cost = cost + math.dist(nodes[index], point)
                if new_cost < best.get(j, math.inf):
                    best[j] = new_cost
                    heapq.heappush(queue, (new_cost, j, [*route, j]))
        return start

    def act(self, goal: Goal, state):
        robot = state.robots[state.me]
        active = state.hubs[robot.alliance].active
        hub = self.planner.hub(state)
        low, high = self.planner.bounds()
        distance_hub = math.hypot(robot.x - hub[0], robot.z - hub[1])
        shooting = goal.kind == "Score" and active and robot.hopper > 0
        own_side = robot.x < -SCORING_X if robot.alliance == "blue" else robot.x > SCORING_X
        at_spot = math.dist((robot.x, robot.z), goal.point) < .7
        if shooting and own_side and at_spot and robot.gate is not None and not robot.gate.get("open", False):
            if self.closed_since is None:
                self.closed_since = state.t
            self.recovering = state.t - self.closed_since > 1.5
        else:
            self.closed_since = None
            self.recovering = False
        target = self.planner.score_spot(state) if self.recovering else goal.point
        point = self.waypoint((robot.x, robot.z), target)
        dx, dz = point[0] - robot.x, point[1] - robot.z
        distance = math.hypot(dx, dz)
        direction = math.atan2(dz, dx)
        arriving = distance < .3
        final_waypoint = math.dist(point, target) < .01
        vmax = self.spec.max_speed
        # Conservative stopping distance also leaves room for the engine's velocity response.
        speed = min(vmax, 2 * distance, math.sqrt(2 * self.spec.accel * .4 * max(0, distance - .12)))
        if goal.kind == "Collect":
            if distance < 1:
                speed = min(speed, vmax * .6)
            speed *= max(.05, math.cos(angle_error(direction, robot.a)))
        if arriving and final_waypoint:
            speed = 0
        elif arriving:
            speed = max(speed, .35)
        score_speed = self.spec.max_shot_speed or self.spec.brain_equivalents.get("scoreSpeedTol", .15)
        if shooting and final_waypoint and low <= distance_hub <= high and distance < .6:
            speed = min(speed, score_speed * .5)
        vx, vz = speed * math.cos(direction), speed * math.sin(direction)
        margin = math.hypot(self.spec.bumper_length, self.spec.bumper_width) / 2 + .1
        if robot.x * vx > 0 and abs(robot.x + vx * .3) > FIELD_X - margin:
            vx = 0
        if robot.z * vz > 0 and abs(robot.z + vz * .3) > FIELD_Z - margin:
            vz = 0
        norm = math.hypot(vx, vz) / vmax
        stick = inverse_deadzone(norm)
        sign = 1 if robot.alliance == "blue" else -1
        x = sign * stick * vz / math.hypot(vx, vz) if norm else 0
        y = sign * stick * vx / math.hypot(vx, vz) if norm else 0
        aim = math.atan2(hub[1] - robot.z, hub[0] - robot.x) + math.pi if goal.kind != "Collect" else direction
        turn = inverse_deadzone(max(-1, min(1, -4 * angle_error(aim, robot.a) / self.spec.turn_rate)))
        # Generic score controls its own aim; recovery explicitly faces the hub at rest.
        if shooting and not self.recovering:
            turn = 0
        points = self.planner.fuel_points(state)
        path_fuel = False
        if len(points):
            rel = points - [robot.x, robot.z]
            ahead = rel @ np.array([math.cos(robot.a), math.sin(robot.a)])
            side = rel @ np.array([-math.sin(robot.a), math.cos(robot.a)])
            path_fuel = bool(np.any((ahead > 0) & (ahead < 1.2) & (abs(side) < self.spec.bumper_width / 2)))
        intake = robot.hopper < self.planner.capacity(state) and (goal.kind == "Collect" or path_fuel)
        score = shooting and own_side and at_spot and low <= distance_hub <= high and math.hypot(robot.vx, robot.vz) <= score_speed
        return Inputs(x=x, y=y, turn=turn, intake=intake, score=score, field_relative=True)
