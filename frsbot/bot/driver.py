"""Obstacle-aware, braking-limited field-relative pursuit."""
import heapq
import math

import numpy as np

from frsbot.bot.planner import FIELD_X, FIELD_Z, SCORING_X, Goal, Planner, angle_error
from frsbot.bot.tuning import PARAMS
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
        self.recovering = False
        self.crossing = None
        self.progress_at = None
        self.open_since = None
        self.previous_hopper = None
        self.recovery_stage = 0
        self.recovery_point = None
        self.wiggle_until = 0
        self.abandon = False
        self.shoot_while_collecting = self.planner.profile.get("shoot_while_collecting", False)
        self.reaim_until = 0

    def progress(self, goal, state):
        robot = state.robots[state.me]
        self.abandon = False
        scoring = goal.kind == "Score" and state.hubs[robot.alliance].active and robot.hopper > 0
        if not scoring:
            self.progress_at = None
            self.open_since = None
            self.previous_hopper = robot.hopper
            self.recovery_stage = 0
            self.recovery_point = None
            self.recovering = False
            return goal
        point = self.recovery_point or goal.point
        if self.previous_hopper is None or robot.hopper < self.previous_hopper:
            self.progress_at = state.t
            self.open_since = None
            self.recovery_stage = 0
        self.previous_hopper = robot.hopper
        if math.dist((robot.x, robot.z), point) > .7:
            self.progress_at = state.t
        if self.progress_at is None:
            self.progress_at = state.t
        opened = robot.gate is not None and robot.gate.get("open", False)
        if not opened:
            self.open_since = None
        elif self.open_since is None:
            self.open_since = state.t
        stalled = state.t - self.progress_at >= 2 or (self.open_since is not None and state.t - max(self.open_since, self.progress_at) >= 1)
        if state.t >= self.wiggle_until and stalled:
            self.recovery_stage += 1
            self.progress_at = state.t
            if self.recovery_stage == 1:
                self.reaim_until = state.t + .6
                self.progress_at = self.reaim_until
            elif self.recovery_stage == 2:
                point = self.planner.score_spot(state, math.pi / 6)
                if math.dist(point, (robot.x, robot.z)) < .5:
                    point = self.planner.score_spot(state, -math.pi / 6)
                self.recovery_point = point
                self.crossing = None
            elif self.recovery_stage == 3:
                self.wiggle_until = state.t + 1.2
            elif self.recovery_stage >= 4:
                if robot.hopper <= 5:
                    self.abandon = True
                else:
                    self.recovery_point = self.planner.score_spot(state, -math.pi / 4)
                    self.crossing = None
                self.recovery_stage = 1
        self.recovering = self.recovery_stage > 0
        return Goal("Score", self.recovery_point or goal.point)

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
        goal = self.progress(goal, state)
        robot = state.robots[state.me]
        active = state.hubs[robot.alliance].active
        hub = self.planner.hub(state)
        low, high = self.planner.bounds()
        distance_hub = math.hypot(robot.x - hub[0], robot.z - hub[1])
        shooting = goal.kind == "Score" and active and robot.hopper > 0
        own_side = robot.x < -SCORING_X if robot.alliance == "blue" else robot.x > SCORING_X
        at_spot = math.dist((robot.x, robot.z), goal.point) < .7
        target = goal.point
        point = self.waypoint((robot.x, robot.z), target)
        dx, dz = point[0] - robot.x, point[1] - robot.z
        distance = math.hypot(dx, dz)
        direction = math.atan2(dz, dx)
        moving_shot = (self.shoot_while_collecting and goal.kind == "Collect" and active and own_side
                       and robot.hopper > 0 and (self.spec.max_shot_speed or 0) >= 2
                       and low + .2 <= distance_hub <= high - .5)
        arriving = distance < .3
        final_waypoint = math.dist(point, target) < .01
        vmax = self.spec.max_speed
        # Conservative stopping distance also leaves room for the engine's velocity response.
        speed = min(vmax, 2 * distance, math.sqrt(2 * self.spec.accel * .4 * max(0, distance - .12)))
        if goal.kind == "Collect":
            if distance < 1:
                speed = min(speed, vmax * PARAMS["collect_near_frac"])
            speed *= max(PARAMS["collect_cos_floor"], math.cos(angle_error(direction, robot.a)))
        if arriving and final_waypoint:
            speed = 0
        elif arriving:
            speed = max(speed, .35)
        score_speed = self.spec.max_shot_speed or self.spec.brain_equivalents.get("scoreSpeedTol", .15)
        if moving_shot:
            speed = min(speed, score_speed * .85)
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
        intake = robot.hopper < self.planner.capacity(state) and (path_fuel or (goal.kind == "Collect" and not PARAMS["intake_needs_fuel"]))
        if shooting and state.hubs[robot.alliance].seconds_to_toggle <= self.planner.score_time(state) + 1:
            intake = False
        if shooting and self.recovering:
            intake = False
        score = shooting and own_side and at_spot and low <= distance_hub <= high and math.hypot(robot.vx, robot.vz) <= score_speed
        score = score or moving_shot and math.hypot(robot.vx, robot.vz) <= score_speed
        if shooting and self.recovery_stage == 1 and state.t < self.reaim_until:
            x = y = 0
            score = False
        if shooting and state.t < self.wiggle_until:
            pulse = 1 if int((self.wiggle_until - state.t) / .3) % 2 else -1
            y = sign * inverse_deadzone(pulse / self.spec.max_speed)
            x = 0
            turn = inverse_deadzone(math.copysign(.25, pulse))
            score = True
        return Inputs(x=x, y=y, turn=turn, intake=intake, score=score, field_relative=True)
