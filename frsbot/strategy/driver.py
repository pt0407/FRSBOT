"""Strategy-only pursuit, collision avoidance, and conservative foul guards."""
import math
from dataclasses import replace

import numpy as np

from frsbot.bot.driver import Driver, inverse_deadzone
from frsbot.bot.planner import FIELD_X, FIELD_Z, Goal, angle_error
from frsbot.core.field import HUB_CENTRES
from frsbot.core.inputs import Inputs
from frsbot.core.state import Phase
from frsbot.strategy import fouls
from frsbot.strategy.intent import ALLIANCE_ZONE_EDGE


LANE_BOXES = fouls.TRENCH_BOXES + fouls.BUMP_BOXES


def half_diagonal(spec):
    return math.hypot(spec.bumper_length, spec.bumper_width) / 2


def inside(point, box, pad=0.):
    x, z = point
    left, right, bottom, top = box
    return left - pad <= x <= right + pad and bottom - pad <= z <= top + pad


def safe_point(point, spec, lanes=True):
    """Nearest rectangular-obstacle projection, including the whole bumper."""
    pad = half_diagonal(spec) + .12
    x = max(-FIELD_X + pad, min(FIELD_X - pad, float(point[0])))
    z = max(-FIELD_Z + pad, min(FIELD_Z - pad, float(point[1])))
    boxes = [(cx - .597, cx + .597, cz - .597, cz + .597) for cx, cz, _ in HUB_CENTRES.values()]
    if lanes:
        boxes += list(LANE_BOXES)
    if not any(inside((x, z), box, pad) for box in boxes):
        return x, z
    xs, zs = {x, -FIELD_X + pad, FIELD_X - pad}, {z, -FIELD_Z + pad, FIELD_Z - pad}
    for left, right, bottom, top in boxes:
        xs.update((left - pad - .01, right + pad + .01))
        zs.update((bottom - pad - .01, top + pad + .01))
    candidates = [(a, b) for a in xs for b in zs if abs(a) <= FIELD_X - pad and abs(b) <= FIELD_Z - pad
                  and not any(inside((a, b), box, pad) for box in boxes)]
    return min(candidates, key=lambda p: (math.dist(p, (x, z)), p))


class StrategicDriver:
    def __init__(self, spec, specs_by_team):
        self.spec = spec
        self.specs = specs_by_team
        self.base = Driver(spec)
        self.base.shoot_while_collecting = False
        self.contacts = {}
        self.retreat_target = None
        self.retreat_point = None
        self.clear_since = None

    def velocity(self, inputs, robot):
        norm = math.hypot(inputs.x, inputs.y)
        if norm <= .12:
            return np.zeros(2)
        speed = min(1., (norm - .12) / .88) * self.spec.max_speed
        sign = 1 if robot.alliance == "blue" else -1
        return sign * speed * np.array([inputs.y, inputs.x]) / norm

    def command(self, velocity, robot, original=None):
        vx, vz = velocity
        speed = math.hypot(vx, vz)
        stick = inverse_deadzone(speed / self.spec.max_speed)
        sign = 1 if robot.alliance == "blue" else -1
        return replace(original or Inputs(field_relative=True),
                       x=sign * stick * vz / speed if speed else 0.,
                       y=sign * stick * vx / speed if speed else 0.)

    def path_fuel(self, state, velocity):
        robot = state.robots[state.me]
        points = self.base.planner.fuel_points(state)
        if not len(points):
            return False
        speed = np.linalg.norm(velocity)
        direction = velocity / speed if speed > .1 else np.array([math.cos(robot.a), math.sin(robot.a)])
        rel = points - [robot.x, robot.z]
        ahead = rel @ direction
        across = rel[:, 0] * direction[1] - rel[:, 1] * direction[0]
        return robot.hopper < self.base.planner.capacity(state) and bool(np.any(
            (ahead > 0) & (ahead < 1.2) & (abs(across) < self.spec.bumper_width / 2)))

    def act(self, goal, state, target=None):
        robot = state.robots[state.me]
        if state.phase == Phase.AUTO and goal.kind in ("Defend", "Contest"):
            inputs = Inputs(field_relative=True)
        elif goal.kind == "Defend":
            if target is None:
                raise ValueError("Defend requires a target robot")
            point = self.base.waypoint((robot.x, robot.z), goal.point)
            velocity = 3. * (np.array(point) - [robot.x, robot.z])
            if math.dist(point, goal.point) < .01:
                velocity += [target.vx, target.vz]
            aim = math.atan2(target.z - robot.z, target.x - robot.x)
            turn = inverse_deadzone(max(-1., min(1., -4 * angle_error(aim, robot.a) / self.spec.turn_rate)))
            inputs = self.command(velocity, robot, Inputs(turn=turn, intake=self.path_fuel(state, velocity), field_relative=True))
        elif goal.kind in ("Contest", "Feed"):
            inputs = self.base.act(Goal("Collect" if goal.kind == "Contest" else "Stage", goal.point), state)
            inputs.score = (goal.kind == "Feed" and robot.hopper > 0 and
                            abs(robot.x) < ALLIANCE_ZONE_EDGE - half_diagonal(self.spec) - .1 and
                            math.dist((robot.x, robot.z), goal.point) < .6 and math.hypot(robot.vx, robot.vz) < .3)
            if inputs.score:
                inputs.turn = 0.  # Native passing controls its own aim.
        else:
            inputs = self.base.act(goal, state)
        return self.guard(inputs, goal, state, target)

    def _pin_retreat(self, state):
        me = state.robots[state.me]
        opponents = [r for r in state.robots if r.alliance != me.alliance]
        for other in opponents:
            radius = half_diagonal(self.spec) + half_diagonal(self.specs[other.team])
            if math.dist((me.x, me.z), (other.x, other.z)) < radius + fouls.CONTACT_DISTANCE_MARGIN:
                self.contacts.setdefault(other.id, state.t)
                if self.retreat_target is None and state.t - self.contacts[other.id] >= fouls.PIN_RETREAT_AFTER_SECONDS - 1e-7:
                    self.retreat_target = other.id
                    self.clear_since = None
            else:
                self.contacts.pop(other.id, None)
        other = next((r for r in opponents if r.id == self.retreat_target), None)
        if other is None:
            self.retreat_target = self.retreat_point = self.clear_since = None
            return None
        radius = half_diagonal(self.spec) + half_diagonal(self.specs[other.team])
        clearance = radius + fouls.PIN_CLEAR_DISTANCE_METERS + fouls.PIN_RETREAT_DISTANCE_MARGIN
        away = np.array([me.x - other.x, me.z - other.z])
        distance = np.linalg.norm(away)
        if distance >= clearance:
            if self.clear_since is None:
                self.clear_since = state.t
            if state.t - self.clear_since >= fouls.PIN_CLEAR_SECONDS + fouls.PIN_RETREAT_TIME_MARGIN - 1e-7:
                self.retreat_target = self.retreat_point = self.clear_since = None
                self.contacts.clear()
                return None
        else:
            self.clear_since = None
        if distance < 1e-6:
            away = np.array([-1. if me.alliance == "blue" else 1., 0.])
            distance = 1.
        point = np.array([other.x, other.z]) + away / distance * (clearance + .15)
        self.retreat_point = safe_point(point, self.spec)
        return self.base.act(Goal("Stage", self.retreat_point), state)

    def guard(self, inputs, goal, state, target=None):
        if state.phase in (Phase.COUNTDOWN, Phase.SETTLE, Phase.DONE):
            return inputs
        me = state.robots[state.me]
        retreat = self._pin_retreat(state)
        if retreat is not None:
            inputs = replace(retreat, score=False, intake=False)
            target = None
        velocity = self.velocity(inputs, me)
        original_velocity = velocity.copy()
        for other in state.robots:
            if other.id == me.id:
                continue
            relative = np.array([other.x - me.x, other.z - me.z])
            distance = np.linalg.norm(relative)
            opponent = other.alliance != me.alliance
            radii = half_diagonal(self.spec) + half_diagonal(self.specs[other.team])
            future_x = other.x + .5 * other.vx
            opponent_half = max(other.x, future_x) > 0 if me.alliance == "blue" else min(other.x, future_x) < 0
            relative_velocity = np.array([other.vx, other.vz]) - velocity
            closest_t = np.clip(-relative @ relative_velocity / max(relative_velocity @ relative_velocity, 1e-9), 0., .5)
            closest = relative + closest_t * relative_velocity
            if state.phase == Phase.AUTO and opponent and opponent_half and np.linalg.norm(closest) < radii + .3:
                velocity = -relative / max(distance, .01) * self.spec.max_speed
                continue
            if opponent and target is not None and target.id == other.id:
                continue
            speed = np.linalg.norm(velocity)
            if distance < 1.5 and speed > .1 and relative @ velocity > 0:
                direction = velocity / speed
                lateral = np.array([-direction[1], direction[0]])
                side = -1. if relative @ lateral >= 0 else 1.
                velocity = .2 * velocity + side * lateral * self.spec.max_speed * .6
        # Never complete the opposite member of a teammate's trench/bump pair.
        future = np.array([me.x, me.z]) + .3 * velocity
        for teammate in state.robots:
            if teammate.id == me.id or teammate.alliance != me.alliance:
                continue
            for boxes in (fouls.TRENCH_BOXES, fouls.BUMP_BOXES):
                for i, box in enumerate(boxes):
                    if inside((teammate.x, teammate.z), box, half_diagonal(self.specs[teammate.team])) and inside(future, boxes[i ^ 1], half_diagonal(self.spec)):
                        velocity = (np.array(safe_point(future, self.spec)) - [me.x, me.z]) / .3
        if not np.array_equal(velocity, original_velocity):
            future = np.array([me.x, me.z]) + .3 * velocity
            velocity = (np.array(safe_point(future, self.spec, lanes=False)) - [me.x, me.z]) / .3
            inputs = self.command(velocity, me, inputs)
        if goal.kind in ("Contest", "Feed") and any(math.dist((me.x, me.z), hub[:2]) < fouls.HUB_INTAKE_AVOID_DISTANCE for hub in HUB_CENTRES.values()):
            inputs = replace(inputs, intake=False)
        return inputs
