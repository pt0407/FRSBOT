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
ZONE_X = 4.2418  # Alliance-zone boundary (hub east face), from engine module 2960.
FEED_X = 3.9  # Centre margin: |x| below this is safely outside the zone for passing.

# Times at which a shift starts (also hub-toggle instants).
SHIFT_START = {Phase.SHIFT1: 33., Phase.SHIFT2: 58., Phase.SHIFT3: 83., Phase.SHIFT4: 108.}


def angle_error(target, actual):
    return (target - actual + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Goal:
    kind: str
    point: tuple[float, float]
    path: tuple = ()


class Planner:
    def __init__(self, spec: RobotSpec):
        self.spec = spec
        self.goal = None
        self.profile = profile(spec.team)
        self.avoid_points = []
        self.avoid_slots = []
        self.score_offset = 0.
        self.collect_gain_t = 0.
        self.collect_prev_hopper = 0
        self.collect_dead = []
        self._collect_point = None
        self._starve_t = 0.
        self._starve_prev = 0

    def capacity(self, state):
        # hopperFuel/hopperCapacity are storage counts, not the UI's feed-inclusive count.
        if PARAMS["use_total_capacity"]:
            return self.spec.total_capacity
        return state.robots[state.me].capacity or self.spec.total_capacity

    def bounds(self):
        return self.spec.min_range or min(self.spec.hub_shot["distance"]), self.spec.max_range or max(self.spec.hub_shot["distance"])

    def hub(self, state):
        return HUB_CENTRES[state.robots[state.me].alliance][:2]

    def hub_open(self, state):
        """Effective fire window: active, plus the pre-fire lead before activation
        (shots launched ~tof early still count) and the short post-grace after
        deactivation measured at ~0.3 s in the engine. Timetable-derived so the
        JS port doesn't need the phase enum."""
        hub = state.hubs[state.robots[state.me].alliance]
        if hub.active:
            return True
        if hub.seconds_to_toggle <= PARAMS["pre_fire_s"]:
            return True
        start = max((s for s in SHIFT_START.values() if s <= state.t), default=None)
        return start is not None and state.t - start <= PARAMS["post_fire_s"]

    def own_zone(self, robot):
        """Robot is inside its own scoring zone (centre-based, with bumper margin)."""
        return robot.x < -ZONE_X if robot.alliance == "blue" else robot.x > ZONE_X

    def score_spot(self, state, angle_offset=0):
        robot = state.robots[state.me]
        cx, cz = self.hub(state)
        low, high = self.bounds()
        frac = PARAMS.get("score_radius_frac", 0.5)
        radius = max(low + .3, min(high - .5, low + frac * (high - low)))
        heading = math.atan2(robot.z - cz, robot.x - cx) + angle_offset + self.score_offset
        margin = math.hypot(self.spec.bumper_length, self.spec.bumper_width) / 2 + .1
        # The robot body must clear the hub box (engine collider, +-0.597 m
        # around the hub centre) inflated by its half-diagonal — otherwise the
        # spot is unreachable and the router rams the hub face. Close spots
        # stay legal by flanking the hub in z, like the pros' parks.
        body = margin + .05
        candidates = []
        for delta in [0, *[s * i * math.pi / 36 for i in range(1, 37) for s in (-1, 1)]]:
            x, z = cx + radius * math.cos(heading + delta), cz + radius * math.sin(heading + delta)
            own_side = x < -SCORING_X if robot.alliance == "blue" else x > SCORING_X
            if (own_side and abs(x) <= FIELD_X - margin and abs(z) <= FIELD_Z - margin
                    and not (abs(x - cx) < .597 + body and abs(z - cz) < .597 + body)):
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
        if not candidates:
            return preferred
        return min(candidates, key=lambda p: math.dist(p, preferred))

    def fuel_points(self, state):
        pos = state.fuel.pos
        return pos[(state.fuel.where == 0) & (pos[:, 1] < .2)][:, [0, 2]]

    def inzone_fuel(self, state):
        """Resting fuel inside our scoring zone (x beyond the -4.2418/+4.2418 line)."""
        robot = state.robots[state.me]
        points = self.fuel_points(state)
        if not len(points):
            return points
        inside = points[:, 0] < -SCORING_X if robot.alliance == "blue" else points[:, 0] > SCORING_X
        return points[inside]

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
        # Momentum lane bias: reversing direction costs a full stop plus
        # re-acceleration and, unlike the point-to-point model assumes, the
        # capture rate scales with speed while the intake is in dense fuel.
        # Penalise clusters behind the velocity vector so lanes chain forward.
        speed = math.hypot(robot.vx, robot.vz)
        if speed > .8:
            dist = np.hypot(dx, dz) + 1e-6
            align = (dx * robot.vx + dz * robot.vz) / (dist * speed)
            travel = travel * (1 + PARAMS["lane_bias"] * (1 - align))
        turn = np.abs((np.arctan2(dz, dx) - robot.a + np.pi) % (2 * np.pi) - np.pi) / self.spec.turn_rate
        values = balls / (travel + turn + PARAMS["cluster_bias"])
        own = self.score_spot(state)
        if state.hubs[robot.alliance].active:
            values /= 1 + PARAMS["active_hub_bias"] * np.hypot(centres[:, 0] - own[0], centres[:, 1] - own[1])
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
        if self.collect_dead:
            self.collect_dead = [d for d in self.collect_dead if d[2] > state.t]
            result = [(value, point) for value, point in result
                      if all(math.dist(point, (dx, dz)) >= 1.0 for dx, dz, _ in self.collect_dead)]
        return sorted(result, reverse=True)

    def sweep_goal(self, state, centroid):
        # Lane-sweep primitive: the goal point is the lane EXIT beyond the
        # cluster, so the robot drives through the pile at cruise instead of
        # aiming at it and reversing on arrival. The centroid rides in
        # goal.path so valid() can check the pile rather than the exit.
        robot = state.robots[state.me]
        cx, cz = centroid
        dx, dz = cx - robot.x, cz - robot.z
        dist = math.hypot(dx, dz)
        speed = math.hypot(robot.vx, robot.vz)
        # Direction commitment: once the robot is passing through or past the
        # pile (centroid no longer clearly ahead of the velocity vector), the
        # lane direction must come from the velocity itself — otherwise the
        # robot->centroid bearing flips 180 deg mid-sweep and the regenerated
        # lane points back the way we came.
        ahead = dist > 1e-6 and speed > .8 and (dx * robot.vx + dz * robot.vz) / (dist * speed) > .3
        if speed > .8 and not ahead:
            ux, uz = robot.vx / speed, robot.vz / speed
        elif dist > 1e-6:
            ux, uz = dx / dist, dz / dist
        else:
            ux, uz = math.cos(robot.a), math.sin(robot.a)
        lane = PARAMS["sweep_lane_m"]
        # Truncate the lane where it would end inside a wall or hub structure —
        # an unreachable exit wedges the robot against the geometry it cannot
        # enter (observed: 42 s pinned at x=-7.16 chasing a point inside the hub).
        margin = .45
        ex, ez = cx + ux * lane, cz + uz * lane
        for hx, hz, _ in HUB_CENTRES.values():
            if abs(ex - hx) < 1.5 and abs(ez - hz) < 1.5:
                # walk the lane back out of the hub box
                for back in np.linspace(0, lane, 24):
                    ex, ez = cx + ux * (lane - back), cz + uz * (lane - back)
                    if not (abs(ex - hx) < 1.5 and abs(ez - hz) < 1.5):
                        break
                break
        ex = max(-FIELD_X + margin, min(FIELD_X - margin, ex))
        ez = max(-FIELD_Z + margin, min(FIELD_Z - margin, ez))
        return Goal("Collect", (ex, ez), path=(cx, cz))

    def valid(self, goal, state):
        if goal is None:
            return False
        robot = state.robots[state.me]
        if goal.kind == "Collect":
            points = self.fuel_points(state)
            centre = goal.path if len(goal.path) == 2 else goal.point
            return robot.hopper < self.capacity(state) and bool(len(points) and np.any(np.linalg.norm(points - centre, axis=1) < .85))
        if goal.kind == "Score":
            # With a live in-zone pile the Score goal should release once the
            # hopper drains to the floor so the planner can refill and cycle.
            floor = PARAMS["score_floor"] if len(self.inzone_fuel(state)) >= PARAMS["pile_exit"] else 0
            return robot.hopper > floor and self.hub_open(state)
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
        if state.t >= 162.5:
            return Goal("Score", (robot.x, robot.z)) if robot.hopper and hub.active and own_side and low <= distance <= high else Goal("Stage", (robot.x, robot.z))
        clusters = self.clusters(state)
        if self.goal and self.goal.kind == "Collect":
            if self._collect_point is None or math.dist(self.goal.point, self._collect_point) > .6:
                self._collect_point = self.goal.point
                self.collect_gain_t = state.t
            progress = robot.hopper != self.collect_prev_hopper
            if progress:
                self.collect_gain_t = state.t
            elif state.t - self.collect_gain_t > PARAMS["collect_stall_s"] and (
                    math.dist((robot.x, robot.z), self._collect_point) < 1.5
                    or math.hypot(robot.vx, robot.vz) < .2):
                self.collect_dead.append((*self._collect_point, state.t + PARAMS["collect_dead_s"]))
                self.collect_gain_t = state.t
                self._collect_point = None
                self.goal = None
        else:
            self._collect_point = None
        self.collect_prev_hopper = robot.hopper
        if robot.hopper > self._starve_prev:
            self._starve_t = state.t
        self._starve_prev = robot.hopper
        # Diminishing returns: with the hub open, a hopper that stopped growing
        # for ~6 s is worth more shot now than chased to full (1678 ground 45 s
        # for ~10 balls instead of scoring 55).
        starved = state.t - self._starve_t > 6.0
        if hub.active and robot.hopper and (state.t < PARAMS["early_score_until"] or hub.seconds_to_toggle <= score_time + 1 or
                robot.hopper >= PARAMS["active_score_frac"] * self.capacity(state) or not clusters or clusters[0][0] < 2
                or (starved and robot.hopper >= .45 * self.capacity(state))):
            goal = Goal("Score", spot)
        elif not hub.active and robot.hopper and (robot.hopper >= PARAMS["stage_full_frac"] * self.capacity(state) or hub.seconds_to_toggle < score_time + PARAMS["arrive_margin_s"]):
            goal = Goal("Stage", spot)
        elif clusters:
            goal = self.sweep_goal(state, clusters[0][1])
        else:
            goal = Goal("Stage", spot if robot.hopper else (robot.x, robot.z))
        if self.goal and self.goal.kind == "Stage" and robot.hopper and hub.active:
            goal = Goal("Score", spot)
        # Finish unloading instead of changing targets as soon as one ball leaves.
        if self.goal and self.goal.kind == "Score" and self.valid(self.goal, state):
            goal = self.goal
        self.goal = goal
        return goal
