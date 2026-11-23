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
        self.conv_side = 1
        self.conv_cooldown = 0.
        self.collect_gain_t = 0.
        self.collect_prev_hopper = 0
        self.collect_dead = []
        self._collect_point = None
        self._sweep_prog = 0
        self.pile_spot = None
        self.seed_spot = None

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

    def feed_until(self, state):
        """Stop feeding this many seconds before the hub activates — enough to
        reach the score spot with a small margin."""
        robot = state.robots[state.me]
        spot = self.score_spot(state)
        travel = math.dist((robot.x, robot.z), spot) / self.spec.max_speed + .5
        return travel + PARAMS["feed_lead_s"]

    def score_spot(self, state, angle_offset=0):
        robot = state.robots[state.me]
        cx, cz = self.hub(state)
        low, high = self.bounds()
        frac = PARAMS.get("score_radius_frac", 0.5)
        radius = max(low + .3, min(high - .5, low + frac * (high - low)))
        if PARAMS["bank"]:
            radius = max(low + .25, min(high - .3, PARAMS["score_close_m"]))
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

    def inzone_fuel(self, state):
        """Resting fuel inside our scoring zone (x beyond the -4.2418/+4.2418 line)."""
        robot = state.robots[state.me]
        points = self.fuel_points(state)
        if not len(points):
            return points
        inside = points[:, 0] < -SCORING_X if robot.alliance == "blue" else points[:, 0] > SCORING_X
        return points[inside]

    def mow_path(self, side_start):
        """Boustrophedon over the deep zone: legs along ±z at x columns ~0.9 m apart,
        shallow end first. Stays clear of the tower/depot boxes (x < -7) by keeping the
        deep columns at |z| > 1.0."""
        sign = 1 if side_start == "blue" else -1
        margin = .7
        # Columns stay clear of the tower (x<-7.12, |z|<.8) and depot (x<-7.58, z -2.5..-1.4).
        cols = [-sign * (SCORING_X + .8 + .9 * i) for i in range(3)]
        path = []
        lo, hi = -(FIELD_Z - margin), FIELD_Z - margin
        flip = False
        for cx_ in cols:
            a_, b_ = (hi, lo) if flip else (lo, hi)
            path.append((cx_, a_))
            path.append((cx_, b_))
            flip = not flip
        return tuple(path)

    def seed_path(self, alliance):
        """Lane 1.5 m inside the zone: the outtake disc (~1.5-2.5 m radius) then
        lands mostly in the patrol band x -7.5..-4.5, clear of posts/tower/depot."""
        sign = 1 if alliance == "blue" else -1
        x = -sign * (SCORING_X + 1.55)
        return ((x, -2.9), (x, 2.9))

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

    def valid(self, goal, state):
        if goal is None:
            return False
        robot = state.robots[state.me]
        if goal.kind == "Collect":
            points = self.fuel_points(state)
            return robot.hopper < self.capacity(state) and bool(len(points) and np.any(np.linalg.norm(points - goal.point, axis=1) < .85))
        if goal.kind == "Sweep":
            return robot.hopper < self.capacity(state) and math.dist((robot.x, robot.z), goal.point) > .5
        if goal.kind == "Score":
            # With a live in-zone pile the Score goal should release once the
            # hopper drains to the floor so the planner can refill and cycle.
            floor = PARAMS["score_floor"] if len(self.inzone_fuel(state)) >= PARAMS["pile_exit"] else 0
            return robot.hopper > floor and self.hub_open(state)
        if goal.kind == "Feed":
            return (not self.hub_open(state) and state.phase in SHIFT_START
                    and robot.hopper > PARAMS["score_floor"])
        if goal.kind == "ZoneMow":
            return (self.hub_open(state) and self.spec.shoot_on_move
                    and (robot.hopper > 0 or len(self.inzone_fuel(state)) >= PARAMS["pile_exit"] * .5))
        if goal.kind == "Patrol":
            return state.hubs[robot.alliance].active
        if goal.kind in ("Seed", "SeedMow"):
            return (robot.hopper > PARAMS["score_floor"] and not self.hub_open(state)
                    and state.phase in SHIFT_START)
        return robot.hopper > 0 and not state.hubs[robot.alliance].active

    def conveyor_points(self, state, side):
        robot = state.robots[state.me]
        cx, cz = self.hub(state)
        sign = 1 if robot.alliance == "blue" else -1
        spot = (cx - sign * PARAMS["conv_spot_x"], cz + side * PARAMS["conv_spot_z"])
        blob = (cx + sign * PARAMS["conv_blob_x"], cz)
        return spot, blob

    def sweep_path(self, state, side):
        """Long diagonal passes through the dense collection band (x ~ -4..+1.5, z on `side`),
        including a dip inside the zone line, and exits toward the dump spot.
        Avoids the trench band z~2.4-2.9 at x~-3.6."""
        robot = state.robots[state.me]
        cx, cz = self.hub(state)
        sign = 1 if robot.alliance == "blue" else -1
        return (
            (cx + sign * 1.1, cz + side * 2.0),
            (cx + sign * 2.6, cz + side * 2.5),
            (cx + sign * 5.0, cz + side * 1.6),
            (cx + sign * 5.0, cz + side * .0),
            (cx + sign * 3.0, cz + side * .2),
            (cx + sign * .4, cz + side * .9),
            (cx - sign * .6, cz + side * 1.5),
            (cx + sign * .35, cz + side * 1.8),
        )

    def conveyor_goal(self, state):
        robot = state.robots[state.me]
        cx, cz = self.hub(state)
        gkind = self.goal.kind if self.goal is not None else None
        arrived = False
        if gkind == "Sweep":
            path = self.goal.path
            i = self._sweep_prog
            # Mirror Driver's waypoint advancement so progress and arrival agree.
            while i < len(path) - 1 and (math.dist((robot.x, robot.z), path[i]) < .55 or
                    math.dist((robot.x, robot.z), path[i + 1]) + .4 < math.dist((robot.x, robot.z), path[i])):
                i += 1
            self._sweep_prog = i
            arrived = i == len(path) - 1 and math.dist((robot.x, robot.z), path[i]) < .6
        if gkind == "Score" and robot.hopper > PARAMS["dump_floor"]:
            return self.goal
        if robot.hopper >= PARAMS["conv_full"] or arrived and robot.hopper >= PARAMS["conv_dump_min"]:
            self.conv_side = 1 if robot.z >= cz else -1
            return Goal("Score", self.conveyor_points(state, self.conv_side)[0])
        if gkind == "Sweep" and not arrived and robot.hopper < self.capacity(state):
            return self.goal
        if gkind == "Sweep" and arrived:
            self.conv_cooldown = state.t + 4.0
            return None
        if state.t < self.conv_cooldown:
            return None
        if gkind == "Collect" and self.valid(self.goal, state):
            return self.goal
        points = self.fuel_points(state)
        if not len(points):
            return None
        band = points[np.abs(points[:, 0] - (cx + (1 if robot.alliance == "blue" else -1) * 2.0)) < 3.5]
        if len(band):
            self.conv_side = 1 if (band[:, 1] > cz).sum() >= (~(band[:, 1] > cz)).sum() else -1
        else:
            self.conv_side = 1 if robot.z >= cz else -1
        _, blob = self.conveyor_points(state, self.conv_side)
        d = np.linalg.norm(points - np.array(blob), axis=1)
        if d.min() > PARAMS["conv_blob_r"]:
            return None
        path = self.sweep_path(state, self.conv_side)
        self._sweep_prog = 0
        return Goal("Sweep", path[-1], path=path)

    def plan(self, state: GameState) -> Goal:
        robot = state.robots[state.me]
        hub = state.hubs[robot.alliance]
        spot = self.score_spot(state)
        if PARAMS["conveyor"]:
            cz = self.hub(state)[1]
            spot = self.conveyor_points(state, 1 if robot.z >= cz else -1)[0]
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
        iz = self.inzone_fuel(state)
        clusters = self.clusters(state)
        if self.goal and self.goal.kind in ("Collect", "Feed", "ZoneMow"):
            if self._collect_point is None or math.dist(self.goal.point, self._collect_point) > .6:
                self._collect_point = self.goal.point
                self.collect_gain_t = state.t
                self._patrol_iz = len(iz)
            progress = robot.hopper != self.collect_prev_hopper
            if self.goal.kind != "Collect":
                progress = progress or len(iz) != getattr(self, "_patrol_iz", len(iz))
            self._patrol_iz = len(iz)
            if progress:
                self.collect_gain_t = state.t
            elif state.t - self.collect_gain_t > PARAMS["collect_stall_s"] and (
                    math.dist((robot.x, robot.z), self._collect_point) < 1.5
                    or math.hypot(robot.vx, robot.vz) < .2):
                self.collect_dead.append((*self._collect_point, state.t + PARAMS["collect_dead_s"]))
                self.collect_gain_t = state.t
                self._collect_point = None
                self.goal = None
        elif self.goal and self.goal.kind == "Patrol":
            # Firing drains the hopper; filling grows it; pile shrink means intake.
            # All three are progress — only a *flat* hopper and pile is a stall.
            if self._collect_point is None or math.dist(self.goal.point, self._collect_point) > .6:
                self._collect_point = self.goal.point
                self.collect_gain_t = state.t
                self._patrol_iz = len(iz)
            if robot.hopper != self.collect_prev_hopper or len(iz) < getattr(self, "_patrol_iz", 0) - 3:
                self.collect_gain_t = state.t
                self._patrol_iz = len(iz)
            elif state.t - self.collect_gain_t > PARAMS["collect_stall_s"]:
                self.collect_dead.append((*self._collect_point, state.t + PARAMS["collect_dead_s"]))
                self.collect_gain_t = state.t
                self._collect_point = None
                self.goal = None
        else:
            self._collect_point = None
        self.collect_prev_hopper = robot.hopper
        # Bank during inactive shifts: fill the hopper on midfield clusters, then
        # park ~1.2 m deep of the score spot and outtake the hopper — a dense
        # pile right at the firing point. When the hub opens the standard
        # Collect<->Score cycle consumes it with almost zero travel.
        if PARAMS["bank"]:
            sign = -1 if robot.alliance == "blue" else 1
            if not self.hub_open(state) and state.phase in SHIFT_START:
                if hub.seconds_to_toggle > self.feed_until(state):
                    cx, cz = self.hub(state)
                    dx, dz = spot[0] - cx, spot[1] - cz
                    n = math.hypot(dx, dz) or 1
                    seed_pt = (spot[0] + 1.2 * dx / n, spot[1] + 1.2 * dz / n)
                    full = robot.hopper >= PARAMS["seed_hi_frac"] * self.capacity(state)
                    seeding = self.goal is not None and self.goal.kind == "Seed" and robot.hopper > PARAMS["score_floor"]
                    if full or seeding:
                        self.goal = Goal("Seed", seed_pt)
                        return self.goal
                    # Prefer midfield clusters — the banked pile gets consumed
                    # once the hub opens; driving back in now just wastes travel.
                    out = [c for c in clusters if sign * c[1][0] < ZONE_X]
                    if out:
                        self.goal = Goal("Collect", out[0][1])
                        return self.goal
                    if clusters:
                        self.goal = Goal("Collect", clusters[0][1])
                        return self.goal
                    if robot.hopper > PARAMS["score_floor"]:
                        self.goal = Goal("Seed", seed_pt)
                        return self.goal
                # Time to head in-zone; the Score goal parks at the spot and the
                # driver's hub_open window presses score during pre-fire.
                self.goal = Goal("Score", spot) if robot.hopper else Goal("Stage", spot)
                return self.goal
        # Pro cycle: during inactive shifts, bank collected fuel into the deep zone as a
        # spread pile (SeedMow); during active shifts, mow that in-zone fuel while firing
        # (Patrol) so intake and shooter run in the same region at once.
        patrolling = self.goal is not None and self.goal.kind == "Patrol"
        if (PARAMS["patrol"] and hub.active
                and state.phase in (Phase.TRANSITION, Phase.SHIFT1, Phase.SHIFT2, Phase.SHIFT3, Phase.SHIFT4, Phase.ENDGAME)
                and len(iz) >= (PARAMS["patrol_exit"] if patrolling else PARAMS["patrol_min"])):
            # Collect the densest in-zone cluster — the driver's in-zone collect
            # path fires while creeping through the pile (the pro's mow pattern).
            inside = [c for c in clusters if (c[1][0] < -SCORING_X if robot.alliance == "blue" else c[1][0] > SCORING_X)]
            if inside:
                self.goal = Goal("Patrol", inside[0][1])
                return self.goal
        if (PARAMS["seed_inactive"] and not hub.active and robot.hopper
                and state.phase in (Phase.SHIFT1, Phase.SHIFT2, Phase.SHIFT3, Phase.SHIFT4)
                and hub.seconds_to_toggle > PARAMS["seed_lead_s"]):
            cap = self.capacity(state)
            seeding = self.goal is not None and self.goal.kind == "SeedMow" and robot.hopper > PARAMS["seed_floor"]
            if seeding or robot.hopper >= PARAMS["seed_hi_frac"] * cap:
                path = self.seed_path(robot.alliance)
                self.goal = Goal("SeedMow", path[-1], path=path)
                return self.goal
        if PARAMS["conveyor"] and hub.active and state.phase in (Phase.TRANSITION, Phase.SHIFT1, Phase.SHIFT2, Phase.SHIFT3, Phase.SHIFT4, Phase.ENDGAME):
            goal = self.conveyor_goal(state)
            if goal is not None:
                self.goal = goal
                return goal
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
