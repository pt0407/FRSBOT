"""Obstacle-aware, braking-limited field-relative pursuit."""
import heapq
import math

import numpy as np

from frsbot.bot.planner import FIELD_X, FIELD_Z, SCORING_X, ZONE_X, FEED_X, Goal, Planner, angle_error
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
        # Shoot-on-move robots (maxShotSpeed >= 2) collect and fire together by
        # default; a profile entry can still force it off.
        self.shoot_while_collecting = self.planner.profile.get(
            "shoot_while_collecting", (self.spec.max_shot_speed or 0) >= 2)
        self.reaim_until = 0
        self.stall_t = None
        self.pin_t = None
        self.unwedge_until = 0

    def progress(self, goal, state):
        robot = state.robots[state.me]
        self.abandon = False
        scoring = goal.kind == "Score" and self.planner.hub_open(state) and robot.hopper > 0
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
        # Engine geometry (module 2960), rotated (-x,-z) per alliance side:
        # trench posts x in |3.048..4.242|, |z| in 2.451-2.756 (tall robots can't
        # pass the 0.565 m bar); tower x |7.12..8.3| z -0.21..+0.79 on blue
        # (mirrored on red); depot x |7.58..8.3|, blue z -2.46..-1.39.
        # The bump ramps (|z| 0.597-2.451) are drivable and not obstacles.
        for s in (-1, 1):
            if not self.spec.trench_cut:
                plo, phi = min(s * 3.048, s * 4.242), max(s * 3.048, s * 4.242)
                boxes += [(plo, phi, 2.40, 2.85), (plo, phi, -2.85, -2.40)]
            tlo, thi = min(s * 7.12, s * 8.35), max(s * 7.12, s * 8.35)
            dlo, dhi = min(s * 7.55, s * 8.35), max(s * 7.55, s * 8.35)
            if s == -1:
                boxes += [(tlo, thi, -.25, .83), (dlo, dhi, -2.5, -1.35)]
            else:
                boxes += [(tlo, thi, -.83, .25), (dlo, dhi, 1.35, 2.5)]
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
                if j == index:
                    continue
                # Edges into the target may end inside the box containing it;
                # every other box still blocks.
                blockers = boxes if j != 1 else [b for b in boxes
                    if not (b[0] < target[0] < b[1] and b[2] < target[1] < b[3])]
                if any(intersects(nodes[index], point, box) for box in blockers):
                    continue
                new_cost = cost + math.dist(nodes[index], point)
                if new_cost < best.get(j, math.inf):
                    best[j] = new_cost
                    heapq.heappush(queue, (new_cost, j, [*route, j]))
        return start

    def act(self, goal: Goal, state):
        goal = self.progress(goal, state)
        robot = state.robots[state.me]
        active = self.planner.hub_open(state)
        hub = self.planner.hub(state)
        low, high = self.planner.bounds()
        distance_hub = math.hypot(robot.x - hub[0], robot.z - hub[1])
        shooting = goal.kind == "Score" and active and robot.hopper > 0
        sign = 1 if robot.alliance == "blue" else -1
        own_side = robot.x < -SCORING_X if robot.alliance == "blue" else robot.x > SCORING_X
        in_zone = -sign * robot.x > ZONE_X - .34  # bumper-edge legality (~|x| 3.9)
        at_spot = math.dist((robot.x, robot.z), goal.point) < .7
        target = goal.point
        if goal.kind == "Score" and math.dist((robot.x, robot.z), target) < 2.0:
            point = target
        else:
            point = self.waypoint((robot.x, robot.z), target)
        dx, dz = point[0] - robot.x, point[1] - robot.z
        distance = math.hypot(dx, dz)
        direction = math.atan2(dz, dx)
        points = self.planner.fuel_points(state)
        path_fuel = False
        mow_dir = None
        if len(points):
            rel = points - [robot.x, robot.z]
            ahead = rel @ np.array([math.cos(robot.a), math.sin(robot.a)])
            lateral = rel @ np.array([-math.sin(robot.a), math.cos(robot.a)])
            cone = (ahead > 0) & (ahead < 1.6) & (abs(lateral) < self.spec.bumper_width * .6)
            path_fuel = bool(np.any((ahead > 0) & (ahead < 1.2) & (abs(lateral) < self.spec.bumper_width / 2)))
            if cone.sum() >= PARAMS["mow_cone_min"]:
                m = rel[cone].mean(axis=0)
                mow_dir = math.atan2(m[1], m[0])
        if goal.kind == "Collect" and mow_dir is not None:
            point = (robot.x + 2.5 * math.cos(mow_dir), robot.z + 2.5 * math.sin(mow_dir))
            dx, dz = point[0] - robot.x, point[1] - robot.z
            distance = math.hypot(dx, dz)
            direction = mow_dir
        elif goal.kind == "Collect" and distance < .6 and path_fuel:
            point = (robot.x + 1.2 * math.cos(robot.a), robot.z + 1.2 * math.sin(robot.a))
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
        brake_dist = distance
        speed = min(vmax, PARAMS["drive_dist_gain"] * brake_dist,
                    math.sqrt(2 * self.spec.accel * PARAMS["drive_brake_frac"] * max(0, brake_dist - .12)))
        if goal.kind == "Collect":
            # Human speed law (measured from 136k pro replay frames): pros never
            # brake for collect points and never throttle for heading error —
            # they sustain ~0.68*vmax through every distance bin, point-blank
            # included. Collection is pure pass-through at cruise speed.
            speed = vmax * PARAMS["collect_cruise_frac"]
        if arriving and final_waypoint:
            speed = 0
            if goal.kind == "Collect" and path_fuel:
                speed = vmax * PARAMS["collect_cruise_frac"]
        elif arriving:
            speed = max(speed, .35)
        score_speed = self.spec.max_shot_speed or self.spec.brain_equivalents.get("scoreSpeedTol", .15)
        if moving_shot:
            speed = min(speed, min(score_speed, self.spec.max_shot_speed * .6))
        if shooting and final_waypoint and low <= distance_hub <= high and distance < .6:
            speed = 0  # stop dead so the engine's hold/firing/xLock engages
        actual_v = math.hypot(robot.vx, robot.vz)
        if speed > .8:
            if actual_v < .18:
                if self.stall_t is None:
                    self.stall_t = state.t
                elif state.t - self.stall_t > 1.1:
                    self.unwedge_until = state.t + .9
                    self.stall_t = None
            else:
                self.stall_t = None
        # Physically pinned with a distant target and no escape: the speed gate
        # above misses it whenever the commanded speed is already low/zero
        # (1678 pinned on the tower face for 25 s during Collect). Score has
        # its own progress() recovery and legitimately sits at v~0 on the spot.
        if goal.kind != "Score" and distance > .5 and actual_v < .15:
            if self.pin_t is None:
                self.pin_t = state.t
            elif state.t - self.pin_t > 1.5:
                self.unwedge_until = state.t + .9
                self.pin_t = None
        else:
            self.pin_t = None
        if state.t < self.unwedge_until:
            direction = math.atan2(-dz, -dx)
            speed = .9
            vx, vz = speed * math.cos(direction), speed * math.sin(direction)
        else:
            vx, vz = speed * math.cos(direction), speed * math.sin(direction)
        margin = math.hypot(self.spec.bumper_length, self.spec.bumper_width) / 2 + .1
        if robot.x * vx > 0 and abs(robot.x + vx * .3) > FIELD_X - margin:
            vx = 0
        if robot.z * vz > 0 and abs(robot.z + vz * .3) > FIELD_Z - margin:
            vz = 0
        norm = math.hypot(vx, vz) / vmax
        stick = inverse_deadzone(norm)
        x = sign * stick * vz / math.hypot(vx, vz) if norm else 0
        y = sign * stick * vx / math.hypot(vx, vz) if norm else 0
        self.last_steer = dict(point=point, kind=goal.kind, rule=(x, y), learned=False)
        learned = None
        terminal = goal.kind != "Collect" and distance < PARAMS.get("learned_terminal_m", 1.5)
        if PARAMS.get("learned_drive") and state.t >= self.unwedge_until and not terminal:
            from frsbot.bot.learned_drive import steer
            learned = steer(state, self.spec, point, goal.kind)
        self.last_steer["learned"] = learned is not None
        if learned is not None:
            lx, ly, lturn = learned
            vx_l, vz_l = sign * ly * vmax, sign * lx * vmax
            if robot.x * vx_l > 0 and abs(robot.x + vx_l * .3) > FIELD_X - margin:
                ly = 0
            if robot.z * vz_l > 0 and abs(robot.z + vz_l * .3) > FIELD_Z - margin:
                lx = 0
            blend = PARAMS.get("learned_blend", .6)
            x, y = blend * lx + (1 - blend) * x, blend * ly + (1 - blend) * y
        # Shooter heading: rear shooters aim away from the hub (+pi), front
        # shooters (shooterYawDeg 180) aim at it, shifted by the mount yaw.
        aim = (math.atan2(hub[1] - robot.z, hub[0] - robot.x) + math.pi - math.radians(self.spec.shooter_yaw_deg)
               if goal.kind != "Collect" else direction)
        intake = robot.hopper < self.planner.capacity(state) and (path_fuel or (goal.kind == "Collect" and not PARAMS["intake_needs_fuel"]))
        outtake = False
        if shooting and state.hubs[robot.alliance].active and state.hubs[robot.alliance].seconds_to_toggle <= self.planner.score_time(state) + 1:
            intake = False
        if shooting and self.recovering:
            intake = False
        # Engine-enforced speed gate (scoreSpeedTol): pressing score early only
        # buys aim-translation slowdown, so approach_fire skips those robots.
        speed_gated = self.spec.brain_equivalents.get("scoreSpeedTol") is not None
        if PARAMS["free_fire"]:
            score = active and robot.hopper > 0 and own_side and low <= distance_hub <= high and math.hypot(robot.vx, robot.vz) <= PARAMS["free_fire_speed"]
            intake = robot.hopper < self.planner.capacity(state)
        else:
            score = shooting and in_zone and at_spot and low <= distance_hub <= high and math.hypot(robot.vx, robot.vz) <= score_speed
            score = score or moving_shot and math.hypot(robot.vx, robot.vz) <= min(score_speed, self.spec.max_shot_speed * .6)
            # Press score near the end of approach: hood/flywheel/tag spin-up
            # latches on the command and is ready on arrival without paying
            # the aim-translation slowdown for the whole leg.
            if PARAMS["approach_fire"] and not speed_gated and shooting and in_zone and low <= distance_hub <= high:
                score = score or distance < 1.6 or math.hypot(robot.vx, robot.vz) <= score_speed
        # Engine applies a 0.45 factor to the manual turn rate — divide it out so
        # the P-gain commands the intended omega and doesn't stall ~2.2x early.
        turn = inverse_deadzone(max(-1, min(1, -4 * angle_error(aim, robot.a) / (.45 * self.spec.turn_rate))))
        self.last_steer["turn"] = turn
        if learned is not None:
            tb = PARAMS.get("learned_turn_blend")
            tb = blend if tb is None else tb
            turn = tb * lturn + (1 - tb) * turn
        # While score is pressed the engine brain sets desiredRotation directly
        # and t.turn is ignored for every robot (verified: zeroing or steering
        # produced bit-identical evals) — zeroed for clarity. The heading gate
        # counts those dead-steer ticks but 'firing' latches on the first static
        # aligned tick, so they are mostly cosmetic, not lost shots.
        steers_free = self.spec.turret and goal.kind == "Collect"
        if score and not self.recovering and not steers_free:
            turn = 0
        if PARAMS["approach_fire"] and not speed_gated and shooting and own_side and low <= distance_hub <= high and math.hypot(robot.vx, robot.vz) <= score_speed:
            turn = 0
        if shooting and self.recovery_stage == 1 and state.t < self.reaim_until:
            x = y = 0
            score = False
        if shooting and state.t < self.wiggle_until:
            pulse = 1 if int((self.wiggle_until - state.t) / .3) % 2 else -1
            y = sign * inverse_deadzone(pulse / self.spec.max_speed)
            x = 0
            turn = inverse_deadzone(math.copysign(.25, pulse))
            score = True
        return Inputs(x=x, y=y, turn=turn, intake=intake, score=score, outtake=outtake, field_relative=True)
