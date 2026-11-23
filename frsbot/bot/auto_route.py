"""Pro-derived autonomous routine: pure-pursuit route following with parked releases.

Routes are extracted from pro replay telemetry into ``data/auto_routes/<team>.json``.
For the first ``AUTO_END`` seconds the robot chases a lookahead point on the pro's
route at high speed, tracks the pro's heading, intakes while the pro's hopper was
rising, and — during the pro's release windows — drives to the pro's position at
that moment, settles, aims at the hub and holds ``score``. Past ``AUTO_END`` the
caller resumes normal play.

Alliance mirroring is a 180-degree rotation in world coordinates (``-x, -z``,
heading + pi); the start pose lives in the engine's ``fx/fy`` frame and mirrors as
``F - fx, Y - fy, heading + 180``.
"""
import bisect
import json
import math
from pathlib import Path

from frsbot.bot.driver import inverse_deadzone
from frsbot.bot.planner import angle_error
from frsbot.core.inputs import Inputs

ROUTES = Path(__file__).resolve().parents[2] / "data" / "auto_routes"
AUTO_END = 20.0
LOOKAHEAD_S = 0.5
FIELD_F, FIELD_Y = 16.540988, 8.069326
HUB_X = {"blue": -3.645, "red": 3.645}
ZONE_X = 4.2418


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class AutoRoute:
    def __init__(self, spec, route):
        self.spec = spec
        self.points = route["points"]  # [t, x, z, heading]
        self.hopper = route["hopper"]  # [t, fuel]
        self.start = route["start"]    # {fx, fy, headingDeg}, blue-alliance frame
        self._pt = [p[0] for p in self.points]
        self._hp = [h[0] for h in self.hopper]

    @classmethod
    def for_team(cls, spec):
        path = ROUTES / f"{spec.team}.json"
        if not path.exists():
            return None
        return cls(spec, json.loads(path.read_text()))

    def start_for(self, alliance):
        s = self.start
        if alliance == "blue":
            return dict(s)
        return dict(fx=FIELD_F - s["fx"], fy=FIELD_Y - s["fy"], headingDeg=s["headingDeg"] + 180)

    def _point(self, t):
        i = min(len(self._pt) - 1, bisect.bisect_left(self._pt, t))
        return self.points[i]

    def _hopper_delta(self, t):
        """Predicted pro hopper change over the next ~1 s at time t."""
        j = min(len(self._hp) - 1, bisect.bisect_left(self._hp, t))
        j2 = min(len(self._hp) - 1, j + 5)  # ~0.15 s lookahead, not 1.0 s — release detection was braking early
        return self.hopper[j2][1] - self.hopper[j][1], self.hopper[j][1]

    def act(self, state):
        r = state.robots[state.me]
        al = r.alliance
        sign = 1 if al == "blue" else -1
        d, pro_hopper = self._hopper_delta(state.t)
        releasing = d <= -2
        target_t = state.t if releasing else min(state.t + LOOKAHEAD_S, AUTO_END + 2)
        _, px, pz, pa = self._point(target_t)
        tx, tz = (px, pz) if al == "blue" else (-px, -pz)
        aim = pa if al == "blue" else wrap(pa + math.pi)
        dx, dz = tx - r.x, tz - r.z
        dist = math.hypot(dx, dz)
        speed = min(self.spec.max_speed, 2.5 * dist)
        if releasing and dist < 0.6:
            speed = 0.0
        vx, vz = (speed * dx / dist, speed * dz / dist) if dist else (0.0, 0.0)
        stick = inverse_deadzone(min(1.0, math.hypot(vx, vz) / self.spec.max_speed))
        norm = math.hypot(vx, vz) or 1.0
        inputs = Inputs(x=sign * stick * vz / norm, y=sign * stick * vx / norm, field_relative=True)
        hub_x = HUB_X[al]
        if releasing:
            aim = wrap(math.atan2(0 - r.z, hub_x - r.x) + math.pi - math.radians(self.spec.shooter_yaw_deg))
        hub = state.hubs[al]
        distance_hub = math.hypot(r.x - hub_x, r.z)
        own_side = (r.x < -ZONE_X + .34) if al == "blue" else (r.x > ZONE_X - .34)
        settled = math.hypot(r.vx, r.vz) < 0.35
        inputs.score = (releasing and hub.active and own_side and 1.0 <= distance_hub <= 6.0
                        and r.hopper > 0 and (settled or dist < 0.4))
        # Manual turn fights the engine's aim assist while score is held.
        inputs.turn = 0.0 if inputs.score else inverse_deadzone(
            max(-1.0, min(1.0, -4 * angle_error(aim, r.a) / (.45 * self.spec.turn_rate))))
        inputs.intake = not inputs.score and (d >= 1 or r.hopper < pro_hopper - 5)
        return inputs
