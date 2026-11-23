"""Observational best-response roles; no assignment of human teammates' roles."""
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frsbot.core.field import HUB_CENTRES, TIMELINE, is_active, phase_at
from frsbot.strategy.intent import ALLIANCE_ZONE_EDGE, DEFEND_DISTANCE, IntentModel, IntentTracker


STATS_PATH = Path(__file__).resolve().parents[2] / "data" / "analysis" / "match_stats.json"
MATCH_END = 163.


@dataclass(frozen=True)
class Role:
    kind: str
    target: int | None = None

    def __post_init__(self):
        if self.kind not in {"Scorer", "Feeder", "Defender", "Contester"}:
            raise ValueError("unknown strategy role")


def own_zone(robot):
    return robot.x < -ALLIANCE_ZONE_EDGE if robot.alliance == "blue" else robot.x > ALLIANCE_ZONE_EDGE


def active_seconds(alliance, start, horizon, first_inactive=None):
    """Exact schedule integral; average both possible shift orders until observed."""
    end = min(MATCH_END, start + max(0., horizon))
    cuts = sorted({max(0., start), end, *[t for t, _ in TIMELINE if start < t < end]})
    if end <= max(0., start):
        return 0.
    orders = (first_inactive,) if first_inactive else ("blue", "red")
    return sum(sum((b - a) * is_active(alliance, (a + b) / 2, order)
                   for a, b in zip(cuts, cuts[1:])) for order in orders) / len(orders)


def scoring_prior(spec):
    """One hopper per firing time plus a 6 m collection trip and 1 s startup."""
    if spec.fire_rate <= 0 or spec.max_speed <= 0 or spec.total_capacity <= 0:
        return 0.
    return spec.total_capacity / (spec.total_capacity / spec.fire_rate + 6. / spec.max_speed + 1.)


class StrategyLayer:
    """Reconsider roles every 2 s, monitoring pending winners on every observation.

    Lead late: favor denial if our lead exceeds twice their remaining production.
    Chase a deficit: score during our active windows.
    Uncatchable deficit: always score, never spend the remaining time defending.
    Weakest scorer: defend the best unguarded opponent when saved points beat ours.
    Solo: without opponents always score; without teammates never feed.
    """
    def __init__(self, spec, specs_by_team):
        self.spec = spec
        self.specs = dict(specs_by_team)
        self.specs[spec.team] = spec
        self.tracker = IntentTracker(self.specs.values())
        self.model = IntentModel.load()
        try:
            ratio = float(json.loads(STATS_PATH.read_text())["suppression_ratio"])
            self.suppression_ratio = ratio if math.isfinite(ratio) and 0 <= ratio <= 1 else .65
        except (OSError, ValueError, TypeError, KeyError):
            self.suppression_ratio = .65
        self.intents = {}
        self.rates = {}
        self._drops = {}
        self._exposure = {}
        self.score_estimate = {"blue": 0, "red": 0}
        self._scores = deque()
        self._previous = None
        self._first_inactive = None
        self._role = Role("Scorer")
        self._pending = self._role
        self._pending_since = None
        self._last_decision = None

    def observe(self, state):
        accepted = self.tracker.update(state)
        previous = self._previous
        if previous is not None:
            if len(previous.fuel.where) != len(state.fuel.where):
                raise ValueError("FUEL identity count changed")
            dt = max(0., min(state.t, MATCH_END) - max(previous.t, 0.))
            for old, robot in zip(previous.robots, state.robots):
                if own_zone(old) and previous.hubs[old.alliance].active:
                    self._exposure[robot.id] = self._exposure.get(robot.id, 0.) + dt
                    self._drops[robot.id] = self._drops.get(robot.id, 0.) + max(0, old.hopper - robot.hopper) * (dt > 0)
            if 0 <= state.t < MATCH_END:
                for fuel in np.flatnonzero((previous.fuel.where != 3) & (state.fuel.where == 3)):
                    point = state.fuel.pos[fuel, [0, 2]]
                    alliance = min(HUB_CENTRES, key=lambda a: math.dist(point, HUB_CENTRES[a][:2]))
                    if state.hubs[alliance].active and math.dist(point, HUB_CENTRES[alliance][:2]) <= 1.25:
                        self.score_estimate[alliance] += 1
        for robot in state.robots:
            prior = scoring_prior(self.specs[robot.team])
            self.rates[robot.id] = (20. * prior + self._drops.get(robot.id, 0.)) / (20. + self._exposure.get(robot.id, 0.))
        if accepted:
            self.intents = {robot.id: self.model.predict(self.tracker, i) for i, robot in enumerate(state.robots)}
        if phase_at(state.t).value.startswith("shift") and state.hubs["blue"].active != state.hubs["red"].active:
            first_off = phase_at(state.t).value in ("shift1", "shift3")
            inactive = "red" if state.hubs["blue"].active else "blue"
            self._first_inactive = inactive if first_off else ("blue" if inactive == "red" else "red")
        self._scores.append((state.t, dict(self.score_estimate)))
        while len(self._scores) > 1 and self._scores[1][0] <= state.t - 10.:
            self._scores.popleft()
        self._previous = state
        self._choose(state)

    def score_difference(self, state):
        alliance = state.robots[state.me].alliance
        return self.score_estimate[alliance] - self.score_estimate["red" if alliance == "blue" else "blue"]

    def score_trend(self, state):
        if len(self._scores) < 2:
            return 0.
        t, old = self._scores[0]
        alliance = state.robots[state.me].alliance
        other = "red" if alliance == "blue" else "blue"
        return (self.score_difference(state) - old[alliance] + old[other]) / max(state.t - t, 1e-9)

    def _probability(self, robot, label):
        return self.intents.get(robot.id, {}).get("probabilities", {}).get(label, 0.)

    def _teammates(self, state):
        me = state.robots[state.me]
        return [r for r in state.robots if r.id != me.id and r.alliance == me.alliance]

    def _stronger_scorers(self, state):
        me = state.robots[state.me]
        return [r for r in self._teammates(state) if self.rates[r.id] > self.rates[me.id]
                and self._probability(r, "Score") > .5]

    def _guarded(self, state, opponent):
        return any(self._probability(r, "Defend") > .5
                   and self.intents.get(r.id, {}).get("defend_target") == opponent.id for r in self._teammates(state))

    def role_values(self, state):
        """Expected margin relative to scoring; forgone own production is charged once.

        Modelled points are FUEL proxies, not scoreboard truth. Hopper drops miss
        simultaneous intake/fire. Hub entries include HP and exclude tower/fouls.
        """
        me = state.robots[state.me]
        opponents = [r for r in state.robots if r.alliance != me.alliance]
        scorer = Role("Scorer")
        if not opponents:
            return {scorer: 0.}
        remaining = max(0., MATCH_END - state.t)
        horizon = min(remaining, 30.)
        active = {a: active_seconds(a, state.t, horizon, self._first_inactive) for a in HUB_CENTRES}
        own_points = self.rates[me.id] * active[me.alliance]
        crowd = .15 * own_points if self._stronger_scorers(state) else 0.
        values = {scorer: -crowd}
        other_alliance = opponents[0].alliance
        own_remaining = sum(self.rates[r.id] for r in state.robots if r.alliance == me.alliance) * active_seconds(
            me.alliance, state.t, remaining, self._first_inactive)
        difference = self.score_difference(state)
        if difference < 0 and (-difference > own_remaining or state.hubs[me.alliance].active):
            return values
        defense_allowed = 20 <= state.t < MATCH_END - 3
        benefits = {}
        if defense_allowed:
            for opponent in opponents:
                chance = self._probability(opponent, "Score") + self._probability(opponent, "Collect")
                benefit = self.rates[opponent.id] * active[other_alliance] * min(1., chance) * (1. - self.suppression_ratio)
                if self._guarded(state, opponent):
                    benefit *= .01
                benefits[opponent.id] = benefit
                values[Role("Defender", opponent.id)] = benefit - own_points
        if self._stronger_scorers(state) and horizon:
            collection = min(self.spec.intake_rate, self.spec.total_capacity / (6. / max(self.spec.max_speed, .1) + 1.))
            inactive_fraction = 1. - active[me.alliance] / horizon
            values[Role("Feeder")] = collection * horizon * inactive_fraction * .8 - own_points
        neutral = [r for r in opponents if abs(r.x) < ALLIANCE_ZONE_EDGE]
        denied = sum(min(self.specs[r.team].intake_rate, self.specs[r.team].total_capacity /
                         (6. / max(self.specs[r.team].max_speed, .1) + 1.))
                     * self._probability(r, "Collect") * active[other_alliance] * .5 for r in neutral)
        values[Role("Contester")] = denied - own_points
        opp_remaining = sum(self.rates[r.id] for r in opponents) * active_seconds(
            other_alliance, state.t, remaining, self._first_inactive)
        if state.t > 133 and difference > 2. * opp_remaining:
            for role in values:
                if role.kind in ("Defender", "Contester"):
                    values[role] += own_points + crowd + 1.
        if benefits and self.rates[me.id] <= min(self.rates[r.id] for r in [me, *self._teammates(state)]):
            best = max(opponents, key=lambda r: self.rates[r.id])
            if not self._guarded(state, best) and benefits[best.id] > own_points:
                preferred = Role("Defender", best.id)
                values[preferred] = max(values.values()) + .1 * max(1., own_points)
        return values

    def _choose(self, state):
        values = self.role_values(state)
        if self._role not in values:
            self._role = Role("Scorer")
        best = max(values, key=values.get)
        if best != self._pending:
            self._pending, self._pending_since = best, state.t
        elif self._pending_since is None:
            self._pending_since = state.t
        if self._last_decision is not None and state.t - self._last_decision < 2. - 1e-7:
            return
        self._last_decision = state.t
        advantage = values[best] - values[self._role]
        me = state.robots[state.me]
        baseline = self.rates[me.id] * active_seconds(me.alliance, state.t,
                    min(30., max(0., MATCH_END - state.t)), self._first_inactive)
        threshold = .1 * max(1., abs(baseline + values[self._role]))
        if state.t - self._pending_since >= 2. - 1e-7 and advantage >= threshold:
            self._role = best

    def role(self, state):
        if self._previous is None or self._previous.t != state.t:
            raise RuntimeError("observe this state before requesting its role")
        return self._role

    def defended_by(self, state):
        if not self.tracker.history or state.t - self.tracker.history[0].t < 3. - 1e-7:
            return None
        me = state.robots[state.me]
        exposure = {r.id: 0. for r in state.robots if r.alliance != me.alliance}
        history = list(self.tracker.history)
        for a, b in zip(history, history[1:]):
            robot = a.robots[state.me]
            dt = max(0., min(b.t, state.t) - max(a.t, state.t - 3.))
            if own_zone(robot):
                for other in a.robots:
                    if other.id in exposure and math.hypot(robot.x - other.x, robot.z - other.z) <= DEFEND_DISTANCE:
                        exposure[other.id] += dt
        candidates = [r for r in state.robots if exposure.get(r.id, 0.) >= 1.5 - 1e-7]
        return min(candidates, key=lambda r: (-exposure[r.id], math.hypot(r.x - me.x, r.z - me.z), r.id)).id if candidates else None

    def evasion(self, state):
        defender = self.defended_by(state)
        if defender is None:
            return "stay"
        if self._stronger_scorers(state):
            return "feed"
        me = state.robots[state.me]
        active = active_seconds(me.alliance, state.t, min(3., max(0., MATCH_END - state.t)), self._first_inactive)
        loss = self.rates[me.id] * active * (1. - self.suppression_ratio)
        move_cost = self.rates[me.id] * (1.5 / max(self.spec.max_speed, .1) + .5)
        if move_cost >= loss:
            return "stay"
        other = next(r for r in state.robots if r.id == defender)
        hub = np.array(HUB_CENTRES[me.alliance][:2])
        origin = np.array([me.x, me.z])
        direction = hub - origin
        fraction = (np.array([other.x, other.z]) - origin) @ direction / max(direction @ direction, 1e-9)
        distance = np.linalg.norm(np.array([other.x, other.z]) - origin - fraction * direction)
        return "alternate_slot" if 0 < fraction < 1 and distance <= .7 else "stay"

    def teammate_targets(self, state):
        points = state.fuel.pos[(state.fuel.where == 0) & (state.fuel.pos[:, 1] < .2)][:, [0, 2]]
        if not len(points):
            return []
        cells, inverse = np.unique(np.floor(points / .5).astype(int), axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        centres = np.column_stack([np.bincount(inverse, weights=points[:, j]) / counts for j in (0, 1)])
        targets = []
        for robot in self._teammates(state):
            speed = math.hypot(robot.vx, robot.vz)
            if speed < .1 or self._probability(robot, "Collect") <= .5:
                continue
            delta = centres - [robot.x, robot.z]
            along = delta @ (np.array([robot.vx, robot.vz]) / speed)
            across = np.abs(delta[:, 0] * robot.vz - delta[:, 1] * robot.vx) / speed
            eligible = np.flatnonzero((along > 0) & (np.linalg.norm(delta, axis=1) <= 3.) & (across <= .75))
            if len(eligible):
                target = min(eligible, key=lambda i: (across[i], along[i]))
                targets.append(tuple(float(v) for v in centres[target]))
        return targets

    def teammate_slots(self, state):
        low = self.spec.min_range or min(self.spec.hub_shot["distance"])
        high = self.spec.max_range or max(self.spec.hub_shot["distance"])
        radius = max(low + .3, min(high - .5, (low + high) / 2))
        hub = HUB_CENTRES[state.robots[state.me].alliance][:2]
        return [(r.x, r.z) for r in self._teammates(state) if abs(math.dist((r.x, r.z), hub) - radius) <= 1.5]
