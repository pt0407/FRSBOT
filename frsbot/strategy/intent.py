"""Causal 10 Hz intent features and portable NumPy logistic inference."""
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from frsbot.analysis.matches import ALLIANCE_ZONE_EDGE, DEFEND_DISTANCE, INTENT_LABELS
from frsbot.core.field import HUB_CENTRES
from frsbot.core.specs import load_specs


MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "intent_model.json"
FEATURES = ("own_hub_dx", "own_hub_dz", "opponent_hub_dx", "opponent_hub_dz",
            "in_own_zone", "in_opponent_half", "vx", "vz", "speed", "toward_own_hub",
            "toward_opponent", "toward_fuel", "hopper", "hopper_fraction", "hopper_delta_1s",
            "hopper_delta_3s", "opponent_distance", "opponent_closing_speed", "pressure_fraction_3s",
            "hub_active", "seconds_to_toggle", "time_left", "spec_capacity", "spec_max_speed",
            "history_seconds", "hopper_visible", "fuel_cluster_distance")
HOPPER_FEATURES = tuple(FEATURES.index(name) for name in
                        ("hopper", "hopper_fraction", "hopper_delta_1s", "hopper_delta_3s"))


def toward(vx, vz, dx, dz):
    return (vx * dx + vz * dz) / max(math.hypot(dx, dz), 1e-9)


def nearest_opponent(state, i):
    robot = state.robots[i]
    candidates = [(math.hypot(other.x - robot.x, other.z - robot.z), other)
                  for other in state.robots if other.alliance != robot.alliance]
    return min(candidates, key=lambda pair: (pair[0], pair[1].id)) if candidates else (20., None)


def defend_target(state, i):
    """Prefer close opponents, breaking near-ties in favor of closing encounters."""
    robot = state.robots[i]
    candidates = []
    for other in state.robots:
        if other.alliance == robot.alliance:
            continue
        dx, dz = other.x - robot.x, other.z - robot.z
        distance = math.hypot(dx, dz)
        closing = toward(robot.vx - other.vx, robot.vz - other.vz, dx, dz)
        candidates.append((distance - .25 * max(0., closing), other.id))
    return min(candidates)[1] if candidates else None


def fuel_target(state, robot):
    pos = state.fuel.pos
    if not len(pos):
        return robot.x, robot.z
    mask = ((state.fuel.where == 0) & (pos[:, 1] < .2)
            & (np.hypot(pos[:, 0] - robot.x, pos[:, 2] - robot.z) <= 5.))
    cells, counts = np.unique(np.floor(pos[mask][:, [0, 2]] / .5).astype(int), axis=0, return_counts=True)
    if not len(cells):
        return robot.x, robot.z
    density = np.array([counts[np.max(np.abs(cells - cell), axis=1) <= 1].sum() for cell in cells])
    centres = (cells + .5) * .5
    order = np.lexsort((np.hypot(centres[:, 0] - robot.x, centres[:, 1] - robot.z), -density))
    return centres[order[0]]


class IntentTracker:
    """Feed ordered live states; history retains only sampled observations, never future labels."""
    def __init__(self, specs=None):
        self.specs = {s.team: s for s in (load_specs() if specs is None else specs)}
        self.history = deque()
        self._last_seen = None
        self._identity = None

    def reset(self):
        self.history.clear()
        self._last_seen = self._identity = None

    def update(self, state):
        if not math.isfinite(state.t) or (self._last_seen is not None and state.t <= self._last_seen):
            raise ValueError("intent times must be finite and strictly increasing; reset between matches")
        identity = [(r.id, r.team, r.alliance) for r in state.robots]
        if self._identity is not None and identity != self._identity:
            raise ValueError("robot roster changed; reset tracker")
        self._identity, self._last_seen = identity, state.t
        if self.history and state.t - self.history[-1].t < .1 - 1e-7:
            return False
        self.history.append(state)
        while len(self.history) > 1 and self.history[1].t <= state.t - 3. + 1e-7:
            self.history.popleft()
        return True

    def features(self, i, mask_hopper=False):
        if not self.history:
            raise RuntimeError("update tracker before extracting features")
        state = self.history[-1]
        robot = state.robots[i]
        own = HUB_CENTRES[robot.alliance]
        opponent = HUB_CENTRES["red" if robot.alliance == "blue" else "blue"]
        distance, other = nearest_opponent(state, i)
        dx, dz = (other.x - robot.x, other.z - robot.z) if other else (0., 0.)
        closing = toward(robot.vx - other.vx, robot.vz - other.vz, dx, dz) if other else 0.
        fx, fz = fuel_target(state, robot)
        def hopper_ago(seconds):
            candidates = [s for s in self.history if s.t <= state.t - seconds + 1e-7]
            return (candidates[-1] if candidates else self.history[0]).robots[i].hopper
        times = np.array([s.t for s in self.history])
        weights = np.diff(times)
        pressure = [nearest_opponent(s, i)[0] <= DEFEND_DISTANCE for s in list(self.history)[:-1]]
        fraction = float(np.dot(weights, pressure) / weights.sum()) if weights.sum() else float(distance <= DEFEND_DISTANCE)
        spec = self.specs[robot.team]
        values = [own[0] - robot.x, own[1] - robot.z, opponent[0] - robot.x, opponent[1] - robot.z,
                  robot.x < -ALLIANCE_ZONE_EDGE if robot.alliance == "blue" else robot.x > ALLIANCE_ZONE_EDGE,
                  robot.x > 0 if robot.alliance == "blue" else robot.x < 0,
                  robot.vx, robot.vz, math.hypot(robot.vx, robot.vz),
                  toward(robot.vx, robot.vz, own[0] - robot.x, own[1] - robot.z),
                  toward(robot.vx, robot.vz, dx, dz), toward(robot.vx, robot.vz, fx - robot.x, fz - robot.z),
                  robot.hopper, robot.hopper / max(robot.capacity, 1),
                  robot.hopper - hopper_ago(1.), robot.hopper - hopper_ago(3.), distance, closing, fraction,
                  state.hubs[robot.alliance].active, min(166., state.hubs[robot.alliance].seconds_to_toggle),
                  max(0., 163. - state.t), spec.total_capacity, spec.max_speed,
                  min(3., state.t - times[0]), not mask_hopper, math.hypot(fx - robot.x, fz - robot.z)]
        result = np.array(values, dtype=np.float64)
        if mask_hopper:
            result[list(HOPPER_FEATURES)] = 0.
        if not np.isfinite(result).all():
            raise ValueError("non-finite intent features")
        return result


class IntentModel:
    def __init__(self, model):
        if model["features"] != list(FEATURES) or model["labels"] != list(INTENT_LABELS):
            raise ValueError("incompatible intent model features/labels")
        self.model = model

    @classmethod
    def load(cls, path=MODEL_PATH):
        return cls(json.loads(Path(path).read_text()))

    def probabilities(self, features, mask_hopper=False):
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1:] != (len(FEATURES),) or not np.isfinite(values).all():
            raise ValueError("invalid intent features")
        values = values.copy()
        if mask_hopper:
            values[..., list(HOPPER_FEATURES)] = 0
            values[..., FEATURES.index("hopper_visible")] = 0
        params = self.model["models"]["masked" if mask_hopper else "visible"]
        x = (values - np.asarray(params["mean"])) / np.asarray(params["scale"])
        logits = x @ np.asarray(params["coef"]).T + np.asarray(params["intercept"])
        if len(params["classes"]) == 2 and logits.shape[-1] == 1:
            logits = np.concatenate([np.zeros_like(logits), logits], axis=-1)
        logits -= logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        result = np.zeros((*values.shape[:-1], len(INTENT_LABELS)))
        result[..., params["classes"]] = probabilities
        return result

    def predict(self, tracker, i, mask_hopper=False):
        probabilities = self.probabilities(tracker.features(i, mask_hopper), mask_hopper)
        return {"probabilities": dict(zip(INTENT_LABELS, probabilities.tolist())),
                "defend_target": defend_target(tracker.history[-1], i)}
