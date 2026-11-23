"""State-only weak supervision for the three planner goals."""
import numpy as np

from frsbot.bot.planner import Goal, SCORING_X
from frsbot.core.field import HUB_CENTRES


def label_segments(states, specs):
    """Return one goal per robot, using observed transfers and motion, not inputs."""
    xy = np.array([[(r.x, r.z) for r in s.robots] for s in states])
    hoppers = np.array([[r.hopper for r in s.robots] for s in states])
    where = np.array([s.fuel.where for s in states])
    pos = np.array([s.fuel.pos for s in states])
    return [
        _label(robot, specs[robot.team], xy[:, i], hoppers[:, i], pos, where,
               np.argmin(((pos[:-1, :, None, [0, 2]] - xy[:-1, None]) ** 2).sum(-1), axis=2) == i)
        for i, robot in enumerate(states[0].robots)
    ]


def _label(robot, spec, path, hopper, pos, where, nearest):
    local = (np.linalg.norm(pos[:-1][:, :, [0, 2]] - path[:-1, None], axis=2) < 1.3) & nearest
    picks = local & (where[:-1] == 0) & (where[1:] == 2)
    launches = local & (where[:-1] == 2) & (where[1:] == 1) & (pos[1:, :, 1] > .3)
    cx, cz, _ = HUB_CENTRES[robot.alliance]
    centre = np.array([cx, cz])
    dist = np.linalg.norm(path - centre, axis=1)
    low = spec.min_range or min(spec.hub_shot["distance"])
    high = spec.max_range or max(spec.hub_shot["distance"])
    side = -1 if robot.alliance == "blue" else 1
    arc = (path[:, 0] * side >= SCORING_X) & (dist >= low) & (dist <= high)
    arrivals = (where[:-1] == 1) & (where[1:] == 3)
    own_hub = np.linalg.norm(pos[1:][:, :, [0, 2]] - centre, axis=2) < .8
    scored = (arrivals & own_hub & launches.any(axis=0)).any(axis=1)
    delta = int(hopper[-1]) - int(hopper[0])
    if (arc[:-1] & launches.any(axis=1)).any() or scored.any() or (delta < -1 and arc.any()):
        frames = np.flatnonzero(arc[:-1] & launches.any(axis=1))
        if not len(frames):
            frames = np.flatnonzero(arc)
        return Goal("Score", tuple(map(float, path[frames].mean(axis=0) if len(frames) else path[-1])))
    pickups = pos[:-1][picks][:, [0, 2]]
    if len(pickups):
        # Choose the densest observed pickup cluster, rather than the mean of distant trips.
        distances = np.linalg.norm(pickups[:, None] - pickups, axis=2)
        cluster = distances <= .75
        return Goal("Collect", tuple(map(float, pickups[cluster[cluster.sum(axis=1).argmax()]].mean(axis=0))))
    field = pos[0, where[0] == 0][:, [0, 2]]
    if len(field):
        direction = path[-1] - path[0]
        near = np.linalg.norm(field - path[-1], axis=1) <= .9
        ahead = (field - path[0]) @ direction > 0
        if delta > 0 or (np.linalg.norm(direction) > .6 and (near & ahead).sum() >= 3 and not arc[-1]):
            cluster = field[near & ahead]
            target = cluster.mean(axis=0) if len(cluster) else field[np.linalg.norm(field - path[-1], axis=1).argmin()]
            return Goal("Collect", tuple(map(float, target)))
    return Goal("Stage", tuple(map(float, path[-1])))
