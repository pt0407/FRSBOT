"""Weak input labels from replayed FUEL transfers."""
import numpy as np

from frsbot.imitation.ownership import carried_owners


def transition_buttons(states, ticks, events, radius=3):
    where = np.stack([state.fuel.where for state in states])
    valid = (np.diff(ticks) <= 12)[:, None] & (np.diff(ticks) > 0)[:, None]
    pickups = np.argwhere(valid & np.isin(where[:-1], [0, 1]) & (where[1:] == 2))
    releases = np.argwhere(valid & (where[:-1] == 2) & (where[1:] == 1))
    frames = set(pickups[:, 0] + 1) | set(releases[:, 0])
    owners = carried_owners(states, events, frames)
    spans = np.zeros((ticks[-1] - ticks[0] + 2, len(states[0].robots), 2), dtype=np.int32)
    for kind, transfers in enumerate((pickups, releases)):
        for frame, ball in transfers:
            owner = owners[frame + (kind == 0)][ball]
            if owner >= 0:
                center = ticks[frame + 1] - ticks[0]
                spans[max(0, center - radius), owner, kind] += 1
                spans[min(len(spans) - 1, center + radius + 1), owner, kind] -= 1
    return spans[:-1].cumsum(axis=0) > 0


def interval_buttons(buttons, ticks, indices):
    return [buttons[ticks[a] - ticks[0]:ticks[b] - ticks[0]].any(axis=0).tolist()
            for a, b in zip(indices, indices[1:])]


def agreement(tick, duration, packets, legacy):
    """Return TP, FP, FN and TN robot-tick counts per button."""
    actual = packets[:, :, 1, None] & np.array([1, 2]) != 0
    old = legacy["inputs"][:, :, 1, None] & np.array([1, 2]) != 0
    tick = np.repeat(tick - np.cumsum(duration) + duration, duration) + np.arange(duration.sum())
    actual = np.repeat(actual, duration, axis=0)
    pos = np.searchsorted(legacy["tick"], tick, side="right") - 1
    valid = (pos >= 0) & (tick < legacy["tick"][pos] + legacy["ticks"][pos])
    actual, old = actual[valid], old[pos[valid]]
    return np.stack([(actual & old).sum(axis=(0, 1)), (actual & ~old).sum(axis=(0, 1)),
                     (~actual & old).sum(axis=(0, 1)), (~actual & ~old).sum(axis=(0, 1))], axis=1)
