"""Resolve carried FUEL IDs by replay intake events and position."""
import numpy as np


def carried_owners(states, events, starts):
    """Return ball-index to robot-index maps for the requested segment starts.

    RBRP FUEL carries an index, position and where code, but no owner field.
    An intake event identifies a ball and robot ID. Keep that owner until the
    ball stops being carried, and resolve already-held balls geometrically.
    """
    by_id = {robot.id: i for i, robot in enumerate(states[0].robots)}
    events = sorted((int(frame), event["fuel"], by_id[event["robot"]])
                    for frame, event in events if event.get("t") == "intake"
                    and event.get("robot") in by_id and "fuel" in event)
    owners = {}
    if len(by_id) == 1:
        return {k: np.where(states[k].fuel.where == 2, 0, -1).tolist() for k in starts}
    previous = np.full(len(states[0].fuel.where), -1, dtype=np.int16)
    cursor = 0
    for k, state in enumerate(states[:max(starts, default=-1) + 1]):
        carried = state.fuel.where == 2
        previous[~carried] = -1
        # A spectator event may precede its first snapshot by a few frames.
        recent = cursor
        while recent < len(events):
            frame, index, robot = events[recent]
            if frame > k:
                break
            if 0 <= index < len(carried) and carried[index]:
                previous[index] = robot
            recent += 1
        while cursor < len(events) and events[cursor][0] <= k - 6:
            cursor += 1
        unknown = np.flatnonzero(carried & (previous < 0))
        if len(unknown):
            xy = np.array([(robot.x, robot.z) for robot in state.robots])
            pos = state.fuel.pos[unknown][:, [0, 2]]
            previous[unknown] = np.argmin(np.sum((pos[:, None] - xy) ** 2, axis=2), axis=1)
        if k in starts:
            owners[k] = previous.tolist()
    return owners
