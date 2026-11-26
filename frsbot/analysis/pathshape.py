"""Path-shape metrics: curvature, reversals, cruise speed, stop fraction.

The quantitative form of 'the bot doesn't move like a human': pro paths are
gentle arcs (turn ~0.06-0.08 rad/sample), bot paths zigzag (~0.13-0.17) with
3-10x the reversals and 30% lower cruise speed. Use to score any trace.
"""
import numpy as np


def path_stats(positions: np.ndarray, dt: float) -> dict:
    """positions (n,2) world x/z sampled every dt seconds."""
    v = np.diff(positions, axis=0) / dt
    speed = np.linalg.norm(v, axis=1)
    moving = speed > .4
    ang = np.arctan2(v[:, 1], v[:, 0])
    da = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    both = moving[:-1] & moving[1:]
    return dict(
        stop_frac=float(1 - np.mean(moving)),
        reversal_rate=float(np.mean((da > 1.75) & both)) if both.any() else 0.0,
        turn_rate=float(np.mean(da[both])) if both.any() else 0.0,
        cruise=float(speed[moving].mean()) if moving.any() else 0.0,
    )


def stats_from_imai_trace(path) -> dict:
    import json
    d = json.loads(open(path).read())
    samples = d["samples"]
    pos = np.array([[s["robots"][0]["pos"][0], s["robots"][0]["pos"][2]] for s in samples])
    dt = max((samples[1]["tick"] - samples[0]["tick"]) / 60, 1e-9) if len(samples) > 1 else .1
    return path_stats(pos, dt)


PRO_REFERENCE = dict(turn_rate=0.07, reversal_rate=0.0003, cruise=2.6, stop_frac=0.35)
