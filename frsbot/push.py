"""FUEL push model fitted to 30 Hz replay frames.

Per 2-tick step (dt = 1/30 s), for a ground ball (where 0/1, height < 0.2 m):
  1. Free roll: velocity decays by DECAY per step.
  2. Contact: if the rolled position lands inside the robot's bumper rectangle (grown by the ball radius),
     the ball is moved out through the nearest side and leaves at the robot's speed along that side's normal
     (plus its own tangential speed).
Robot frame: +fx forward (cos a, sin a), +fy left, a = pi - quaternion yaw. Intake mouth is the front side.

Usage: python -m frsbot.push   (fits on data/raw/*.rbrp, prints held-out validation)
"""
import json
import sys
from pathlib import Path

import numpy as np

from frsbot.rbrp import parse

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "push_model.json"
DT = 2 / 60
GROUND_Y = 0.2
NEAR = 1.5  # m, balls this close to the robot centre are scored


def heading(quat: np.ndarray) -> np.ndarray:
    return np.pi - 2 * np.arctan2(quat[..., 1], quat[..., 3])


def load(path: Path):
    """Robot 0 pose (x, z, heading) and FUEL arrays for every 30 Hz frame."""
    rep = parse(path.read_bytes(), fuel=True)
    o = rep.robot_offsets[0]["at"]
    st = np.array([s for _, s in rep.frames], np.int32)
    xz = st[:, [o, o + 2]] * 5e-4
    quat = st[:, o + 3:o + 7].astype(float)
    pos = np.array([p for p, _ in rep.fuel], np.float32) * 5e-4
    where = np.array([w for _, w in rep.fuel], np.uint8)
    team = rep.robots[0]["team"]
    return {"xz": xz, "a": heading(quat), "pos": pos, "where": where, "radius": rep.radius, "team": team}


def to_robot(p, xz, a):
    r = p - xz[..., None, :]
    c, s = np.cos(a)[..., None], np.sin(a)[..., None]
    return np.stack([r[..., 0] * c + r[..., 1] * s, -r[..., 0] * s + r[..., 1] * c], -1)


def from_robot(f, xz, a):
    c, s = np.cos(a)[..., None], np.sin(a)[..., None]
    return np.stack([f[..., 0] * c - f[..., 1] * s, f[..., 0] * s + f[..., 1] * c], -1) + xz[..., None, :]


def predict(prm, p_prev, p_now, xz_now, a_now, xz_next, a_next, radius):
    """Next ground position of each ball, given the robot's next pose."""
    v = (p_now - p_prev) * prm["decay"]
    p = p_now + v
    f = to_robot(p, xz_next, a_next)
    front, back, half = prm["front"] + radius, prm["back"] + radius, prm["half"] + radius
    inside = (f[..., 0] < front) & (f[..., 0] > -back) & (np.abs(f[..., 1]) < half)
    gaps = np.stack([front - f[..., 0], f[..., 0] + back, half - f[..., 1], half + f[..., 1]], -1)
    side = gaps.argmin(-1)
    out = f.copy()
    out[..., 0] = np.where(side == 0, front, np.where(side == 1, -back, f[..., 0]))
    out[..., 1] = np.where(side == 2, half, np.where(side == 3, -half, f[..., 1]))
    pushed = from_robot(out, xz_next, a_next)
    return np.where(inside[..., None], pushed, p), inside


def samples(run, step=1):
    """Index triples (k-1, k, k+1) and a mask of balls that stay loose on the ground near the robot."""
    pos, where = run["pos"], run["where"]
    ks = np.arange(1, len(pos) - 1, step)
    g = (where[ks - 1] <= 1) & (where[ks] <= 1) & (where[ks + 1] <= 1)
    g &= (pos[ks - 1, :, 1] < GROUND_Y) & (pos[ks, :, 1] < GROUND_Y) & (pos[ks + 1, :, 1] < GROUND_Y)
    d = np.linalg.norm(pos[ks][..., [0, 2]] - run["xz"][ks][:, None], axis=-1)
    return ks, g & (d < NEAR)


def errors(prm, run, ks, mask):
    xz, a, P = run["xz"], run["a"], run["pos"][..., [0, 2]]
    pred, inside = predict(prm, P[ks - 1], P[ks], xz[ks], a[ks], xz[ks + 1], a[ks + 1], run["radius"][None])
    err = np.linalg.norm(pred - P[ks + 1], axis=-1)
    still = np.linalg.norm(P[ks] - P[ks + 1], axis=-1)
    return err[mask], still[mask], inside[mask]


def fit(runs):
    best = None
    for front in (0.40, 0.44, 0.48, 0.52):
        for back in (0.40, 0.44, 0.48, 0.52):
            for half in (0.40, 0.44, 0.48, 0.52):
                for decay in (0.85, 0.9, 0.95, 0.98):
                    prm = dict(front=front, back=back, half=half, decay=decay)
                    e = np.concatenate([errors(prm, r, *r["_s"])[0] for r in runs])
                    m = float(np.mean(e))
                    if best is None or m < best[0]:
                        best = (m, prm)
    return best[1]


def report(prm, runs, label):
    e, still, inside = (np.concatenate(x) for x in zip(*[errors(prm, r, *r["_s"]) for r in runs]))
    moving = still > 0.01
    print(f"{label}: n={len(e)} mean err {e.mean()*100:.2f} cm (stay-put baseline {still.mean()*100:.2f} cm)")
    print(f"  moving balls n={moving.sum()}: model {e[moving].mean()*100:.2f} cm vs stay-put {still[moving].mean()*100:.2f} cm")
    print(f"  contact steps n={inside.sum()}: model {e[inside].mean()*100:.2f} cm vs stay-put {still[inside].mean()*100:.2f} cm")
    return {"n": int(len(e)), "mean_cm": round(float(e.mean()) * 100, 2), "stay_put_cm": round(float(still.mean()) * 100, 2),
            "contact_model_cm": round(float(e[inside].mean()) * 100, 2), "contact_stay_put_cm": round(float(still[inside].mean()) * 100, 2)}


def main(n_runs=12):
    paths = sorted(RAW.glob("*.rbrp"))[:n_runs]
    runs = []
    for p in paths:
        r = load(p)
        r["_s"] = samples(r, step=3)
        runs.append(r)
        print("loaded", p.stem, "team", r["team"], "frames", len(r["pos"]), file=sys.stderr)
    split = max(1, len(runs) * 2 // 3)
    prm = fit(runs[:split])
    print("fit", prm)
    train = report(prm, runs[:split], "train")
    test = report(prm, runs[split:], "held-out")
    OUT.write_text(json.dumps({"params": prm, "dt": DT, "train": train, "heldout": test,
                               "runs": [p.stem for p in paths]}, indent=1))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
