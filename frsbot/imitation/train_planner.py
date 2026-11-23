"""Train a small planner on recovered pro examples for one robot team, scored leave-one-replay-out.

Predicts the goal kind (classification) and the goal point (regression) from the robot's own state.
Baselines: majority goal kind, and the mean goal point of the training replays.

Usage: python -m frsbot.imitation.train_planner --team 2910 --files data/imitation/solo-*.npz
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from torch import nn

from frsbot.imitation.dataset import examples

KINDS = ["Collect", "Score", "Stage"]


def features(state):
    r = state.robots[state.me]
    hub = state.hubs[r.alliance]
    return [r.x, r.z, np.cos(r.a), np.sin(r.a), r.hopper / max(1, r.capacity), float(hub.active),
            hub.seconds_to_toggle / 60, state.t / 166]


def load(path, team):
    rows = []
    for state, _, _, goal in examples(path):
        if state.robots[state.me].team != team or goal.kind not in KINDS:
            continue
        rows.append((features(state), KINDS.index(goal.kind), goal.point))
    return rows


def fit(x, y_kind, y_point, epochs=300, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(KINDS) + 2))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    xt, kt, pt = torch.tensor(x, dtype=torch.float32), torch.tensor(y_kind), torch.tensor(y_point, dtype=torch.float32)
    for _ in range(epochs):
        out = net(xt)
        loss = nn.functional.cross_entropy(out[:, :len(KINDS)], kt) + nn.functional.mse_loss(out[:, len(KINDS):], pt)
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def predict(net, x):
    with torch.no_grad():
        out = net(torch.tensor(x, dtype=torch.float32)).numpy()
    return out[:, :len(KINDS)].argmax(1), out[:, len(KINDS):]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", type=int, required=True)
    parser.add_argument("--files", nargs="+", default=sorted(glob.glob("data/imitation/solo-*.npz")))
    args = parser.parse_args()
    folds = {Path(f).stem: load(f, args.team) for f in args.files}
    folds = {k: v for k, v in folds.items() if v}
    print(f"team {args.team}: {sum(len(v) for v in folds.values())} examples from {len(folds)} replays")
    results = []
    for held, test in folds.items():
        train = [row for k, rows in folds.items() if k != held for row in rows]
        x = np.array([r[0] for r in train]); yk = np.array([r[1] for r in train]); yp = np.array([r[2] for r in train])
        net = fit(x, yk, yp)
        tx = np.array([r[0] for r in test]); tk = np.array([r[1] for r in test]); tp = np.array([r[2] for r in test])
        pk, pp = predict(net, tx)
        majority = np.bincount(yk, minlength=len(KINDS)).argmax()
        acc = (pk == tk).mean(); base = (tk == majority).mean()
        err = np.linalg.norm(pp - tp, axis=1).mean(); base_err = np.linalg.norm(yp.mean(0) - tp, axis=1).mean()
        results.append((acc, base, err, base_err))
        print(f"held out {held}: kind acc {acc:.3f} (majority {base:.3f})  goal error {err:.2f} m (mean-point {base_err:.2f} m)")
    r = np.array(results).mean(0)
    print(f"mean over {len(results)} folds: kind acc {r[0]:.3f} vs majority {r[1]:.3f}; goal error {r[2]:.2f} m vs mean-point {r[3]:.2f} m")


if __name__ == "__main__":
    main()
