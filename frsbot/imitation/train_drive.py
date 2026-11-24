"""Goal-conditioned drive model: (state, spec, routed point, goal kind) -> (x, y, turn).

Unlike input_policy.py, the human's labelled intent is part of the input so the
net learns human velocity/turn responses *given a target* — the planner keeps
choosing where, the net reproduces how pros actually move there.
Usage: python -m frsbot.imitation.train_drive [--holdout 4] [--epochs 40]
"""
import argparse
import glob
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from frsbot.core.specs import load_specs
from frsbot.imitation.dataset import examples
from frsbot.imitation.input_policy import features as base_features

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data" / "models" / "drive_policy.pt"
SPECS = {s.team: s for s in load_specs()}
KINDS = ("Collect", "Score", "Stage")
N_FEAT = 20  # 14 base + bearing cos/sin + dist + 3 goal-kind one-hot


def features(state, spec, goal):
    r = state.robots[state.me]
    gx, gz = goal.point
    bearing = math.atan2(gz - r.z, gx - r.x) - r.a
    onehot = [float(goal.kind == kind) for kind in KINDS]
    return base_features(state, spec) + [math.cos(bearing), math.sin(bearing),
                                         math.hypot(gx - r.x, gz - r.z) / 8.] + onehot


def load(files):
    X, Y, W, runs = [], [], [], []
    for f in files:
        for state, inp, residual, goal in examples(f):
            spec = SPECS.get(state.robots[state.me].team)
            if spec is None:
                continue
            X.append(features(state, spec, goal))
            Y.append([inp.x, inp.y, inp.turn])
            W.append(math.exp(-abs(residual.get("pose_m", 0)) / .1))
            runs.append(f)
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(W, np.float32), np.array(runs)


def build():
    return nn.Sequential(nn.Linear(N_FEAT, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))


def fit(X, Y, W, epochs, seed=0):
    torch.manual_seed(seed)
    net = build()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    xt, yt, wt = torch.tensor(X), torch.tensor(Y), torch.tensor(W)
    n = len(X)
    for _ in range(epochs):
        order = torch.randperm(n)
        for s in range(0, n, 512):
            idx = order[s:s + 512]
            loss = ((net(xt[idx]) - yt[idx]) ** 2).mean(-1)
            loss = (loss * wt[idx]).sum() / wt[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output", type=Path, default=MODEL)
    args = parser.parse_args()
    files = sorted(glob.glob(str(ROOT / "data" / "imitation" / "*.npz")))
    train, held = files[:-args.holdout], files[-args.holdout:]
    Xt, Yt, Wt, _ = load(train)
    Xh, Yh, Wh, rh = load(held)
    net = fit(Xt, Yt, Wt, args.epochs)
    with torch.no_grad():
        pred = net(torch.tensor(Xh))
        err = (pred - torch.tensor(Yh)).abs().numpy()
    print(f"train {len(Xt)} frames / held {len(Xh)}; mean |err| x={err[:,0].mean():.3f} y={err[:,1].mean():.3f} turn={err[:,2].mean():.3f}")
    for f in sorted(set(rh)):
        m = rh == f
        print(f"  {Path(f).stem}: x={err[m,0].mean():.3f} y={err[m,1].mean():.3f} turn={err[m,2].mean():.3f} (n={m.sum()})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
