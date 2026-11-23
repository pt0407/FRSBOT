"""Learned input policy: bot-visible state + robot spec -> sticks (x, y, turn) and intake/score buttons.

Trained on recovered pro frames (frsbot.imitation.dataset). Held out by replay.
Usage: python -m frsbot.imitation.input_policy [--holdout 4] [--epochs 30]
"""
import argparse
import glob
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.imitation.dataset import examples

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data" / "models" / "input_policy.pt"
SPECS = {s.team: s for s in load_specs()}
N_FEAT = 14


def features(state, spec):
    r = state.robots[state.me]
    hub = state.hubs[r.alliance]
    cap = max(1, r.capacity)
    return [r.x / 8.27, r.z / 4.03, math.cos(r.a), math.sin(r.a), r.vx / 5.1, r.vz / 5.1,
            r.hopper / cap, cap / 120, float(hub.active), min(hub.seconds_to_toggle, 60) / 60, state.t / 166,
            spec.max_speed / 5.1, spec.accel / 10, spec.fire_rate / 32]


def load(files):
    X, Y = [], []
    for f in files:
        for state, inp, _, _ in examples(f):
            spec = SPECS.get(state.robots[state.me].team)
            if spec is None:
                continue
            X.append(features(state, spec))
            Y.append([inp.x, inp.y, inp.turn, float(inp.intake), float(inp.score)])
    return np.array(X, np.float32), np.array(Y, np.float32)


def build():
    return nn.Sequential(nn.Linear(N_FEAT, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 5))


def fit(X, Y, epochs, seed=0):
    torch.manual_seed(seed)
    net = build()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    xt, yt = torch.tensor(X), torch.tensor(Y)
    n = len(X)
    for _ in range(epochs):
        order = torch.randperm(n)
        for s in range(0, n, 512):
            idx = order[s:s + 512]
            o = net(xt[idx])
            loss = nn.functional.mse_loss(torch.tanh(o[:, :3]), yt[idx, :3]) + \
                nn.functional.binary_cross_entropy_with_logits(o[:, 3:], yt[idx, 3:])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def predict(net, X):
    with torch.no_grad():
        o = net(torch.tensor(X))
    sticks = torch.tanh(o[:, :3]).numpy()
    buttons = (torch.sigmoid(o[:, 3:]) > 0.5).numpy()
    return sticks, buttons


def to_inputs(net, feats):
    s, b = predict(net, np.array([feats], np.float32))
    x, y, turn = (float(v) for v in s[0])
    return Inputs(x=x, y=y, turn=turn, intake=bool(b[0, 0]), score=bool(b[0, 1]), field_relative=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    files = sorted(glob.glob(str(ROOT / "data" / "imitation" / "solo-*.npz")))
    test = files[:: max(1, len(files) // args.holdout)][: args.holdout]
    train = [f for f in files if f not in test]
    Xtr, Ytr = load(train)
    print(f"train {len(train)} replays ({len(Xtr)} frames), held-out {len(test)}")
    net = fit(Xtr, Ytr, args.epochs)
    errs, base_errs, acc, base_acc = [], [], [], []
    for f in test:
        Xt, Yt = load([f])
        if not len(Xt):
            continue
        s, b = predict(net, Xt)
        errs.append(np.mean((s - Yt[:, :3]) ** 2))
        base_errs.append(np.mean(Yt[:, :3] ** 2))
        acc.append(np.mean(b == Yt[:, 3:].astype(bool)))
        base_acc.append(np.mean(Yt[:, 3:].astype(bool) == (Yt[:, 3:].mean(0) >= 0.5)))
    print(f"held-out stick MSE {np.mean(errs):.3f} (predict-zero {np.mean(base_errs):.3f})")
    print(f"held-out button accuracy {np.mean(acc):.3f} (majority {np.mean(base_acc):.3f})")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "held_out": test}, MODEL)
    print("saved", MODEL)


if __name__ == "__main__":
    main()
