"""Train a planner on exact pro telemetry (30 Hz samples), held out by replay.

Labels come from the hopper count over the next second (exact, not inferred):
  2 = intake (hopper gains >= EVENT), 1 = score (hopper loses >= EVENT), 0 = neither.
Targets also include the robot's displacement over the next second.

Usage: python -m frsbot.imitation.pro_planner [--holdout 9] [--epochs 60]
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
CAP = {r["team"]: r["hopperCapacity"] for r in json.loads((ROOT / "data" / "robot_feel.json").read_text())}
HORIZON = 30  # samples = 1 s at 30 Hz
EVENT = 4  # fuel
CLASSES = 3


def yaw(q):
    return math.pi - 2 * math.atan2(q[1], q[3])


def rows(path):
    d = json.loads(Path(path).read_text())
    S = d["samples"]
    robot = d["robots"][0]
    cap = max(1, CAP.get(robot["team"], 72))
    hop = [s["robots"][0]["hopperFuel"] for s in S]
    feats, kinds, moves = [], [], []
    for i in range(len(S) - HORIZON):
        a, b = S[i]["robots"][0], S[i + HORIZON]["robots"][0]
        d_hop = hop[i + HORIZON] - hop[i]
        kinds.append(2 if d_hop >= EVENT else (1 if d_hop <= -EVENT else 0))
        q = a["quat"]
        feats.append([a["pos"][0], a["pos"][2], math.cos(yaw(q)), math.sin(yaw(q)), a["vel"][0], a["vel"][2],
                      hop[i] / cap, cap / 120, S[i]["tick"] / (166 * 60)])
        moves.append([b["pos"][0] - a["pos"][0], b["pos"][2] - a["pos"][2]])
    return np.array(feats, np.float32), np.array(kinds, np.int64), np.array(moves, np.float32)


def load_all(files):
    out = {}
    for f in files:
        X, K, Y = rows(f)
        out[Path(f).stem] = (X, K, Y)
    return out


def fit(X, K, Y, epochs, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, CLASSES + 2))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    xt, kt, yt = torch.tensor(X), torch.tensor(K), torch.tensor(Y)
    n = len(X)
    for _ in range(epochs):
        order = torch.randperm(n)
        for s in range(0, n, 2048):
            idx = order[s:s + 2048]
            o = net(xt[idx])
            loss = nn.functional.cross_entropy(o[:, :CLASSES], kt[idx]) + nn.functional.mse_loss(o[:, CLASSES:], yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--save", type=Path, default=ROOT / "data" / "models" / "pro_planner.pt")
    args = parser.parse_args()
    data = load_all(sorted(glob.glob(str(ROOT / "data" / "telemetry" / "*.json"))))
    names = sorted(data)
    test_names = names[:: max(1, len(names) // args.holdout)][: args.holdout]
    train_names = [n for n in names if n not in test_names]
    X = np.concatenate([data[n][0] for n in train_names]); K = np.concatenate([data[n][1] for n in train_names])
    Y = np.concatenate([data[n][2] for n in train_names])
    print(f"train replays {len(train_names)} ({len(X)} rows), held-out {len(test_names)} ({sum(len(data[n][1]) for n in test_names)} rows)")
    net = fit(X, K, Y, args.epochs)
    accs, base_accs, errs, base_errs = [], [], [], []
    for n in test_names:
        tX, tK, tY = data[n]
        with torch.no_grad():
            o = net(torch.tensor(tX)).numpy()
        accs.append((o[:, :CLASSES].argmax(1) == tK).mean())
        base_accs.append((tK == np.bincount(K, minlength=CLASSES).argmax()).mean())
        errs.append(np.linalg.norm(o[:, CLASSES:] - tY, axis=1).mean())
        base_errs.append(np.linalg.norm(Y.mean(0) - tY, axis=1).mean())
    print(f"held-out event accuracy {np.mean(accs):.3f} vs majority {np.mean(base_accs):.3f}")
    print(f"held-out 1s pose error {np.mean(errs):.3f} m vs mean-move {np.mean(base_errs):.3f} m")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "held_out": test_names}, args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
