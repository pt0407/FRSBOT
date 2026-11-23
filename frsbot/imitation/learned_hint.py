"""Turn the pro-trained event model into a planner hint: score or collect when the model is confident.

Used by ImAI when IMAI_PARAMS sets learned_hint=1; returns None otherwise, so the planner decides.
"""
import math
from pathlib import Path

import torch

from frsbot.bot.planner import Goal
from frsbot.imitation.pro_planner import CLASSES, nn

MODEL = Path(__file__).resolve().parents[2] / "data" / "models" / "pro_planner.pt"


def load(path: Path = MODEL):
    net = nn.Sequential(nn.Linear(9, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, CLASSES + 2))
    net.load_state_dict(torch.load(path)["state"])
    net.eval()
    return net


def make_hint(net, planner, threshold: float):
    def hint(state):
        r = state.robots[state.me]
        cap = max(1, r.capacity)
        x = torch.tensor([[r.x, r.z, math.cos(r.a), math.sin(r.a), r.vx, r.vz, r.hopper / cap, cap / 120, state.t / 166]],
                         dtype=torch.float32)
        with torch.no_grad():
            p = torch.softmax(net(x)[0, :CLASSES], -1).numpy()
        if p[1] > threshold and r.hopper > 0:
            return Goal("Score", planner.score_spot(state))
        if p[2] > threshold and r.hopper < r.capacity:
            clusters = planner.clusters(state)
            if clusters:
                return Goal("Collect", clusters[0][1])
        return None
    return hint
