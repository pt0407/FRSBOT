"""Runtime for the goal-conditioned drive model trained on pro replays.

Loaded lazily; absent torch or model file means the rule driver keeps control.
Feature layout must stay in sync with imitation/train_drive.py."""
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data" / "models" / "drive_policy.pt"
KINDS = ("Collect", "Score", "Stage")
_net = None
_mtime = 0.


def _load():
    global _net, _mtime
    if MODEL.exists():
        try:
            mtime = MODEL.stat().st_mtime
            if _net is None or mtime != _mtime:
                import torch
                from torch import nn
                net = nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))
                net.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
                net.eval()
                _net, _mtime = net, mtime
        except Exception:
            if _net is None:
                _net = False
    return _net or None


def steer(state, spec, point, kind):
    """Human-fitted field-relative sticks toward `point`, or None if unavailable."""
    net = _load()
    if net is None:
        return None
    import torch
    from frsbot.imitation.input_policy import features as base_features
    r = state.robots[state.me]
    bearing = math.atan2(point[1] - r.z, point[0] - r.x) - r.a
    feat = base_features(state, spec) + [math.cos(bearing), math.sin(bearing),
                                         math.hypot(point[0] - r.x, point[1] - r.z) / 8.]
    feat += [float(kind == k) for k in KINDS]
    with torch.no_grad():
        x, y, turn = net(torch.tensor([feat], dtype=torch.float32))[0].tolist()
    return max(-1., min(1., x)), max(-1., min(1., y)), max(-1., min(1., turn))
