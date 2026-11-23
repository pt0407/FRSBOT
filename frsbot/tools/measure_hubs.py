"""Measure hub openings from hub-contained FUEL near the rim in cached replays."""
import gc

import numpy as np

from frsbot.replay.rbrp import POS_SCALE, parse
from frsbot.tools.validate_schedule import RAW


def main():
    paths = sorted(RAW.glob("*.rbrp"))
    if not paths:
        raise FileNotFoundError(f"no cached replays in {RAW}")
    points = {"blue": [], "red": []}
    for path in paths:
        rep = parse(path.read_bytes(), fuel=True)
        for pos, where in rep.fuel:
            near_rim = pos[(where == 3) & (pos[:, 1] * POS_SCALE >= 1.7)] * POS_SCALE
            points["blue"].append(near_rim[near_rim[:, 0] < 0])
            points["red"].append(near_rim[near_rim[:, 0] > 0])
        del rep
        gc.collect()
    for alliance, chunks in points.items():
        p = np.concatenate(chunks)
        if not len(p):
            print(f"{alliance}: no hub-contained FUEL near rim")
            continue
        centre = (p.min(axis=0) + p.max(axis=0)) / 2
        print(f"{alliance}: samples={len(p)} x={centre[0]:.6f} z={centre[2]:.6f} rim_y={p[:, 1].max():.6f}")


if __name__ == "__main__":
    main()
