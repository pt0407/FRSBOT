"""Bundle decoded replay telemetry into viewer/replays.js so the viewer works from file:// (no server needed).

Usage: python -m frsbot.tools.build_viewer
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEL = ROOT / "data" / "telemetry"
OUT = ROOT / "viewer" / "replays.js"


def yaw(q):
    # robot quats are [x, y, z, w]; the robot sits flat, so yaw is a rotation about y
    return 2 * math.atan2(q[1], q[3])


def compact(path: Path):
    d = json.loads(path.read_text())
    r0 = d["robots"][0]
    lb = d.get("leaderboard", {})
    score = (d.get("score") or {}).get(r0["alliance"])
    if isinstance(score, dict):
        score = score["total"]
    samples = []
    for s in d["samples"]:
        p = s["robots"][0]
        samples.append([s["tick"], p["pos"][0], p["pos"][2], round(yaw(p["quat"]), 3), p["vel"][0], p["vel"][2], p.get("hopperFuel")])
    return {
        "id": d["id"], "name": r0["name"], "team": r0["team"], "alliance": r0["alliance"],
        "kind": d["kind"], "score": score,
        "durationMs": d["durationMs"], "board": lb.get("points"), "samples": samples,
    }


def main():
    sim = ROOT / "data" / "sim"
    runs = [compact(p) for p in sorted(TEL.glob("*.json")) + sorted(sim.glob("*.json"))
            + sorted((sim / "imai").glob("imai-*.json"))]
    runs.sort(key=lambda r: -(r["board"] or 0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.REPLAYS = " + json.dumps(runs, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT} with {len(runs)} runs")


if __name__ == "__main__":
    main()
