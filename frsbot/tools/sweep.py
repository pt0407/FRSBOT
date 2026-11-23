"""Run ImAI with overridden tuning constants through the solo evaluator and report the mean score.

Usage:
  python -m frsbot.tools.sweep --name collect40 --set '{"collect_frac": 0.4}'            # screening set (5 cases)
  python -m frsbot.tools.sweep --name collect40 --set '{"collect_frac": 0.4}' --full     # protocol set (30 cases)
Results go to data/tune/<name>/ and a one-line summary is printed.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEAMS = [254, 2910, 1114, 4414, 60026]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--set", default="{}", help="JSON overrides for frsbot/bot/tuning.py DEFAULTS")
    parser.add_argument("--full", action="store_true", help="all 30 protocol cases instead of the 5-case screen")
    parser.add_argument("--teams", type=int, nargs="+", default=TEAMS)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    overrides = json.loads(args.set)
    out = ROOT / "data" / "tune" / args.name
    cmd = [sys.executable, "-m", "frsbot.tools.evaluate", "--teams", *map(str, args.teams),
           "--seeds", *(["1", "2", "3"] if args.full else ["1"]), "--alliances", *(["blue", "red"] if args.full else ["blue"]),
           "--jobs", str(args.jobs), "--output", str(out)]
    env = {**os.environ, "IMAI_PARAMS": json.dumps(overrides)}
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    runs = json.loads((out / "summary.json").read_text())["runs"]
    scores = [r["score"] for r in runs]
    per_team = {t: round(statistics.mean(r["score"] for r in runs if r["team"] == t), 1) for t in args.teams}
    print(f"{args.name} {json.dumps(overrides)} n={len(scores)} mean={statistics.mean(scores):.1f} per_team={per_team}")


if __name__ == "__main__":
    main()
