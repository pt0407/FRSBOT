"""Report open-loop validation and training example counts for recovered runs."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from frsbot.imitation.recover import RESIDUALS
from frsbot.scrape.harvest import DATA


def summarize(paths):
    runs = defaultdict(list)
    for path in paths:
        with np.load(path, allow_pickle=False) as run:
            segment = run["segment"]
            end = np.r_[segment[:-1] != segment[1:], True]
            begin = np.r_[True, end[:-1]]
            residuals = run["residual"][end].reshape(-1, len(RESIDUALS))
            runs[str(run["source"])].append(dict(
                segments=int(np.count_nonzero(end)), examples=run["goal_kind"].size,
                labels=Counter(run["goal_kind"][begin].ravel()), residuals=residuals,
                initial_hopper=(run["initial_hopper"] == run["hopper"][begin]).ravel(),
                flagged=int(run["flagged"].sum()),
            ))
    return {name: _summary(group) for name, group in
            [("all", [r for group in runs.values() for r in group]), *sorted(runs.items())] if group}


def _summary(group):
    residuals = np.concatenate([r["residuals"] for r in group])
    return dict(
        replays=len(group), segments=sum(r["segments"] for r in group),
        examples=sum(r["examples"] for r in group), flagged=sum(r["flagged"] for r in group),
        labels=sum((r["labels"] for r in group), Counter()),
        median_cm=float(np.median(residuals[:, 0]) * 100),
        p90_cm=float(np.percentile(residuals[:, 0], 90) * 100),
        under_10cm=float(np.mean(residuals[:, 0] < .1)),
        initial_hopper=float(np.mean(np.concatenate([r["initial_hopper"] for r in group]))),
        hopper=float(np.mean(residuals[:, 3] == 0)),
        fuel_where=float(residuals[:, 4].mean()), fuel=float(residuals[:, 5].mean()),
    )


def markdown(summary):
    lines = [
        "| Source | Replays | Segments | Robot/control examples | Pose median / p90 | Pose <10 cm | Hopper exact (start / end) | FUEL where / where + position | Collect / Score / Stage | Flagged examples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, row in summary.items():
        labels = " / ".join(str(row["labels"][g]) for g in ("Collect", "Score", "Stage"))
        lines.append(
            f"| {kind} | {row['replays']:,} | {row['segments']:,} | {row['examples']:,} | "
            f"{row['median_cm']:.2f} / {row['p90_cm']:.2f} cm | {row['under_10cm']:.1%} | "
            f"{row['initial_hopper']:.1%} / {row['hopper']:.1%} | {row['fuel_where']:.1%} / {row['fuel']:.1%} | "
            f"{labels} | {row['flagged']:,} |"
        )
    lines += ["", "Pose is per-robot planar endpoint distance after an uninterrupted ~2 s rollout. "
              "FUEL rates compare indexed balls at the endpoint; position match is within 10 cm and requires equal "
              "where code. Labels are counted once per robot per segment."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DATA / "imitation")
    args = parser.parse_args()
    summary = summarize(sorted(args.directory.glob("*.npz")))
    if not summary:
        parser.error(f"no recovered runs in {args.directory}")
    print(markdown(summary))


if __name__ == "__main__":
    main()
