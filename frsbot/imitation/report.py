"""Report open-loop validation and training example counts for recovered runs."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from frsbot.imitation.contacts import CAUSES
from frsbot.imitation.recover import RESIDUALS
from frsbot.scrape.harvest import DATA


def summarize(paths):
    runs = defaultdict(list)
    for path in paths:
        with np.load(path, allow_pickle=False) as run:
            segment = run["segment"]
            end = np.r_[segment[:-1] != segment[1:], True]
            begin = np.r_[True, end[:-1]]
            endpoints = run["residual"][end]
            residuals = endpoints.reshape(-1, len(RESIDUALS))
            flags = run["contacts"] if "contacts" in run else None
            buttons = (run["inputs"][:, :, 1, None] & np.array([1, 2])) != 0
            runs[str(run["source"])].append(dict(
                segments=int(np.count_nonzero(end)), examples=run["goal_kind"].size,
                labels=Counter(run["goal_kind"][begin].ravel()), residuals=residuals,
                initial_hopper=(run["initial_hopper"] == run["hopper"][begin]).ravel(),
                flagged=int(run["flagged"].sum()),
                contact_free=None if flags is None else endpoints[~flags.any(axis=1), :, 0].ravel(),
                contacts=None if flags is None else flags.ravel(),
                button_counts=buttons.sum(axis=(0, 1)),
                button_ticks=(buttons * run["ticks"][:, None, None]).sum(axis=(0, 1)),
                agreement=run["button_agreement"] if "button_agreement" in run else None,
            ))
    return {name: _summary(group) for name, group in
            [("all", [r for group in runs.values() for r in group]), *sorted(runs.items())] if group}


def _summary(group):
    residuals = np.concatenate([r["residuals"] for r in group])
    diagnostics = [r for r in group if r["contacts"] is not None]
    contact_free = np.concatenate([r["contact_free"] for r in diagnostics]) if diagnostics else []
    causes = {}
    if diagnostics:
        poses = np.concatenate([r["residuals"][:, 0] for r in diagnostics])
        flags = np.concatenate([r["contacts"] for r in diagnostics])
        for bit, label in CAUSES.items():
            mask = (flags & bit != 0) if bit else (flags == 0)
            causes[label] = (int(mask.sum()), int((poses[mask] >= .1).sum()),
                             float(np.percentile(poses[mask], 90) * 100) if mask.any() else None)
    comparisons = [r["agreement"] for r in group if r["agreement"] is not None]
    return dict(
        replays=len(group), segments=sum(r["segments"] for r in group),
        examples=sum(r["examples"] for r in group), flagged=sum(r["flagged"] for r in group),
        labels=sum((r["labels"] for r in group), Counter()),
        median_cm=float(np.median(residuals[:, 0]) * 100),
        p90_cm=float(np.percentile(residuals[:, 0], 90) * 100),
        contact_free_count=len(contact_free),
        contact_free_pose=None if not len(contact_free) else np.percentile(contact_free, [50, 90]) * 100,
        causes=causes,
        under_10cm=float(np.mean(residuals[:, 0] < .1)),
        initial_hopper=float(np.mean(np.concatenate([r["initial_hopper"] for r in group]))),
        hopper=float(np.mean(residuals[:, 3] == 0)),
        fuel_where=float(residuals[:, 4].mean()), fuel=float(residuals[:, 5].mean()),
        button_counts=np.sum([r["button_counts"] for r in group], axis=0),
        button_ticks=np.sum([r["button_ticks"] for r in group], axis=0),
        agreement=np.sum(comparisons, axis=0) if comparisons else None,
    )


def markdown(summary):
    lines = [
        "| Source | Replays | Segments | Robot/control examples | Pose median / p90 | Contact-excluded median / p90 (endpoints) | Pose <10 cm | Hopper exact (start / end) | FUEL where / where + position | Collect / Score / Stage | Flagged examples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, row in summary.items():
        labels = " / ".join(str(row["labels"][g]) for g in ("Collect", "Score", "Stage"))
        contact_free = "—"
        if row["contact_free_pose"] is not None:
            median, p90 = row["contact_free_pose"]
            contact_free = f"{median:.2f} / {p90:.2f} cm ({row['contact_free_count']:,})"
        lines.append(
            f"| {kind} | {row['replays']:,} | {row['segments']:,} | {row['examples']:,} | "
            f"{row['median_cm']:.2f} / {row['p90_cm']:.2f} cm | {contact_free} | {row['under_10cm']:.1%} | "
            f"{row['initial_hopper']:.1%} / {row['hopper']:.1%} | {row['fuel_where']:.1%} / {row['fuel']:.1%} | "
            f"{labels} | {row['flagged']:,} |"
        )
    lines += ["", "Pose is per-robot planar endpoint distance after an uninterrupted ~2 s rollout. "
              "FUEL rates compare indexed balls at the endpoint; position match is within 10 cm and requires equal "
              "where code. Labels are counted once per robot per segment."]
    lines += ["", "| Source | Button | Positive examples | Positive robot-ticks | Agreement vs legacy | Both on / transition only / legacy only / both off |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for kind, row in summary.items():
        for i, button in enumerate(("Intake", "Score")):
            comparison = row["agreement"]
            counts, accuracy = "—", "—"
            if comparison is not None and comparison[i].sum():
                tp, fp, fn, tn = comparison[i]
                counts = f"{tp:,} / {fp:,} / {fn:,} / {tn:,}"
                accuracy = f"{(tp + tn) / comparison[i].sum():.1%}"
            lines.append(f"| {kind} | {button} | {row['button_counts'][i]:,} | {row['button_ticks'][i]:,} | {accuracy} | {counts} |")
    if any(row["causes"] for row in summary.values()):
        lines += ["", "| Source | Replay proxy | Robot endpoints | Error ≥10 cm | Pose p90 |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for kind, row in summary.items():
            for cause, (count, high_error, p90) in row["causes"].items():
                tail = "—" if p90 is None else f"{p90:.2f} cm"
                lines.append(f"| {kind} | {cause} | {count:,} | {high_error:,} | {tail} |")
        lines += ["", "Contact categories can overlap. Contact-excluded metrics drop a whole segment "
                  "if any robot has a proxy flag. High error with no detected contact is a search-failure "
                  "candidate, not proof: hidden engine state and missed contacts can also produce it."]
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
