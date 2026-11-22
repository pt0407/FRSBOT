"""Exploratory match-grouped intent training; sklearn is offline-only."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from frsbot.analysis.matches import INTENT_LABELS, future_labels, prepare_match, replay_paths
from frsbot.strategy.intent import FEATURES, HOPPER_FEATURES, IntentTracker


def dataset(directory):
    features, labels, groups, persistence = [], [], [], []
    sources = []
    for path in replay_paths(directory):
        states, samples, _, identity = prepare_match(path)
        targets, _ = future_labels(states, samples)
        tracker = IntentTracker()
        times = np.array([s.t for s in states])
        group = hashlib.sha256(identity.encode()).hexdigest()[:16]
        count = 0
        for k, state in enumerate(states):
            accepted = tracker.update(state)
            past = np.searchsorted(times, state.t - 3. + 1e-7, side="right") - 1
            if not accepted or past < 0 or state.t - times[past] > 3.05 or np.any(np.diff(times[past:k + 1]) > .15):
                continue
            for i in range(len(state.robots)):
                if targets[k, i] < 0 or targets[past, i] < 0:
                    continue
                features.append(tracker.features(i))
                labels.append(targets[k, i])
                persistence.append(targets[past, i])
                groups.append(group)
                count += 1
        sources.append({"path": str(path), "group": group, "robots": len(states[0].robots) if states else 0,
                        "samples": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"loaded {path.name}: robots={sources[-1]['robots']} samples={count} group={group}", flush=True)
    if not features:
        raise ValueError("no complete 3 s past/future examples in replay directory")
    return np.array(features), np.array(labels), np.array(groups), np.array(persistence), sources


def metrics(y, prediction):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
    classes = np.arange(len(INTENT_LABELS))
    return {"samples": len(y), "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=classes, average="macro", zero_division=0)),
            "recall": dict(zip(INTENT_LABELS, recall_score(y, prediction, labels=classes, average=None, zero_division=0).tolist())),
            "support": dict(zip(INTENT_LABELS, np.bincount(y, minlength=len(classes)).tolist())),
            "confusion": confusion_matrix(y, prediction, labels=classes).tolist()}


def fit(x, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) < 2:
        raise ValueError("training partition requires at least two observed labels")
    scaler = StandardScaler().fit(x)
    estimator = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=17).fit(scaler.transform(x), y)
    return scaler, estimator


def export(scaler, estimator):
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": estimator.coef_.tolist(), "intercept": estimator.intercept_.tolist(),
            "classes": estimator.classes_.tolist()}


def train(x, y, groups, persistence, sources):
    from sklearn.model_selection import GroupKFold
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("need >=3 distinct matches for development GroupKFold plus whole-match holdout")
    rng = np.random.default_rng(17)
    shuffled = rng.permutation(unique)
    heldout_groups = shuffled[:max(1, math.ceil(len(unique) * .2))]
    heldout = np.isin(groups, heldout_groups)
    development = ~heldout
    folds = min(5, len(np.unique(groups[development])))
    output = {"version": 1, "features": list(FEATURES), "labels": list(INTENT_LABELS),
              "sources": sources, "matches": len(unique), "robots": sum(s["robots"] for s in sources),
              "samples": len(y), "label_counts": dict(zip(INTENT_LABELS, np.bincount(y, minlength=5).tolist())),
              "split": {"seed": 17, "development_groups": sorted(set(groups[development])),
                        "heldout_groups": heldout_groups.tolist(), "cv_folds": folds,
                        "final_fit": "development matches only; heldout never used for fitting or scaling"},
              "notes": ["Exploratory weak labels, overlapping 3 s windows; not human intent ground truth.",
                        "Score/Feed/Collect from upstream native-rate FUEL attribution, future (t,t+3].",
                        "Defend requires 1.5 m center pressure on opponent half for >=50% of future window.",
                        "No feature uses future observations. Masking zeros all robot hopper features, not annotations.",
                        "Ranked identities anonymized: owner 404 cannot be excluded; do not deploy/promote without identity policy.",
                        "Small partial-match corpus; performance estimates are preliminary, not generalization claims."],
              "models": {}, "metrics": {}}
    indices = np.flatnonzero(development)
    for name in ("visible", "masked"):
        values = x.copy()
        if name == "masked":
            values[:, list(HOPPER_FEATURES)] = 0.
            values[:, FEATURES.index("hopper_visible")] = 0.
        cv_true, cv_pred, cv_majority, cv_persistence = [], [], [], []
        fold_results = []
        for a, b in GroupKFold(n_splits=folds).split(indices, y[development], groups[development]):
            a, b = indices[a], indices[b]
            scaler, estimator = fit(values[a], y[a])
            prediction = estimator.predict(scaler.transform(values[b]))
            majority = Counter(y[a]).most_common(1)[0][0]
            cv_true.extend(y[b]); cv_pred.extend(prediction)
            cv_majority.extend([majority] * len(b)); cv_persistence.extend(persistence[b])
            fold_results.append({"train_groups": sorted(set(groups[a])), "test_groups": sorted(set(groups[b])),
                                 **metrics(y[b], prediction)})
        scaler, estimator = fit(values[development], y[development])
        prediction = estimator.predict(scaler.transform(values[heldout]))
        majority = Counter(y[development]).most_common(1)[0][0]
        output["models"][name] = export(scaler, estimator)
        output["metrics"][name] = {
            "cv": metrics(np.array(cv_true), np.array(cv_pred)), "folds": fold_results,
            "cv_majority": metrics(np.array(cv_true), np.array(cv_majority)),
            "cv_persistence": metrics(np.array(cv_true), np.array(cv_persistence)),
            "heldout": metrics(y[heldout], prediction),
            "heldout_majority": metrics(y[heldout], np.full(heldout.sum(), majority)),
            "heldout_persistence": metrics(y[heldout], persistence[heldout])}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/intent_model.json"))
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = train(*dataset(args.directory))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"matches={result['matches']} robots={result['robots']} samples={result['samples']}")
    print("label_counts=" + json.dumps(result["label_counts"]))
    print("split=" + json.dumps(result["split"]))
    print("confusion rows=true columns=predicted order=" + json.dumps(result["labels"]))
    for name, report in result["metrics"].items():
        for partition in ("cv", "heldout", "heldout_majority", "heldout_persistence"):
            print(f"{name} {partition}: " + json.dumps(report[partition], sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
