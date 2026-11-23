from dataclasses import replace

import numpy as np
import pytest

from frsbot.analysis.matches import (INTENT_LABELS, analyze, future_labels, intent_summary,
                                     load_match, resample)
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState


def sequence(kind="Idle", seconds=7):
    result = []
    for k in range(seconds * 10 + 1):
        t = k / 10
        x = 5. if kind == "Defend" else 0. if kind == "Feed" else -6.
        hopper = int(t * 2) if kind == "Collect" else max(0, 40 - int(t * 2)) if kind in ("Score", "Feed") else 20
        robots = [RobotState(1, 254, "blue", x, 0, 0, 0, 0, hopper, 72),
                  RobotState(2, 254, "red", 5.9, 0, 0, 0, 0, 20, 72)]
        result.append(GameState(t, Phase.AUTO, {a: HubState(True, 20 - t) for a in ("blue", "red")},
                                robots, FuelState(np.empty((0, 3), np.float32), np.empty(0, np.uint8)), 0))
    return result


@pytest.mark.parametrize("kind", INTENT_LABELS)
def test_future_labels(kind):
    states = sequence(kind)
    labels, targets = future_labels(states)
    assert labels[0, 0] == INTENT_LABELS.index(kind)
    assert labels[40, 0] == INTENT_LABELS.index(kind)
    assert labels[41, 0] == -1
    assert targets[0, 0] == (2 if kind == "Defend" else -1)


def test_defense_requires_opponent_half_and_half_window():
    states = sequence("Defend")
    for state in states[:20]:
        state.robots[1].x = -5
    assert future_labels(states)[0][0, 0] == INTENT_LABELS.index("Idle")
    for state in states:
        state.robots[0].x = -5
        state.robots[1].x = -4
    assert future_labels(states)[0][0, 0] == INTENT_LABELS.index("Idle")


def test_active_zone_sign_and_future_window_excludes_current_drop():
    states = sequence("Score")
    for state in states:
        state.hubs["blue"].active = False
    assert future_labels(states)[0][0, 0] == INTENT_LABELS.index("Idle")
    for state in states:
        state.robots[0].alliance = "red"
        state.robots[0].x = 6
    assert future_labels(states)[0][0, 0] == INTENT_LABELS.index("Score")
    states = sequence()
    previous = replace(states[0], t=-.1,
                       robots=[replace(r, hopper=40) for r in states[0].robots])
    assert future_labels([previous, *states])[0][1, 0] == INTENT_LABELS.index("Idle")


def test_upstream_attribution_labels_even_with_constant_hopper():
    states = sequence()
    for state in states:
        where = 2 if state.t < 1 else 1 if state.t < 2 else 3
        point = [-6, .7, 0] if where == 2 else [-5, 1, 0] if where == 1 else [-3.645, 1.8, 0]
        state.fuel = FuelState(np.array([point, point], dtype=np.float32), np.array([where, where], np.uint8))
    result = analyze(states, include_samples=True)
    samples = result.pop("samples")
    assert future_labels(states)[0][0, 0] == INTENT_LABELS.index("Idle")
    assert future_labels(states, samples)[0][0, 0] == INTENT_LABELS.index("Score")
    summary = intent_summary(states, samples, result, "synthetic")
    assert summary["robots"] == 2 and summary["label_counts"]["Score"] > 0
    assert "samples" not in analyze(states)


def test_resampling_empty_gaps_errors_and_fixture():
    assert resample([]) == []
    assert future_labels([])[0].shape == (0, 0)
    states = [replace(sequence()[0], t=k / 60) for k in range(61)]
    assert len(resample(states)) == 11
    with pytest.raises(ValueError):
        resample(states, hz=0)
    with pytest.raises(ValueError):
        resample([states[0], states[0]])
    assert future_labels(sequence()[::2])[0][0, 0] == -1
    from pathlib import Path
    fixture = Path(__file__).parent / "fixtures" / "multiplayer_excerpt.rbrp.gz"
    assert len(load_match(fixture)[0].robots) == 4
