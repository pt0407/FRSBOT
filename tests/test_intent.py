import json
from dataclasses import replace

import numpy as np
import pytest

from frsbot.analysis.matches import INTENT_LABELS
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState
from frsbot.strategy.intent import FEATURES, HOPPER_FEATURES, IntentModel, IntentTracker, defend_target
from frsbot.strategy.train_intent import export, fit, train


def state(t):
    return GameState(t, Phase.AUTO, {a: HubState(True, 20 - t) for a in ("blue", "red")},
                     [RobotState(1, 254, "blue", 5, 0, 0, 1, 0, int(t * 2), 72),
                      RobotState(2, 254, "red", 6, 0, 0, 0, 0, 20, 72)],
                     FuelState(np.array([[5, .1, 1], [5, .1, 1]], np.float32), np.zeros(2, np.uint8)), 0)


def test_tracker_causal_shape_mask_and_pressure():
    tracker = IntentTracker()
    with pytest.raises(RuntimeError):
        tracker.features(0)
    for k in range(31):
        assert tracker.update(state(k / 10))
    x = tracker.features(0)
    assert x.shape == (len(FEATURES),) and np.isfinite(x).all()
    assert x[FEATURES.index("hopper_delta_1s")] == 2
    assert x[FEATURES.index("hopper_delta_3s")] == 6
    assert x[FEATURES.index("pressure_fraction_3s")] == 1
    assert x[FEATURES.index("in_opponent_half")] == 1
    assert defend_target(tracker.history[-1], 0) == 2
    masked = tracker.features(0, mask_hopper=True)
    assert not masked[list(HOPPER_FEATURES)].any()
    assert masked[FEATURES.index("hopper_visible")] == 0
    np.testing.assert_array_equal(x, tracker.features(0))
    assert not tracker.update(state(3.01))
    np.testing.assert_array_equal(x, tracker.features(0))
    with pytest.raises(ValueError, match="increasing"):
        tracker.update(state(3.01))
    tracker.reset()
    tracker.update(state(0))
    changed = state(.1)
    changed.robots[0].id = 99
    with pytest.raises(ValueError, match="roster"):
        tracker.update(changed)


@pytest.mark.parametrize("classes", [2, 5])
def test_numpy_matches_sklearn_saved_round_trip(tmp_path, classes):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, len(FEATURES)))
    y = np.arange(100) % classes
    scaler, estimator = fit(x, y)
    params = export(scaler, estimator)
    output = {"features": list(FEATURES), "labels": list(INTENT_LABELS),
              "models": {"visible": params, "masked": params}}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(output))
    model = IntentModel.load(path)
    actual = model.probabilities(x)
    np.testing.assert_allclose(actual[:, estimator.classes_], estimator.predict_proba(scaler.transform(x)), atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=1), 1)
    assert np.all(actual[:, classes:] == 0)
    with pytest.raises(ValueError):
        model.probabilities(np.zeros(2))
    with pytest.raises(ValueError):
        IntentModel({**output, "features": []})


def test_training_group_separation_and_both_visibility_models():
    pytest.importorskip("sklearn")
    x = np.random.default_rng(3).normal(size=(90, len(FEATURES)))
    y = np.arange(90) % 5
    groups = np.repeat(["match-a", "match-b", "match-c"], 30)
    result = train(x, y, groups, y, [{"robots": 4}] * 3)
    assert result["split"]["cv_folds"] == 2
    assert not set(result["split"]["development_groups"]) & set(result["split"]["heldout_groups"])
    for name in ("visible", "masked"):
        for fold in result["metrics"][name]["folds"]:
            assert not set(fold["train_groups"]) & set(fold["test_groups"])
        assert result["metrics"][name]["heldout"]["samples"] == 30
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match=">=3"):
        train(x, y, np.repeat("one-match", 90), y, [])
