import pytest

from frsbot.core.specs import load_specs
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.tools.driver_parity import compare, record, synthetic

SPECS = {s.team: s for s in load_specs()}


@pytest.mark.parametrize("team", [254, 2910, 60026])
def test_js_driver_matches_python_on_synthetic_states(team):
    for moving in (None, True):
        result = compare(SPECS[team], synthetic(SPECS[team], n=300, seed=team), shoot_while_collecting=moving)
        assert result["max_axis_error"] < 1e-9 and result["button_mismatches"] == 0 and result["flag_mismatches"] == 0


@pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").exists(), reason="requires python3 -m frsbot.sim.engine.fetch")
def test_js_driver_matches_python_on_recorded_engine_states():
    result = compare(SPECS[254], record(SPECS[254], "red", seed=2, seconds=60))
    assert result["states"] > 1000
    assert result["max_axis_error"] < 1e-9 and result["button_mismatches"] == 0 and result["flag_mismatches"] == 0
