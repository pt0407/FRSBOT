"""Offline merge smoke: pytest -q -s tests/test_integration_smoke.py (requires cached engine/replay)."""
import json
import multiprocessing as mp
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.core.state import GameState, Phase
from frsbot.imitation.dataset import examples
from frsbot.imitation.recover import RESIDUALS, recover_run
from frsbot.live import page_driver
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.sim.policies import ImAIPolicy
from frsbot.sim.vec_env import RobotSlot, VecWasmEnv
from frsbot.strategy.bot import StrategicImAI
from frsbot.tools.evaluate_strategy import BotPolicy

ROOT = Path(__file__).resolve().parents[1]
SPECS = {s.team: s for s in load_specs()}
engine = pytest.mark.skipif(not (ENGINE_DIR / "manifest.json").is_file(), reason="requires cached real engine assets")


@engine
def test_full_2v2_strategy_and_rule_bots(capsys):
    started = time.monotonic()
    strategy = StrategicImAI(SPECS[254])
    robots = [RobotSlot(254, "blue", policy=BotPolicy(strategy)),
              RobotSlot(1114, "blue", policy=ImAIPolicy(SPECS[1114])),
              RobotSlot(2910, "red", policy=ImAIPolicy(SPECS[2910])),
              RobotSlot(4414, "red", policy=ImAIPolicy(SPECS[4414]))]
    with VecWasmEnv(n_workers=1, matches_per_worker=1) as env:
        env.reset([dict(robots=robots, seed=1, auto="none", event="event", hp=True, unload=True)])
        result, = env.rollout(166, ticks=3)
    assert result.done and result.t == 166
    assert len(result.views) == 4 and all(result.scores[a] > 0 for a in ("blue", "red"))
    assert strategy.layer.tracker.history
    for i, state in result.views.items():
        assert state.me == i and len(state.robots) == 4 and state.phase == Phase.DONE
        assert state.fuel.pos.shape == (504, 3) and np.isfinite(state.fuel.pos).all()
    with capsys.disabled():
        print(f"SMOKE 2v2 t={result.t} blue={result.scores['blue']} red={result.scores['red']} "
              f"strategy=1 rule_bots=3 wall_s={time.monotonic()-started:.2f}", flush=True)


@engine
def test_solo_254_seed_through_evaluate(tmp_path, capsys):
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-m", "frsbot.tools.evaluate", "--teams", "254", "--seeds", "1",
                             "--alliances", "blue", "--jobs", "1", "--output", str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    run, = summary["runs"]
    assert (run["team"], run["alliance"], run["seed"], run["auto"]) == (254, "blue", 1, "none")
    # Broad regression bound around the historical ~584 mean, not a claim of paired seed parity.
    assert 400 <= run["score"] <= 850
    trace = json.loads((tmp_path / "imai-254-blue-1.json").read_text())
    assert trace["durationMs"] == 166000 and trace["samples"][-1]["phase"] == "done"
    assert trace["evaluation"] == run and run["fuel"] > 0
    with capsys.disabled():
        print(f"SMOKE solo team=254 alliance=blue seed=1 score={run['score']} fuel={run['fuel']} "
              f"reference_mean=584 wall_s={time.monotonic()-started:.2f}", flush=True)


@engine
def test_cached_replay_recovery_dataset(tmp_path, capsys):
    replays = sorted((ROOT / "data/raw").glob("*.rbrp"))
    if not replays:
        pytest.skip("requires a cached shared solo replay")
    started = time.monotonic()
    output = tmp_path / "recovered.npz"
    pool = mp.get_context("spawn").Pool(1)
    try:
        recover_run(pool, replays[0], "solo", output, control_ticks=6, limit=1,
                    options=dict(resetVelocity="estimate"))
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    assert output.is_file(), "cached replay yielded no accepted recovery segment"
    with np.load(output, allow_pickle=False) as run:
        frames, robots = run["hopper"].shape
        assert frames >= 3 and robots == 1
        assert run["pose"].shape == (frames, robots, 5)
        assert run["axes"].shape == (frames, robots, 3)
        assert run["inputs"].shape == (frames, robots, 10)
        assert run["residual"].shape == (frames, robots, len(RESIDUALS))
        assert run["fuel_pos"].shape == (frames, 504, 3)
        assert run["fuel_where"].shape == (frames, 504)
        assert run["goal_kind"].shape == run["weight"].shape == (frames, robots)
        assert run["goal_point"].shape == (frames, robots, 2)
        assert set(run["goal_kind"].ravel()) <= {"Collect", "Score", "Stage"}
        assert np.isfinite(run["axes"]).all() and (np.abs(run["axes"]) <= 1).all()
        assert np.isfinite(run["residual"]).all() and (run["residual"] >= 0).all()
        weights = run["weight"].copy()
        assert np.isfinite(weights).all() and ((0 <= weights) & (weights <= 1)).all()
        np.testing.assert_allclose(weights, np.exp(-run["residual"][:, :, 0] / .1))
        np.testing.assert_array_equal(run["flagged"], run["residual"][:, :, 0] > .1)
        labels = run["goal_kind"].copy()
    loaded = list(islice(examples(output), 3))
    assert len(loaded) == 3
    for i, (state, inputs, residual, goal) in enumerate(loaded):
        assert isinstance(state, GameState) and isinstance(inputs, Inputs)
        assert state.fuel.pos.shape == (504, 3) and state.fuel.where.shape == (504,)
        assert goal.kind == labels[i, state.me] and len(goal.point) == 2
        assert set(residual) == set(RESIDUALS)
    with capsys.disabled():
        print(f"SMOKE recovery replay={replays[0].name} examples={frames * robots} loaded=3 "
              f"pose=({frames},{robots},5) fuel=({frames},504,3) labels={sorted(set(labels.ravel()))} "
              f"weight={weights.min():.4f}..{weights.max():.4f} wall_s={time.monotonic()-started:.2f}", flush=True)


def test_live_offline_units_and_canonical_driver(capsys):
    started = time.monotonic()
    canonical = ROOT / "frsbot/sim/engine/driver.mjs"
    assert page_driver.SOURCE.samefile(canonical)
    definition = canonical.read_text().split("// CLI used by the parity test:")[0]
    definition = definition[definition.index("export class Driver"):].replace("export ", "")
    assert definition in page_driver.browser_source()
    assert "class Driver" not in (ROOT / "frsbot/live/page_driver.js").read_text()
    tests = ["tests/test_live.py", "tests/test_live_latency.py", "tests/test_live_page.py", "tests/test_live_predict.py"]
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert " skipped" not in result.stdout
    with capsys.disabled():
        print(f"SMOKE live_offline {result.stdout.strip().splitlines()[-1]} "
              f"canonical_driver=shared browser=none wall_s={time.monotonic()-started:.2f}", flush=True)
