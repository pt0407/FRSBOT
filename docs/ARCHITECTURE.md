# ImAI architecture

One bot, two backends. The bot sees a `GameState` and returns `Inputs`; the offline sim and (later) the live
browser game both speak that interface, so nothing in `bot/` knows which one it is driving.

```
            GameState                     Inputs
  backend ─────────────►  bot.Bot.act()  ─────────► backend
  (sim.Env | live.Bridge)   planner → driver        (sim.Env | live.Bridge)
```

## Layout

```
frsbot/
  core/        shared types, no game logic
    state.py     GameState, RobotState, FuelState(arrays), HubState, Phase
    inputs.py    Inputs (mirrors the game's input packet), encode()/decode() to the 10-byte wire form
    specs.py     RobotSpec loaded from data/robot_feel.json (feel + bumper size + hopper capacity)
    field.py     field constants: hub centres, alliance walls, match timeline, hub activity schedule
  replay/      reading the game's recordings
    rbrp.py      RBRP container + robot state
    fuel.py      FUEL section codec
    convert.py   Replay -> sequence of GameState (so replays and sim share one type)
  scrape/      network fetchers (harvest.py, events.py, fetch_board.mjs)
  sim/         offline backend
    drive.py     robot drive model (was sim.py)
    push.py      FUEL push/roll model fitted to replays (writes data/push_model.json)
    intake.py    intake mouth + shooting model fitted to replays (writes data/shoot_model.json)
    env.py       Env.reset(spec, seed) -> GameState; Env.step(Inputs) -> GameState; scoring + hub schedule
  bot/
    planner.py   picks a Goal (Collect(point) | Score(point, aim) | Stage(point)) from state
    driver.py    Goal + state -> Inputs (pursuit, intake-first heading, aim and hold score)
    imai.py      Bot: replans every 0.5 s or on goal completion, calls driver every tick
  baseline/
    route_follow.py  old pro-route waypoint follower (was controller.py), kept as a benchmark
  live/        (later) Playwright bridge: reads state from the page, sends Inputs
  tools/
    build_viewer.py
    evaluate.py  run Bot and baseline in Env for N seeds per robot, print score table
tests/         pytest: codec round-trips, Inputs encode/decode, Env determinism, planner rules
```

Dependency direction: `core` <- `replay`, `sim`, `bot`, `live`; `bot` never imports `sim`, `replay` or `live`.

## Real-engine offline backend

`frsbot.sim.wasm_env.WasmEnv` runs the game's actual Rust/WASM match physics in Node, not the approximate
drive/push models. Download the public worker chunks and WASM once with `python3 -m frsbot.sim.engine.fetch`
(requires curl). Asset URLs are discovered from the live page and worker factory; assets and their manifest
live in gitignored `data/engine/`. No game code is committed. Changed bundling/constructor signatures fail
explicitly rather than silently falling back to approximate physics. Node is found on PATH or under `~/.nvm`.

```python
from frsbot.core.inputs import Inputs
from frsbot.sim.wasm_env import WasmEnv

with WasmEnv() as env:
    state = env.reset(2910, alliance="blue", start={"fx": 0, "fy": 0, "headingDeg": 0}, seed=1)
    state = env.step(Inputs(y=1, field_relative=True), ticks=60)
    full_view = env.view  # score breakdowns, gates, FUEL arrays, hubs, raw robot pose/quat/vel
```

`WasmEnv` talks to `frsbot/sim/engine/vec_server.mjs` over a length-prefixed binary protocol (`frsbot/sim/ipc.py`;
fixed f64 header and robot rows plus raw f32 FUEL positions and u8 `where`). Multi-robot matches (1-6 robots):

```python
from frsbot.sim.policies import GoalImAIPolicy, ReplayPolicy
from frsbot.sim.vec_env import RobotSlot, VecWasmEnv

state = env.reset(robots=[RobotSlot(254, "blue"), RobotSlot(2910, "red", policy=GoalImAIPolicy(spec_2910))], ego=0, seed=1)
result = env.step({0: Inputs(y=1)}, ticks=6)  # StepResult: views {i: GameState}, scores, breakdowns, done
with VecWasmEnv(n_workers=8, matches_per_worker=4) as vec:  # one Node worker per core, k matches each
    vec.reset([dict(robots=[...], seed=s) for s in range(32)])
    results = vec.step([{0: action} for action in batched_actions], ticks=6)
```

An action is `Inputs`, a planner `Goal` (the JS port of the driver, `frsbot/sim/engine/driver.mjs`, drives it
every tick inside the engine process) or `None` (idle). Robots without an action use their `RobotSlot.policy`:
`ImAIPolicy` (Python reference), `GoalImAIPolicy` (Python planner + JS driver), `ReplayPolicy` (a telemetry
recording) or idle. Each worker runs in its own Python process so slot policies run next to their engine.
Every robot's view is the full shared field (the live client hides nothing either); only `me` differs.
`WasmEnv(latency=(lo, hi))` / `VecWasmEnv(latency=...)` draw an observation delay of U(lo, hi) seconds and an
input hold of 1-3 ticks per episode, seeded by the match seed (docs/LIVE.md §2). The delay applies to the whole
observation (slot policies see it too). `python3 -m frsbot.tools.driver_parity` checks the JS driver against
`frsbot/bot/driver.py`; keep them in sync. `python3 -m frsbot.replay.cache` decodes `data/raw/*.rbrp` on every
core into memory-mappable `data/cache/*.npz` (`cache.load`, `cache.states`).

Reset accepts `auto="none"`, `event="event"`, `hp=True`, `unload=True`, `freeplay=False`, and `countdown=False`.
Start uses the game's field fx/fy coordinates (the game may clamp the requested pose); observations use replay
world x/z meters, velocities in m/s, and `a = pi - 2*atan2(quat_y, quat_w)`. Time is seconds from auto start,
including negative countdown time. Freeplay preserves the engine's phase and has no hub toggles.
Every step advances exactly `ticks` 1/60-second ticks, holding the input constant, and returns the last state.
`RobotState.capacity` is the engine view's `hopperCapacity`; it is not the UI's hopper-plus-feed capacity.

`JsonWasmEnv` keeps the original one-robot JSON backend: `frsbot/sim/engine/server.mjs [data/engine]` accepts one JSON object per line: `new` with a `config` object,
`start` (optional `countdown`), `step` (`inputs` or `goals` keyed by robot id, `n` ticks), `state`, and `close`.
`new` may carry `drivers` (`{id: {spec, profile}}`) so `step` goals run the JS driver every tick.
Each response is `{ok:true, view, engine, build}` (close omits view), or `{ok:false, error}`. Typed arrays are
serialized as ordinary flat JSON arrays. Diagnostics go to stderr; after fetching, simulation makes no network calls.
Use a context manager or `close()` to release the child process.

Checks: `python3 -m pytest -q tests/test_engine_fetch.py tests/test_wasm_env.py tests/test_vec_env.py
tests/test_driver_parity.py tests/test_replay_cache.py` (engine tests skip without downloaded assets);
traces/throughput: `python3 -m frsbot.tools.benchmark_engine [--step all --workers 8]`.

## Core types

- `Inputs`: `x, y, turn` in [-1, 1] and booleans `intake, score, outtake, field_relative, climb, block, stow, hood`.
  Wire form from the bundle: u8 type=1, u8 button bits (intake 1, score 2, outtake 4, fieldRelative 8, climb 16,
  block 32, stow 64, hood 128), u16 seq, i16 x, i16 y, i16 turn.
- `RobotState`: id, team, alliance, x, z, heading a (forward = (cos a, sin a), a = pi - quat yaw), vx, vz, hopper, capacity.
- `FuelState`: `pos (N,3) float32`, `where (N,) uint8`. Where codes inferred from replays (not from game source):
  0 resting on field, 1 moving/in flight, 2 in a robot, 3 inside a hub, 4 staged at a human player, 6 being fed in.
- `HubState`: per alliance, `active` and seconds until it next toggles.
- `GameState`: t (s since match start), phase, hubs, robots, fuel, `me` (index of the robot we drive).

## Match model (core/field.py)

Measured from replay `metas` (45 runs, 0 mismatches), t from auto start (some recordings have a 3 s countdown first):
auto 0-20 (both active), gap 20-23 (both inactive), transition 23-33 (both active), shifts 33-58 / 58-83 / 83-108 /
108-133 (alternate; `firstInactive` alliance, the one that scored more auto FUEL, is inactive in shifts 1 and 3),
end game 133-163 (both active), settle 163-166 (inactive). FUEL scores 1 only into an active hub.

## Planner (the decision part)

Replans every 0.5 s. Inputs it uses: hopper h, capacity C, own hub active/time-to-toggle, ball density map,
time left. Definitions:
- `T_score(p)`: route travel time plus measured score-button startup plus h / measured drain rate.
  `data/bot_profiles.json` holds per-robot controller calibrations; unknown robots fall back to nominal fire rate.
- Ball clusters: where==0 and height < 0.2 m, binned to 0.5 m cells, summed over a kernel the width of the intake;
  value of a cell = balls / (drive time + 0.5 s).

Rules, first match wins:
1. Hub active and the active window left <= `T_score(here)` + 1 s and h > 0: Score now.
2. Hub active and (h >= 0.85 C, or no cluster worth >= 2 balls/s within reach): Score.
3. Hub inactive and h >= 0.95 C: Stage at the shooting spot, arriving when the hub turns active; start Score then.
4. Hub inactive and time to activation < `T_score(here)`: Stage.
5. Otherwise Collect at the best cluster. While the hub is active, discount clusters by their distance from the
   shooting spot; while it is inactive, don't.
6. End game: return with `T_score` + 3 s left, reserving time for recovery. Do not start collection without
   time to return and unload. In the last 3 s, score in place if already in a valid own-zone arc, otherwise
   finish returning. Empty robots stop instead of collecting FUEL they cannot score.

## Driver

- Collect(point): pursue the point with the intake (front) leading; slow to 60% within 1 m; intake held on.
- Score(point, aim): drive to the point, turn to the aim, hold `score` once range/heading/speed gates are met
  (thresholds from data/shoot_model.json).
- Stage(point): drive there and face the hub, intake on for any balls on the way.
- Commands go through the inverse deadzone so the game sees what we meant.
- Score progress: after 1 s continuously open without a hopper drop, or 2 s stalled overall at the spot,
  re-aim, try another arc point, then wiggle. After these fail with <=5 FUEL, collect again for at least 2 s
  unless the final deadline prevents another trip. Release score briefly to gain manual heading authority;
  the real 2910 engine ignores manual turn while score is held.
- 2910 uses empirically visible +/-35 degree rear-side arc points. Independently aimed fast shooters can
  hold score during collection in a valid own-zone range at safe speed; the validated policy enables 4414.
  Collection still steers the chassis toward FUEL while its turret aims independently.

## Models fitted from replays (all held-out validated, numbers in the JSON next to the params)

- push.py: free roll decay + rectangle push-out using the robot's bumper length/width.
- intake.py: mouth rectangle in robot frame from where 0/1 -> 2 transitions; shot rate (balls/s), and the
  distance / heading-error / speed envelope in which hopper count drops while hub-bound balls appear.

## Evaluation

Implemented backend-independent entry point: `ImAI(spec).act(GameState) -> Inputs`. Optional robot `gate`
flags (`True` means blocked, except `open`) and `mode` are filled by WASM and left `None` by replays.
An HP-disabled scoring experiment on both alliances verifies blue scores into x≈−3.645 and red into x≈+3.645.
Native brains pass from outside the own-alliance zone; valid score spots are constrained to that zone and the
field boundary as well as the specified range circle. The shooter faces opposite the intake. Barrier routes
use the bump lanes instead of trenches, which are not traversable by every robot in every orientation.

`python3 -m frsbot.tools.evaluate` runs the real engine for teams 254/2910/1114/4414/60026, both alliances,
seeds 1/2/3, no auto, event rules and default HP throws/unloads. Default control is 20 Hz (3 physics ticks),
with `--ticks 1` available for 60 Hz; `--jobs` controls parallel matches. Per-run viewer-compatible traces,
goal times, gate diagnostics and leaderboard comparisons are in gitignored `data/sim/imai/`.
`python3 -m frsbot.tools.build_viewer` includes them. Gate durations and rate-based blocked fire opportunities
overlap across reasons and are not actual lost-shot counts. Scores include human-player contributions.
Leaderboard comparisons exclude driver 404. Native auto is always `none`: ImAI drives from t=0, including
the auto phase, just as in teleop. No climbing is attempted. Leaderboard strategies, seeds and start poses differ.
Use `--no-moving-score` for the collection-shooting ablation. Evaluation also records time in Score with <=5
FUEL, FUEL carried at 163 s, robot-attributed FUEL, and collection-scoring command time. See `IMAI_TUNING.md`.

Longer-term evaluation target: compare Bot and baseline on every robot spec for 5 seeds, reporting mean score,
FUEL scored, FUEL scored while inactive (wasted), and hopper overflow. A change to the bot should not lower mean score.
