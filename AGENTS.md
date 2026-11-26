# FRSBOT / ImAI: handoff notes

ImAI is a state-only AI driver for FRS Simulator 2026 REBUILT (https://rebuilt.frsim.com),
targeting solo and 1v1/2v2/3v3. `ImAI(spec).act(GameState) -> Inputs` must not import sim/replay/live.

## Package layout

- `frsbot/core/`: backend-independent state, inputs/wire codec, match schedule and specs for all 20 robots.
- `frsbot/bot/`: planner, driver, ImAI interface and calibration; profiles in `data/bot_profiles.json`.
- `frsbot/sim/wasm_env.py`, `frsbot/sim/engine/`: real-game backend and asset fetcher/Node server.
  `frsbot/sim/drive.py`, `frsbot/sim/push.py` and `frsbot/baseline/route_follow.py` are legacy models/baselines.
- `frsbot/replay/`: RBRP/FUEL decoding, spectator packaging and multi-robot GameState conversion.
- `frsbot/scrape/`: leaderboard/event scraping, replay harvesting, passive spectator capture,
  continuous recorder, GitHub release upload queue and downloader.
- `frsbot/analysis/matches.py`: observational FUEL attribution, roles and defense analysis.
- `frsbot/tools/`: evaluation, benchmarks, calibration/validation and viewer building.
- `data/robot_feel.json`: all 20 robots' extracted drive, intake and shooting specs.
  Solo telemetry is in `data/telemetry/`; generated raw replays, engine assets, multiplayer captures
  and sim traces are gitignored under `data/raw/`, `data/engine/`, `data/raw_mp/` and `data/sim/`.
- `tests/`: Python/Node tests and recorded fixtures. `viewer/`: replay/sim viewer and bundled runs.

## Engine and replay facts

- `WasmEnv` runs the game's real Rust/WASM physics, FUEL, intake, shooting and match rules in a
  local Node process at 60 Hz. Fetch assets once; **never commit downloaded game files**.
  Node is found on PATH or under `~/.nvm/versions/node`; put NVM's Node on PATH for all tests.
- Solo replays are **30 Hz**; the observed ranked spectator stream is **60 Hz**. The 80-tick interval
  is between keyframes, not pose samples. Replays contain state, not controller inputs.
- FUEL positions and where-codes are decoded. Observed meanings: `0` resting on field, `1` moving/in
  flight, `2` in a robot, `3` inside a hub, `4` staged at a human player, `6` being fed in.
- Coordinates are meters in world x/z; forward is `(cos(a), sin(a))`, with `a = pi - quaternion yaw`.
  Replay and engine observations share these conventions. Replay gate/mode observations are `None`.
- Leaderboard inventory: 46 listed shared IDs, 45 usable solo replays across 17 robots; the extra ID
  returns HTTP 404. Spectator captures include a complete real 2v2 (FJ5S). Ranked identities are
  anonymized, so account-level training exclusions cannot be guaranteed.

## Match schedule

Time is normalized to auto start (countdown is negative), as implemented in `frsbot/core/field.py`:
auto 0–20 s; gap 20–23; transition 23–33; shifts 33–58, 58–83, 83–108, 108–133;
endgame 133–163; settle 163–166; done at 166. Both hubs are active in auto, transition and endgame,
inactive in gap/settle. During shifts they alternate: `firstInactive` is off in shifts 1 and 3.
FUEL scores one point in an active hub. ImAI drives from t=0 with native `auto="none"`.

## Docs index

- [ARCHITECTURE](docs/ARCHITECTURE.md): core interfaces, planner/driver and engine backend.
- [TRAINING](docs/TRAINING.md): proposed input recovery, imitation, RL and multiplayer strategy.
- [LIVE](docs/LIVE.md): proposed browser bridge, latency and teammate intent work.
- [PERF](docs/PERF.md): proposed JS driver, binary IPC and vector-environment optimizations.
- [MATCH_ANALYSIS](docs/MATCH_ANALYSIS.md): role/defense methodology and FJ5S findings.
- [MULTIPLAYER_CAPTURE](docs/MULTIPLAYER_CAPTURE.md): passive capture, continuous recording and releases.
- [LEADERBOARD_DATA](docs/LEADERBOARD_DATA.md): supported queries, provenance and harvest counts.
- [IMAI_TUNING](docs/IMAI_TUNING.md): tuning results, calibrations and remaining weaknesses.

## Key commands

Run from the repository root (Node v24.22.2 is available under `~/.nvm/versions/node/`):

```sh
python3 -m frsbot.sim.engine.fetch
python3 -m frsbot.tools.evaluate
python3 -m frsbot.tools.benchmark_engine
python3 -m frsbot.tools.build_viewer
python3 -m frsbot.scrape.record_forever
python3 -m frsbot.scrape.fetch_mp
~/.nvm/versions/node/v24.22.2/bin/node frsbot/scrape/spectate.mjs --minutes 60 --concurrency 2 --decode --output data/raw_mp/batch
~/.nvm/versions/node/v24.22.2/bin/node frsbot/scrape/fetch_board.mjs data/board_queries.json --all
python3 -m frsbot.scrape.harvest data/board.json data/board_queries.json
python3 -m pytest -q
```

Evaluation defaults to 30 full solo matches (five robots, both alliances, three seeds), native auto
disabled, site-default HP throws/unloads and 20 Hz control. Traces/summary go to `data/sim/imai/`.
The continuous recorder uses four passive REBUILT watches, compact files, dedupe and restart backoff.
Export `FRSBOT_GITHUB_TOKEN` for daily `mp-data-YYYYMMDD` release uploads; without it, files stay queued
locally. `fetch_mp` downloads those releases. Only `rankedLive`/`rankedWatch` are sent by spectators.

## Constraints

- Don't use a ranked or online room with the bot without checking the game's rules first.
- Driver name `404` is the owner's account. Its replays are excluded from training data.
- auto plays like teleop; climbing is rare and not a priority

## Current status: live singleplayer score

- Best live singleplayer runs: 626 (team 254) and 608 (team 4414) with `run_imai.command`. Target 1200 is not reached.
- Offline 30-case eval (feat/trip-sequencing @ 1cff8e5): 574.4 mean. Team means:
  254 727, 1114 641, 2910 505.5, 4414 745.5, 60026 260. Focused extras: 2481 ~638, 1678 ~327.
- Engine facts verified this session: `t.turn` is ignored for every robot while
  `score` is pressed — the brain sets `desiredRotation` directly (bit-identical A/B);
  heading/hood/flywheel gate-blocked ticks are mostly cosmetic since `firing` latches
  on the first static aligned tick and persists through translation (254 drops
  observed at 0.9 m/s); `xLock` robots like 254 are pinned while latched, others
  (1678) creep at ~1 m/s with intermittent gate. Pressing score during an INACTIVE
  hub dumps the hopper (shots don't count) — never do it.
- Measured dead ends (kept OFF): all banking variants (−150 to −230: parked Feed,
  seed pile, mow-feed — pile re-collection is rate-capped below field mowing);
  close score spots r≈2.0/2.4 (−90 to −230 vs r≈3.5); latched ejecta-creep
  (−63 to −163: nearest-ball chasing breaks the latch); active_hub_bias 0.8
  (−73 on 254).
- Every tuning knob in `frsbot/bot/tuning.py` defaults to the shipped value. Sweeps on the 6-case
  screens (team 4414 and 2910) and the full protocol did not beat baseline by a meaningful margin:
  stage fullness, arrival margin, intake gate, scoring trigger, cluster bias, and the learned hint.
- Limit found: collection volume and trip sequencing, not stalls or constants. The pro collects about
  1390 FUEL in a match on the same family; the bot collects about 600. Shooter fire rate caps 4414 at 14.
- The learned event model (`frsbot/imitation/pro_planner.py`) predicts the next second well on held-out
  replays (80% event accuracy, 0.62 m pose error), but overriding goals with it lowered scores.

## Learned drive layer and RL fine-tuning (Nov 2026)

- `frsbot/imitation/train_drive.py`: goal-conditioned BC on recovered pro inputs (`data/imitation/solo-*.npz`,
  re-recovered at `--control-ticks 1`, ~136k frames). Features: 14 base + bearing cos/sin + goal distance +
  goal-kind one-hot (20 total). Writes `data/models/drive_policy.pt` (gitignored).
- `frsbot/bot/learned_drive.py`: lazy torch runtime, reloads weights when the file mtime changes.
- `driver.py` hook (gated `learned_drive`, default 0): sticks become `blend*net + (1-blend)*rule` on x/y,
  independent `learned_turn_blend`; rule keeps terminals (`learned_terminal_m`), buttons, wall clamp, unwedge.
- Pure learned sticks collapse closed-loop (open-loop drift); blending is required. Rule servo + human shape.
- `frsbot/imitation/dagger_rollout.py`: labels net-visited states with the rule expert; merged by train_drive.
  DAgger ceiling is the rule driver itself.
- `frsbot/imitation/train_rl.py`: PPO fine-tune in `VecWasmEnv` (~140x aggregate realtime, ~70 iterations/hour
  on 6 cores). Reward = ego score delta per control step; noise on x/y with blend-aware exact credit
  assignment; turn stays rule. Saves to `data/models/drive_policy.pt` every `--save-every` iterations and on
  new best mean. Reproduce evals with `IMAI_PARAMS='{"learned_drive":1,"learned_blend":0.6,"learned_turn_blend":0}'`.

## Human movement law + evolution (2026-11-25)

- Driver collect law is now the human-measured behavior: flat cruise at
  `collect_cruise_frac*vmax` (default 0.68), no distance braking, no cosine
  heading throttle. Extracted from 136k recovered pro frames (`/tmp/extract_human.py`
  pattern): pros sustain 2.6-3.1 m/s through every collect bin, brake only into the
  hub, leave for the hub at hopper fraction ~0.88 (vs our default collect_frac 0.3).
- Robots are confirmed holonomic (swerve): full 4.4 m/s perpendicular to chassis.
- CMA-ES evolution harness: `python -m frsbot.tools.evolve_params` — 18 genes,
  popsize 12, teams 254/4414/1678, crash-proofed (fitness 0 for crashed candidates,
  planner score-spot fallback). Best params checkpoint to `data/rl/evolved_params.json`.
  ~10 min/generation; running overnight in `data/rl/evolve_run2.log`.
- Dead ends confirmed this session (all measured regressions): pure BC closed-loop,
  BC+rule blends (neutral at best), PPO sparse-reward at small scale, transit-mow
  bending, drive-through targets, mow sweep, free_fire, pass-through alone.
- Learned-drive infrastructure exists but is gated off (`learned_drive`): see
  `frsbot/bot/learned_drive.py`, `frsbot/imitation/train_drive.py` — BC ceiling problem
  documented in git history.
