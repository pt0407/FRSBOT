# FRSBOT / ImAI: handoff notes

Goal: an AI driver ("ImAI") for the FRS Simulator 2026 REBUILT game (https://rebuilt.frsim.com).
It should play singleplayer, 1v1, 2v2, and 3v3 like a strong human, adapt to the robot it drives and to opponents.

## Layout

See `docs/ARCHITECTURE.md` for the design and target layout.

- `frsbot/core/specs.py`: shared robot specs, drive feel, frame/bumper dimensions and hopper capacity.
- `frsbot/core/inputs.py`: shared Inputs and the 10-byte wire codec.
- `frsbot/replay/rbrp.py`: decoder for RBRP replay files (gzip-compressed on the wire). Robot pose, velocity, quaternion, hopper.
- `frsbot/replay/fuel.py`: FUEL section codec.
- `frsbot/scrape/harvest.py`: downloads shared leaderboard replays, decodes to `data/telemetry/*.json`, skips driver name `404`.
- `frsbot/scrape/fetch_board.mjs`: pulls the leaderboard over the websocket (`rebuiltBoard`) into `data/board.json`.
- `frsbot/scrape/events.py`: scrapes completed REBUILT events (matches, scores, breakdowns) into `data/events/*.json`.
- `frsbot/sim/drive.py`: offline drive model ported from the game code. Per-robot feel values, deadzone, acceleration limit, hopper.
- `frsbot/sim/push.py`: fits the FUEL push/roll model from cached replays in `data/raw/`.
- `frsbot/baseline/route_follow.py`: baseline waypoint follower on a pro route. Writes `data/sim/*.json`.
- `frsbot/tools/build_viewer.py`: bundles telemetry and sim runs into `viewer/replays.js`.
- `viewer/index.html`: replay viewer. Serve with `python3 -m http.server 8765` from `viewer/`.
- `data/robot_feel.json`: per-robot drive values (max speed, turn rate, accel, hopper capacity) from the game bundle.

## Facts established

- Match: 160 s. Auto 20 s, transition 10 s, four 25 s shifts, end game 30 s.
- Hubs alternate active and inactive during shifts; the alliance that scores more FUEL in auto starts inactive in shift 1.
- FUEL: 1 point each when scored in an active hub. Tower L1 15 (auto) / 10, L2 20, L3 30. RP: energized, supercharged, traversal, win 3, tie 1.
- Replays: a pose sample every 80 ticks (about 1.33 s at 60 ticks/s). Controller inputs are not stored.
- Robot heading: forward is (cos a, sin a) with a = pi - quaternion yaw (checked against replay velocities, median alignment 0.79).
- Speed cap confirmed: no replay segment averages above 5.01 m/s.
- Robot specs: 20 robots. Overload (team 254) hopper capacity 72 matches the top pro run's peak.
- Practice toggles: Brownouts (battery sag cuts power), Beaching (stuck on FUEL). Blackout is a visibility mode (field dark, robot circle and hub lights only).

## Known gaps

- Replays don't contain controller inputs or the FUEL push model. FUEL positions are in the replay frames but aren't decoded yet.
- 4 of the 20 robot specs are not mapped to a team or name.
- Event scrape incomplete (29 of about 55 done). Run `python3 -m frsbot.events` to resume.
- Turn acceleration limit is assumed (`TURN_ACCEL` in `sim.py`), not confirmed.
- The bot's FUEL count is copied from the pro's timeline, not decided by the bot.
- Multiplayer replays: none yet. Only solo leaderboard runs.

## Commands

```
node frsbot/scrape/fetch_board.mjs data/board.json
python3 -m frsbot.scrape.harvest data/board.json
python3 -m frsbot.scrape.events
python3 -m frsbot.sim.push
python3 -m frsbot.baseline.route_follow
python3 -m frsbot.tools.build_viewer
python3 -m pytest -q
```

## Real-engine offline backend

`frsbot.sim.wasm_env.WasmEnv` uses the game's real Rust/WASM physics, FUEL, intake, shooting and match rules
in a local Node process; the older drive/push models remain approximation/baseline tools. Node is found on
PATH or under `~/.nvm/versions/node`. Download public assets with curl once; game files are gitignored under
`data/engine/` and must never be committed. See `docs/ARCHITECTURE.md` for the API and JSON-lines protocol.

```
python3 -m frsbot.sim.engine.fetch
python3 -m pytest -q tests/test_engine_fetch.py tests/test_wasm_env.py
python3 -m frsbot.tools.benchmark_engine
```

Use `with WasmEnv() as env`, `env.reset(team, alliance="blue", start={"fx":0,"fy":0,"headingDeg":0}, seed=1)`,
and `env.step(Inputs(...), ticks=60)`. Observations match replay world coordinates; `env.view` retains the
full raw engine view (including score, gates, all FUEL arrays and hub state). Inputs are held for each 60 Hz tick.

## Next steps

1. Decode the FUEL section of replay frames (504 balls per run) into per-sample positions.
2. Fit a push model: how a ball moves when a robot's bumper reaches it. Validate against the next sample.
3. Let the bot decide its own intake and scoring from hub status and hopper level.
4. Show hopper count in the viewer.
5. Map the 4 unnamed robot specs to teams. Make the bot use the robot it is driving.
6. Find out how the live feed samples robots (is it denser than 80 ticks?).
7. Decide how to get controller inputs, either a sim outside the browser or a denser live feed.

## Constraints

- Don't use a ranked or online room with the bot without checking the game's rules first.
- Driver name `404` is the owner's account. Its replays are excluded from training data.
