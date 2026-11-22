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

Auto 0-20 s, transition 20-30, shifts 30-55 / 55-80 / 80-105 / 105-130, end game 130-160.
Both hubs active in auto, transition and end game. In shifts they alternate; the alliance that scored more auto
FUEL is inactive in shifts 1 and 3. FUEL scores 1 only into an active hub. Check the schedule against replay `metas`.

## Planner (the decision part)

Replans every 0.5 s. Inputs it uses: hopper h, capacity C, own hub active/time-to-toggle, ball density map,
time left. Definitions:
- `T_score(p)`: drive time from p to the nearest shooting spot + h / shot_rate.
- Ball clusters: where==0 and height < 0.2 m, binned to 0.5 m cells, summed over a kernel the width of the intake;
  value of a cell = balls / (drive time + 0.5 s).

Rules, first match wins:
1. Hub active and the active window left <= `T_score(here)` + 1 s and h > 0: Score now.
2. Hub active and (h >= 0.85 C, or no cluster worth >= 2 balls/s within reach): Score.
3. Hub inactive and h >= 0.95 C: Stage at the shooting spot, arriving when the hub turns active; start Score then.
4. Hub inactive and time to activation < `T_score(here)`: Stage.
5. Otherwise Collect at the best cluster. While the hub is active, discount clusters by their distance from the
   shooting spot; while it is inactive, don't.
6. End game, last 3 s: Score whatever is carried if in range, else keep collecting nothing (stop).

## Driver

- Collect(point): pursue the point with the intake (front) leading; slow to 60% within 1 m; intake held on.
- Score(point, aim): drive to the point, turn to the aim, hold `score` once range/heading/speed gates are met
  (thresholds from data/shoot_model.json).
- Stage(point): drive there and face the hub, intake on for any balls on the way.
- Commands go through the inverse deadzone so the game sees what we meant.

## Models fitted from replays (all held-out validated, numbers in the JSON next to the params)

- push.py: free roll decay + rectangle push-out using the robot's bumper length/width.
- intake.py: mouth rectangle in robot frame from where 0/1 -> 2 transitions; shot rate (balls/s), and the
  distance / heading-error / speed envelope in which hopper count drops while hub-bound balls appear.

## Evaluation

`python -m frsbot.tools.evaluate` runs Bot and baseline on every robot spec for 5 seeds and prints mean score,
FUEL scored, FUEL scored while inactive (wasted), and hopper overflow. A change to the bot lands only if it
doesn't lower mean score.
