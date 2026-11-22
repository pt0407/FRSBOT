# Live play: browser bridge, latency, human teammates

Scope: singleplayer practice first. Don't use ranked or online rooms until the game rules have been checked (AGENTS.md).

## 1. Browser bridge (`frsbot/live/`)

```
Chrome page ──(state tap)──► bridge.py ──GameState──► ImAI.act ──Inputs──► (virtual gamepad) ──► page
```

- **Inputs, as a virtual gamepad.**
  - Playwright `add_init_script` replaces `navigator.getGamepads()` with one connected standard-mapping pad.
  - Its axes and buttons are read from `window.__imai.pad`, which Python writes through `page.evaluate` (or a CDP
    binding) at the bot rate.
  - The game polls gamepads itself, so the bot goes through the exact input path a human uses: deadzone, bindings,
    field-relative toggle. Nothing is patched in the engine.
  - The button-to-action table comes from the game's default controller bindings in the bundle:
    intake, score, outtake, block, stow, hood, field-relative.
  - Keyboard is the fallback if gamepads turn out to be unavailable in some screen.
- **State, from a tap rather than pixels.**
  - Solo: the init script wraps `Worker` so every message the match worker posts to the page is also copied to
    `window.__imai.frames`, keeping the last N. That gives the same view `WasmEnv` gets from `fs_out`/`fs_frame`.
  - Online: wrap `WebSocket` the same way to get the snapshot stream. This is the same bytes as
    `frsbot/replay/spectator.py` decodes, so we reuse it.
  - Both feed one `live_state(view) -> GameState` converter. It must be the same converter as `WasmEnv`, so the
    bot sees identical fields offline and live.
- **Loop.**
  - Poll at 60 Hz, run `ImAI.act` at 20–30 Hz, write the pad.
  - Log `(t, GameState, Inputs)` to `data/live/*.jsonl`. These logs are training data, and the replay viewer can
    play them back.
- **Safety.**
  - The pad is zeroed on any exception, when the page loses focus, or when the match ends.
  - `--mode solo` is the only allowed mode until the rules are checked.

## 2. Latency

- **Measure it.**
  - `ping` messages give the round-trip time online. Solo latency is the worker's message lag plus the gamepad
    poll period.
  - The bridge timestamps every frame and every pad write. It reports the latency from observing a state to the
    input taking effect, measured by step-response tests.
- **Compensate.** Before planning, the bot predicts the observed state forward by the latency L:
  - Ego: if the client's predicted ego pose is in the view (it has `predict`/`rollback`), use it. Otherwise
    integrate the last commanded inputs through the drive model (`data/robot_feel.json`) for L.
  - Other robots: constant-velocity extrapolation, capped at 0.3 s.
  - FUEL in flight: ballistic. FUEL at rest: unchanged.
  - Hub timing: the schedule is deterministic, so evaluate it at t + L.
- **Train for it.**
  - `WasmEnv(latency=(lo, hi))` delays the bot's observation by a random 0–150 ms each episode and holds inputs
    for a random 1–3 ticks.
  - The driver's gains and score gates are tuned under that range, not at zero latency.
  - Evaluation reports score against latency.

## 3. Human teammates (and opponents)

The bot controls only itself. Teammates and opponents are humans, so the strategy layer is a best response to
*inferred* roles, not an assignment:

- **Intent model.**
  - For every other robot, run a classifier on its last ~3 s. Features: velocity toward each FUEL cluster or hub,
    hopper change, distance to opponents, zone.
  - It outputs P(Collect cluster k), P(Score), P(Feed), P(Defend robot j).
  - It is trained on the spectator recordings (`frsbot/analysis`), whose labels come from what actually happened
    next.
- **Best response.** Choose our role to maximise the expected alliance margin, given the teammates' inferred
  roles:
  - If a teammate is the strong scorer, we feed, defend or contest.
  - If nobody is defending the best opponent and we are the weakest scorer, we defend.
  - We switch only after the inference has been stable for about 2 s (hysteresis), so we don't oscillate with a
    human who is also adapting.
- **Deconfliction.**
  - Skip FUEL clusters a teammate is likely heading to (P > 0.5).
  - Keep out of a teammate's shooting lane and arc slot.
  - The driver treats teammates as moving obstacles, using predicted paths.
- **Robustness.** Self-play opponents and teammates are sampled from a population: rule bot, past policies, and
  behaviour-cloned human models from the recordings. The bot must not rely on teammates cooperating.

## 4. Multi-robot `WasmEnv`

- **API.**
  - `reset(robots=[RobotSlot(team, alliance, start)], ego=i, seed)` → `GameState` from robot i's perspective.
  - `step({i: Inputs})` → per-robot views, alliance scores and breakdowns.
  - Robots without a given input use a pluggable policy: rule ImAI, a replayed recording, or idle.
- **Views.** A per-robot view hides nothing the live client wouldn't show. It is tested against live and spectator
  views (§1) for identical fields.
- **Batching.** N matches per Node process, several processes. Throughput target: at least 2k ticks/s per process
  with 6 robots.
