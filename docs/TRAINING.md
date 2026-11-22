# Training architecture

Goal: move from hand rules to a policy learned from pro replays, then improved by self-play in the real engine.

## Layers

```
GameState ─► Planner policy (learned) ─► Goal(kind, target, flags) ─► Driver (analytic) ─► Inputs
```

The driver stays analytic and safe at first (pursuit, deadzone, hub routing, gate recovery). Learning targets the
planner: which FUEL to go for, where to shoot from, and when to shoot or wait. A later stage may learn a residual on
the driver's sticks.

## 1. Input recovery (inverse dynamics)

Solo replays sample state at 30 Hz and spectator recordings can sample at 60 Hz. Neither records inputs.
The recovery CLI partitions each replay into non-overlapping approximately 2-second segments.
Process workers run independent matches using the real WASM engine; they restore full body pose/velocity and
field/air FUEL (`fo_body_set`, `fo_ball_set`) at the start of each segment. They initialize hoppers via engine
preloads, reconstruct angular and ball velocity from adjacent frames, and search field-relative stick trajectories
and intake/score button hypotheses. A trajectory starts from observed displacement and velocity, corrects against
intermediate simulated poses during search, and is ranked against recorded pose, angle, velocity, hopper and FUEL.
The chosen inputs are then replayed from a fresh identical reset, with no corrections or intermediate resets.
The target is a 2-second open-loop planar endpoint error below 10 cm.

The numerical `data/imitation/<run>.npz` contains recorded `GameState` columns, `Inputs` and residuals for
each robot/control interval, and a goal per robot/segment. Duration is in engine ticks; rows sharing a segment
share the same initial reset. Large residuals are flagged and weighted at 0.25 instead of 1.0. Read examples as
`(GameState, Inputs, residual, Goal)` through `frsbot.imitation.dataset.examples(path)`. Dataset and raw recordings
are ignored by Git.

```sh
python3 -m frsbot.sim.engine.fetch
python3 -m frsbot.scrape.harvest data/board.json --retain-raw
node frsbot/scrape/spectate.mjs --minutes 60 --concurrency 2 --decode --output data/raw_mp/batch
python3 -m frsbot.imitation.recover
python3 -m frsbot.imitation.report
```

The CLI defaults to two ticks per control and one worker per available core. Use `--control-ticks 6` and
`--segments-per-replay 8` for a cheaper, evenly spaced validation sample. Solo driver `404` is excluded; ranked
spectator identities are anonymized, so multiplayer results are reported separately and cannot be treated as
identity-filtered data. The simulator runs locally in freeplay with no built-in auto, no climbing, and HP disabled.
Replay phases/hubs remain in observations, but the engine does not restore the match clock, shooter internals,
HP schedule, or off-field FUEL lifecycle. Preload seats cap the hopper state that can be restored. These limitations
affect hopper/FUEL residuals and mean the recovered controls are estimates, not original driver inputs.

## 2. Goal labelling

From states alone, label each pro segment as Collect(cluster), Score(spot) or Stage(spot). Local pickups and hopper
growth indicate collection; local launches around the shooting arc, subsequent hub arrivals, and hopper drawdown
indicate scoring. Motion toward dense nearby field FUEL can also indicate collection. Otherwise the end of the
observed path is the staging target. In multiplayer, ambiguous transfers are attributed to the nearest robot.
These are weak labels for planner training, not ground truth about a driver's intent.

## 3. Imitation

- Planner: classify the goal kind, then regress the target, from features (own pose/vel, hopper/capacity, hub
  active and seconds to toggle, time left, FUEL density grid around robot, robot spec vector).
- Small MLP/CNN-on-grid models in PyTorch, trained across all robots with the spec vector as input, so the policy
  adapts to the robot.
- Held-out runs; metric: goal-kind accuracy, target error, and closed-loop score in `WasmEnv`.

## 4. RL fine-tuning

- Vectorised `WasmEnv` workers (one Node process each, ~8k ticks/s). PPO on the planner at 2–4 Hz decisions.
- Reward: points scored, with small shaping for FUEL collected and a penalty for FUEL held at match end.
- KL penalty to the imitation policy early on so it stays human-like.
- Randomize robots, alliances, seeds and HP settings.

## 5. Multiplayer

Verified: the engine runs several robots in one match. `create()` takes a `robots` list (id, team object, alliance,
start, preload) with `solo` unset. Three robots on two alliances stepped with separate inputs, and per-alliance
scores came back in `view.score.{red,blue}`. The server and `WasmEnv` need a multi-robot `new`/`step`, with an
input per robot id.

### Decision architecture (same model for 1-6 robots)

```
GameState (ego view) ─► Encoder ─► Option head ─► Goal ─► Driver ─► Inputs
                         │   ego: pose, vel, hopper/capacity, spec vector, gate/mode
                         │   entities: teammates + opponents (pose, vel, spec vector, est. hopper, est. goal)
                         │   FUEL grids: resting / in flight / near hub, ~0.5 m cells
                         │   match: phase, own/opp hub active + seconds to toggle, time left, score diff
                         └── entity set via attention, so 1v1, 2v2 and 3v3 share one network
```

Options: `Collect(cell)`, `Score(arc slot)`, `Stage(slot)`, `Pass(cell)` (the brains' passShot, for feeding FUEL into
our zone while the hub is inactive or from far away), `Contest(cell)` (take neutral FUEL an opponent is heading for),
`Defend(opponent)` (screen their shooting lane or route, within the foul rules), `Yield` (move clear of a teammate).
Targets are discrete cells/slots, so the action space stays small. The driver adds robot-robot avoidance (velocity
obstacles, with teammates given priority by role) and foul guards: no pinning or contact timers past the limits.
The limits come from the game rules and bundle, not guesses.

Each robot runs its own copy of the policy on its own view (decentralised execution, as in the live game). It
coordinates through what it observes, plus an optional shared team plan when we control all teammates.

### Team planner baseline (rules, before learning)

- Assign clusters to teammates with a Hungarian match on (value / travel time), so two robots never chase the same
  FUEL.
- Roles by robot and phase: the biggest hopper/fastest shooter scores; a robot with good pass speed feeds while our
  hub is inactive; in a 3v3, one robot may defend during the opponent's active window, if expected points saved >
  points it would score.
- Opponent model: robot spec from team number, hopper estimated by counting the FUEL it picks up and shoots, goal
  predicted by running our own imitation planner from their point of view.

### Strategy layer (match situation and role)

The option policy is conditioned on a strategy state, updated every 2 s, so the robot plays its role, not just
greedy scoring:

- Inputs:
  - Each robot's expected scoring rate: spec rate (capacity, fire rate, intake rate, drive) × observed rate so far.
  - Score difference and its trend.
  - Time left, and the current/upcoming hub windows for both alliances.
  - Remaining neutral FUEL, and who is defending whom.
- Role per robot: `Scorer`, `Feeder` (collects and passes into our zone), `Defender(target)`, `Contester` (denies
  neutral FUEL).
- Assignment: maximise expected alliance margin = our scoring − opponent scoring, where defense against opponent j
  subtracts `rate_j × suppression(defender, j)`. Suppression is measured from real matches (see
  `docs/MATCH_ANALYSIS.md`). The weakest scorer goes to defense when its suppression of the best opponent beats its
  own scoring. Typical case: during the opponent's active window while our hub is inactive.
- Situation rules on top:
  - Protect a lead late by defending or contesting.
  - Chase a deficit by having everyone score during our active windows.
  - Never defend when we can't catch up on points.
- When we are defended (an opponent repeatedly within ~1.5 m, or blocking our lane/spot):
  - Switch to alternate arc slots away from the defender.
  - Shoot from longer range if the spec allows.
  - Pass to a teammate instead of shooting.
  - Bait the defender away from the scoring teammate.
  - Request a teammate screen.
  - Keep the evasion time cost below the points lost by staying put.
- Learned version: the strategy head outputs a role distribution from the same encoder. It is trained first on role
  labels from real matches (imitation), then by self-play (MAPPO). The rule assignment above is the baseline and the
  fallback.

### Real match analysis pipeline

Spectator recording of live ranked/event games (passive, read-only), plus multiplayer replays where available →
multi-robot `GameState` → `frsbot/analysis/matches.py`. That gives per-robot FUEL attribution, role timelines
(scorer, collector, feeder, defender), defended/undefended scoring rates, and when teams switch roles relative to
score, time and hub windows. The outputs:
1. Role labels for imitation.
2. Suppression and evasion-cost tables for the strategy layer.
3. Opponent models of real players.

### Training

- Centralised training, decentralised execution: MAPPO with a critic that sees the full state (all robots, true
  hoppers). The actor sees only the ego view.
- Reward: alliance score difference (zero-sum), plus a win bonus and RP shaping (energized/supercharged). Small
  per-robot shaping for FUEL scored, and penalties for fouls and FUEL held at the end.
- Curriculum: solo (sections 1-4) → 1v1 vs the rule bot → 2v2/3v3 self-play. Robots, alliances, seeds and team
  compositions are randomised.
- League: current policy vs frozen snapshots, the rule bot and exploiter policies trained to beat the current one.
  This avoids strategy collapse. Rank with TrueSkill.
- Infrastructure: vectorised Node engine workers with batched multi-robot steps. Decisions at 4 Hz, driver at
  20-60 Hz.

### Evaluation

TrueSkill/win rate vs the rule bot and the league, points per match, foul rate, and duplicate-cluster rate between
teammates. Plus ablations: no opponent model, no Defend, no Pass.

## Constraints

- Driver `404` replays are excluded.
- No ranked/online use without checking the rules.
