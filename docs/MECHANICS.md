# Deep game mechanics — verified facts and open questions

Sourced from the engine JS/WASM (`data/engine/`) and human replay statistics.
This is the layer strategy actually depends on — update as facts are verified.

## Verified from engine source

- **Intake capture**: balls enter a mouth/belly zone by physical contact, then are
  swallowed into transit at `intakeRate` balls/s (per-robot spec) with a burst budget
  and backpressure scaling (rate drops as hopper fills past `backpressure.from`).
  Capture rate in uniform density = density × mouth width × speed, capped at
  intakeRate — speed is strictly good for capture until the cap binds.
- **Holonomic drive**: translation fully independent of chassis heading (swerve);
  verified empirically at 4.4 m/s perpendicular to facing.
- **Trenches have fixed and hinged sections** (`trench-<alliance>-fixed|hinged`,
  with door bars/pins: `doorMinX/doorMaxX/doorBarMinY/...`). The hinged part is a
  physical door — passability is geometry, not a binary tile.
- **`trenchCut` per-robot spec**: folding the intake arm (`armX`, sometimes `end:"shooter"`)
  to fit under the trench. Implies robots have different folded profiles.
- **Firing**: score press latches on a static aligned tick and persists through
  translation; xLock robots are pinned while latched; pressing score during an
  INACTIVE hub dumps the hopper for zero points.
- **Human speed law** (136k pro frames): flat ~2.6–3.1 m/s through every collect
  distance bin including point-blank, brake only into the hub, leave for the hub
  at hopper fraction ~0.88 (our old default was 0.3).

## Reported by the driver (user), must verify and exploit

1. **Hopper expansion**: some robots' hoppers expand as they fill — a full hopper
   no longer fits under the trench. Pathing must be fullness-aware per robot.
2. **Per-robot trench capability**: some robots can never go under the trench
   regardless of load. Need the per-robot height/fit matrix.
3. **Pre-match auto direction**: before the match you can set your auto direction.
   Need to find how it's communicated to the engine (start config?) and what it
   affects for a bot that drives itself from t=0.

## Open extraction targets (next session)

- Per-robot physical spec table: bumper dims, height, intake arm fold, hopper
  expansion geometry, capacity, intakeRate, fireRate — extract from `2c3zgga2f0b2g.js`
  into `data/robot_feel.json` (partially done) + a capability matrix.
- Trench door physics: when does the hinged door open, can robots force it, does
  it depend on direction of travel.
- HP throw/unload scheduling (seeded per match, in the Rust binary).
- Human tactical patterns beyond movement: positioning vs hub-window phase,
  lane choices, departure timing — mine from `data/imitation/solo-*.npz`.

## Lane-bias experiment (2026-11-25)

`lane_bias` (momentum-aware cluster ranking, penalize targets behind velocity):
eval 254 ~691 / 1678 ~309 vs cruise-only ~700/282 — noise-level difference, kept
as a param (default 1.0) for evolution to judge. Evolution (CMA-ES, 18 genes) at
gen 5 best 614 ≈ baseline; runs in `data/rl/evolve_run2.log`, checkpoints in
`data/rl/evolved_params.json`.

## Trench capability findings (2026-11-25 late)

- Driver treats trench posts as obstacles (x |3.048..4.242|, |z| 2.40-2.85) for
  robots WITHOUT `spec.trench_cut`; the 0.565 m bar height is the fit constraint.
- Extraction status: only 3/20 robots have trenchCut in `data/robot_feel.json`
  (1323, 6329, 9785) and the engine spec file contains exactly 3 trenchCut defs —
  so either most robots genuinely can't fold under, or capability is encoded in
  raw geometry (height) rather than the trenchCut flag. NEEDS: per-robot height
  extraction + empirical drive-through test per robot (`specs.py` has the fields).
- NOT yet modeled: fullness-dependent fit (hopper expansion blocks trench when
  full, per the driver). Pathing is currently binary per robot, load-blind.
- driver.mjs already mirrors the trench-post obstacle logic.

## PASSING — the core inactive-window strategy (2026-11-25, driver-reported + verified)

- Every robot brain has a `passShot` table (hoodDeg ~45 high arc, distances to
  12.4 m, rpm/deg/tof per distance) distinct from `hubShot`. Brain states include
  PASS and PASSING_MANUAL_SHOT; `isReadyToPass` is part of the same gate logic
  as scoring — **pressing `score` while out of hub range makes the brain pass
  automatically** (probe: 4414 at (-4.8,3.5) fired balls that landed ~4m upfield
  in a consistent zone x -1..-2.9 z~0, toward own side).
- Pro strategy (from the driver, visible in records): during the OPPONENT's shift
  (hub inactive), don't park with a full hopper — PASS fuel toward your own side,
  emptying the hopper so collection never stops during the 25 s dead window, and
  pre-positioning fuel near your scoring side for short cycles when your window opens.
- Current bot behavior is the wrong one: trip sequencing fills the hopper during
  inactive windows then Stage-parks frozen for 10-25 s (observed in traces at
  x ±5.5 z ±2.6 — those "stuck" episodes are partly staging waits, NOT wedges).
- The engine has `outposts` — HP feeding stations; passes may target/involve them.
  VERIFY: where exactly pro passes land (mine solo-*.npz: hopper drops during
  inactive windows + fuel where==1 trajectories), whether passing into the inactive
  hub region wastes balls, and whether pass landing zones depend on robot position.
- Movement is STILL too slow per the driver — after passing is in, re-examine.

## Implementation spec (next session)

1. In `imai.py` inactive-window branch: when hopper >= ~0.7*capacity, emit a Pass
   behavior — press score while positioned in own half/midfield (out of hub range),
   until hopper low, then resume collecting. Brain handles arc/landing automatically.
2. Eval on 4414/254/1678 vs current. Watch for: passes landing in inactive hub
   (waste) — if so, aim position matters, face away from hub.
3. Then: departure timing should shift — collect pass cycles during opponent shift,
   arrive at window open with hopper full AND field pre-positioned.

## Ranked multiplayer mining — first stats (10 matches, 2026-11-25)

- Fire position |x| mean 4.99 (close to hub) — humans fire near-stationary:
  only 11% of firing bursts happen while moving >0.5 m/s. Park-and-dump is
  how even good humans score; the differentiator is collection speed/routing.
- Pro speed when moving in ranked: p50 2.51, p90 4.02 m/s — consistent with
  the solo-derived movement law.

## Passing v1: measured REGRESSION (2026-11-25)

pass_frac 0.7 overlay eval vs same-code-no-passing: 4414 690 vs 710 (−21),
1678 283 vs 309 (−26), 254 flat. Hypothesis: emptying the hopper during the
inactive window means the active window opens with nothing to dump — losing
the guaranteed full-hopper first dump beats the pre-positioning gain.
REFINEMENT NEEDED: pass only down to a reserve (e.g. stop passing at 40-50%
so window-open still has a partial dump), or only pass excess collected in
the final seconds, or verify landing zones aren't wasteful first. ALSO:
hopper-full timing — should arrive at window open FULL, pass only early-window
excess. v1 kept in code but needs retune before adoption.

## Methodology corrections (driver, 2026-11-25)

- NEVER aggregate behavior stats across robot archetypes: xLock robots must park
  to fire, moving-shot robots don't. Bin by team number.
- NEVER aggregate ranked players as "pros": mixed skill pool. Use per-match
  winner/high-scorer as the strong-player filter; winner-vs-loser comparison in
  the SAME match is the cleanest signal for what decisions matter.
- Priority order per driver: path FINDING (which balls, which order, which route,
  window timing) over movement speed. Movement law is done; target/route selection
  is the gap. Extraction target: do winners choose different targets/routes than
  losers given the same field state?
