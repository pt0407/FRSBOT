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
