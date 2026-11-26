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

## Path-shape metrics (2026-11-25) — the arc gap, quantified

`frsbot/analysis/pathshape.py` computes the observables. Pro reference
(solo leaderboard): turn_rate ~0.06-0.08, reversal_rate ~0.0003, cruise 2.5-2.7,
stop_frac 0.16-0.48. Bot (imai-4414 evals): turn_rate 0.13-0.17 (2.3x pro),
reversals up to 10x, cruise 1.6-2.0, stop_frac 0.52-0.58. Root cause hypothesis:
point-to-point retargeting every 0.5 s replan forces constant sharp corrections;
pros commit to long arcs. These metrics are fitness-shaped — eval any trace
against PRO_REFERENCE.

## CORRECTED path-shape analysis (units-fixed, dt-invariant)

The earlier "bot curves 2.3x sharper" was a sampling-rate artifact (30hz pro vs
10hz bot traces). With dt-invariant metrics: curvature/m pro 0.58-0.84 vs bot
0.82-0.84 (SAME); accel smoothness pro 2.8-5.2 vs bot 2.8-2.9 (SAME); speed cv
pro 0.34-0.58 vs bot 0.37 (SAME); reversals ~4/match (rare). Real residual
gaps: cruise 1.9-2.0 vs 2.5ish and frac-time-slow 60-65% vs 43-53%.
CONCLUSION: micro-kinematics are within pro range — the visual difference is
semantic (wrong places, wrong targets, dead time), not mechanical. Movement
polish is a dead end; 100% of remaining gap is decision-making: target/route
selection, window timing, passing. Swoop experiment (direction lock near piles)
showed no measurable benefit — left in code behind swoop_m param, unvalidated.

## PRO vs BOT deep run comparison (4414, phase-by-phase, 2026-11-25)

Pro telemetry cnkih62odb6gaok0 vs bot eval imai-4414-blue-1, per match phase
(dist traveled, balls gained, balls dumped/expelled, move%, mean |x|, end hopper):

PRO: auto gain188/dump151; shift1 dist49m gain219 dump169 move81% |x|2.1 endhop82;
shift2 dist17m gain191 dump235 |x|6.8; shift3 dist63m gain176 dump132 move90% endhop81;
shift4 dist16m gain154 dump231 |x|6.8; endgame gain239 dump231.
BOT: auto gain107; shift1 dist22m gain119 dump7 move42% endhop115;
shift2 gain102 dump209 move43%; shift3 gain118 move50%; endgame gain146.

FINDINGS:
1. Pro collects 1246 balls/match vs bot 826 — the entire score gap.
2. THE SMOKING GUN: move% pro 73-90% vs bot 34-56%. Bot sits stationary at piles
   half the match; intake is rate-capped so a stationary robot starves after
   depleting mouth-range fuel. Continuous motion keeps the cap saturated.
   NEVER STOP MOVING during collection — arrival logic at piles is the enemy.
3. Pro window rhythm: opponent shifts = roam midfield (|x|~2.1), gain ~200 AND
   pass/expel 130-170 (passing confirmed in pro behavior at scale), end window
   ~82 full; own shifts = park deep (|x|~6.8), dump 231-235.
4. Bot holds everything through opponent windows (endhop 115, passes ~nothing),
   correct rhythm on paper but half throughput due to static collection.
5. Pro scores ~151 in auto vs bot ~107 — auto collection matters too.

CONCLUSION: the architecture rebuild is continuous lane sweeping during collection —
no arrival, no parking at piles — plus pass-cycling in opponent windows. This is
now measured fact, not hypothesis.

## The reversal mechanism — why no-stop didn't work (2026-11-25 late)

No-stop-at-piles experiment: move% barely moved (shift 42%→50%), scores noise-level.
ROOT CAUSE: the driver aims AT points; after passing through a pile the planner's
next target is often BEHIND the robot — a 180-degree velocity reversal. Reversing
direction physically requires decelerating to zero and re-accelerating, so the
stop-go pattern survives any speed law. THE FIX MUST BE PLANNER-LEVEL: chain
collect targets along forward lanes (directional commitment), i.e. lane sweeping
as the primary collection primitive. lane_bias=1.0 soft bias insufficient.
ARCHITECTURE SPEC (next session, highest priority):
1. Planner emits lane segments: entry point + direction + length along density
   ridges, not cluster centroids. Chain lanes forward — next lane's entry must be
   within ~90 deg of current travel direction, else extend current lane or turn
   to a hub trip.
2. Driver lane mode: hold lane direction at cruise, intake on, no arrival braking;
   lane ends => planner chains next forward lane.
3. Opponent-window rhythm: pass excess (keep reserve for window-open dump — see
   passing v1 failure), end window full and near hub.
4. Validate with move% (target 75%+ in collect windows) AND score.

## THE 3x FIRE-RATE GAP — primary lead for next session (2026-11-25 final)

Measured chain (all verified this session):
- Bot 4414 fire uptime 0.305-0.313 ALREADY MATCHES pro 0.319. Uptime is not the gap.
- PRO DRAIN RATE: sustained 30 balls/s (p50=mean=p90=30 — hard engine cap, 1 ball/2 engine ticks).
- BOT DRAIN RATE: 4414 p50 10 mean 13.6; 254 p50 20 mean 23.8. Same robot, same continuous
  score-press, parked at speed 0 — yet half-to-third the throughput. THIS IS THE SCORE GAP.
- Bot's longest drain: 7.9s continuous press, parked, drains only 13.8/s. NO stutter — smooth
  but slow. So not gate flapping; a continuous mechanical property.
- Pros NEVER press hood/stow/climb/block (147k recovered samples) — no secret button.
- Pro presses score 55% of ticks (bot 40%), intake only 41%.
- AGENTS.md says "fire rate caps 4414 at 14" — PRO DATA CONTRADICTS (30/s achieved).
  The 14/s figure was probably mismeasured from bot behavior!
- Pro parks DEEP: mean |x| 6.7-7.3 in own windows vs bot score spot r≈3.5 from hub center.
  HYPOTHESIS A: fire rate depends on shot distance/hood solution — close shots cycle faster.
  HYPOTHESIS B: flywheel RPM dependent — spin-up state persists at higher RPM when continuously firing.
  HYPOTHESIS C: pass vs hub shot selection — bot's spot triggers slower passShot table.
  NOTE: close score spots r2.0/2.4 previously tested as regression (-90 to -230) BUT that was
  measured in the old architecture with other bottlenecks dominating — retest with rate instrumentation!
- NEXT SESSION: probe engine fire rate vs distance directly (park at r=1.5/2.5/3.5/5, measure drain).
  Then check shot table logic in wasm extraction for rate-limit mechanism (reload time per shot type).

## Lane sweep results (implemented, then defaulted OFF)
- sweep_lane_m param: collect goals target lane EXIT beyond centroid (direction-committed
  by velocity once passing through). Clamped to field margin, truncated at hub boxes.
- BEHAVIORAL WIN: move% during collect 0.34-0.56 -> 0.75-0.84 (pro 0.73-0.90). Path length
  156-208m = pro territory. Collection rate hit 7.3 balls/s (pro level) on 254-red.
- BUT score regression -40 to -60 same-cell (4414: 690 base vs 628 sweep; 254: 666 vs 607).
  Collection was never the binding constraint — fire throughput is. Sweep also lengthens
  trips. KEPT OFF (sweep_lane_m=0) until fire-rate fix lands; then re-evaluate.
- Fixed real bug regardless: unreachable exit points wedged robot 42s at (-7.16, 0) —
  clamping/truncation is correct independent of sweep being on.
- Pro fire pattern: fires EVERY 10s bin of match. Own windows deep |x|~7 (hub shots),
  opponent windows midfield |x|2-4 (passes). Passes land own-half x in [-3.5,-1], 1.5-2.75m upfield.
- Winner-vs-loser aggregate stats flat (cruise/positioning identical) — difference is
  phase-dependent timing, per-window analysis still incomplete.
