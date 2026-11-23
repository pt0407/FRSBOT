# Trip sequencing redesign (spec)

Status: proposal. Branch `feat/trip-sequencing`. `main` stays unchanged until the acceptance gate passes.

## Problem

Live singleplayer scores: 626 (team 254), 608 (team 4414), 460 (team 2910). Target: 1200.

Measured on the offline engine (same physics, solo, seed 1):

| Robot | Time Collect | Time Score (at spot) | Time Stage | Time travel | FUEL collected | FUEL released |
|---|---|---|---|---|---|---|
| 4414 | 28% (46.5 s) | 36% | 23% | 11% | 592 | ~663 |
| 2910 | 34% (57.0 s) | 8% | 30% | 27% | 425 | ~400 |

Pro reference (EinSteiner3, 1407, team 254): hopper rises 36 s, falls 42 s, intake ~38 FUEL/s while
rising, release ~33 FUEL/s while falling, hopper peak 72. Bot: rises 9 s, falls 10 s on live 4414.

Robot limits (from `data/robot_feel.json`): 2910 hopper 47, fire rate 32; 4414 hopper 115, fire rate 14;
254 hopper 72, fire rate 26. Intake rate is 50 for all three, so intake hardware is not the cap.

## What has been ruled out (6-case screens, then full 30-case protocol where noted)

Each change defaults to the shipped value. None beat baseline by more than noise.

- Stall handling near walls (live stalls total ~6 s per match; not the cap).
- Scoring trigger `score_full_frac` (0.5, 0.7, 0.95): lower values hurt both robots; 0.95 helped 4414
  by 15 and hurt 2910 by 16. Within screen noise.
- Stage fullness and arrival margin (`stage_full_frac`, `arrive_margin_s`): worse on both robots.
- Intake gate `intake_needs_fuel`: 535.7 full protocol vs 530.2 baseline; live 598 vs 608.
- Cluster travel bias `cluster_bias` (0.15, 1.0): both below 2910 baseline.
- Learned event hint from `pro_planner` (thresholds 0.6 and 0.85): worse on 4414 and 2910.
- Collect speed factors `collect_near_frac`, `collect_cos_floor`: no effect or +11 (noise).

## Diagnosis

The bot's score is bounded by collection volume and by how little of the match it spends collecting.
Speed formula check during Collect (team 2910): top speed binds 58 steps, 2x-distance binds 39, stopping
distance binds 17; median distance to target 3.5 m. So the driver's limits are not the sole cause.

The likely cause is the goal sequence itself: Stage and travel take 57% of time on 2910, and the bot
releases roughly 400 FUEL while it could collect more between shifts.

## Proposed design

1. Trip planner that commits to a collect run of N FUEL (N = min(capacity, shift-remaining budget)),
   then scores in the active hub, rather than re-planning on each cluster.
2. Shift-aware timing: start the last collect of a shift so the score trip lands when the hub turns on.
   Hub schedule from `frsbot/core/field.py`.
3. Keep the existing Driver for motion. Change only goal sequencing, behind `trip_sequencing=1`
   (default off).

## Acceptance gate

Offline, 30-case protocol (five teams, both alliances, seeds 1-3), mean score:
- must beat the current baseline of 530.2 by at least 40 points, AND
- must not drop more than 20 points on any single team below its baseline value.

Only if the gate passes: one live singleplayer run per tested team, reported as-is.

## Out of scope

- Multiplayer or ranked play. Solo only until the rules review in `AGENTS.md`.
- Training data changes. The pro planner stays as-is.

## Open questions

- Is 1200 reachable with the current drive and intake model, or does the robot need faster intake
  cycles? Unknown until the offline gate is run.
- Which robot to target first. Team 2910 has the fastest shooter (32) but the smallest hopper (47).

## First implementation result (flag off by default)

`trip_sequencing=1`: commit to Collect until the hopper reaches `trip_fill` (0.85) of capacity, then return
to the normal planner. Six-case screens: team 2910 151.5 (baseline 516.8), team 4414 438.8 (baseline 649.5).
Far worse on both. The simple commit rule starves the scoring cadence; the shift-aware timing in step 2 of
the proposed design is still missing, and that is the part that would need to be tested next.

## Second implementation result (shift-aware, flag off by default)

`trip_shift_aware=1` limits the Collect commitment to hub-inactive time. Six-case screens: team 2910 511.3
(baseline 516.8), team 4414 643.7 (baseline 649.5). Both slightly below baseline, so the acceptance gate
(+40 over 530.2 on the full protocol) is not met. The commit rule alone does not raise collection volume.
