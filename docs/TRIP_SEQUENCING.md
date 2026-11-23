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

## Cluster sweep width (flag-free knobs, defaults unchanged)

`sweep_min_m` (default 0.6) and `sweep_scale` (default 0.8) set how many FUEL balls a cluster counts.
Full-protocol six-case screens: team 4414 `sweep_min_m=1.0` 650.2 (baseline 649.5), `sweep_scale=1.2` 664.0.
Team 2910 `sweep_min_m=1.0` 502.0 and `sweep_scale=1.2` 501.7 (baseline 516.8). No setting clears the
acceptance gate, so defaults stay unchanged.

## Collect arrival radius (default 0.3 m, unchanged)

`collect_arrive_m` sets how close the bot must get to a cluster before it re-plans. Full-protocol six-case
screens: team 4414 0.8 m gave 633.3 and 1.5 m gave 654.7 (baseline 649.5). Team 2910 0.8 m gave 525.0 and
1.5 m gave 507.3 (baseline 516.8). All changes are within screen noise, so the default stays at 0.3 m.

## Shoot while collecting (existing flag, unchanged default)

Full protocol on teams 2910 and 4414 (both alliances, seeds 1-3): moving score on gave 516.8 (2910) and
649.5 (4414); off gave 516.8 and 651.3. No meaningful difference, so the default stays as shipped.

## Acceptance gate run (full 30-case protocol)

`trip_sequencing=1, trip_shift_aware=1, trip_fill=1.0`: mean 545.2 vs baseline 530.2 (+15.0).
Per team: 254 629.2 (+28.7), 2910 513.3 (-3.5), 1114 616.5 (+5.3), 4414 667.8 (+18.3), 60026 299.3 (+26.5).
No team fell more than 20 points, but the mean gain misses the +40 gate. Not promoted; flag stays off.

## Hub-state comparison and score-earlier test

Pro CLHNB666 (1250) with the game's own schedule (`is_active`, firstInactive=blue): hub active in auto,
transition, S2, S4, endgame; inactive in S1 and S3. The pro releases 176 FUEL in auto, 97 in transition,
about 250-320 per active shift, and 0 while inactive. In S1 and S3 it intakes about 150 FUEL per shift
instead. The bot's odometry shows it stopped about 70% of S1 and S3.

Bot hub state is computed from the same schedule, not measured per frame.

`active_score_frac=0.3` (score when the hopper reaches 30% during active hub time) on the full protocol:
mean 454.3 vs 530.2 baseline (254 512.0, 2910 408.7, 1114 537.0, 4414 537.8, 60026 276.2). Much worse.
Scoring earlier is not the fix. Default remains 0.85.

## Early-scoring window (auto/transition/first active shift)

`early_score_until=33` forces the bot to score whenever its hub is active before 33 s. Full protocol mean
467.4 vs 530.2 baseline (254 520.7, 2910 451.7, 1114 536.5, 4414 556.3, 60026 272.0). Worse on every team.
Default stays 0 (off).

Replay event log check (pro CLHNB666, 1250): 1075 points scored while blue was active, 175 while inactive,
all within about 4 s of deactivation (the grace period). The late stationary drops in S1 and S3 score nothing.

## Learned input policy (closed-loop result)

`frsbot/imitation/input_policy.py`: state + spec -> sticks and intake/score, trained on 18 recovered pro
replays, held out 4. Held-out stick MSE 0.041 vs 0.155 for predicting zero; button accuracy 0.781 vs 0.698
majority. Closed-loop in the sim on team 2910 (6 cases): mean 11.0 vs about 517 for the current bot.

The policy fails in closed loop. Small per-frame errors compound and the bot stops scoring. Not promoted.
Next step if pursued: DAgger-style training on the bot's own visited states, labelled by the pro-recovered
inputs, or a hybrid where the planner picks goals and the network only supplies stick corrections.

## Capacity check (total vs storage hopper)

The engine reports capacity 47 for team 2910, matching the spec's hopper figure. The pro's peak of 54 is
within the 55 total capacity. `use_total_capacity=1` (planner uses the total) on the full protocol: mean
506.5 vs 530.2 baseline (254 551.2, 2910 434.8, 1114 587.2, 4414 685.5, 60026 273.7). Worse, mainly on 2910.
Capacity is not the cause. Default stays 0.

## Handoff: open items (auto mimic and 20 s handoff)

State:
- `frsbot/tools/auto_mimic.py` replays the recovered pro controls for 0-20 s (team 2910, CLHNB666), then ImAI.
  Blue uses the pro start; red uses the 180-degree mirror.
- Same start, six cases: baseline 509.0 final, mimic 499.5 final. Score at 20 s: 51.8 baseline, 80.5 mimic.
  Within noise over the full match. Not promoted.
- Capacity check and total-capacity knob: not the cause (506.5 vs 530.2). Default off.

Open:
- Handoff at 20 s. Baseline releases about 31 FUEL just before the blue hub closes and scores 40 more in the
  next three seconds. The mimic loses 13 FUEL at about 20 s but gains 1 point. Hypothesis: the mimic's last release
  lands after the hub closes. Check: compare score-event times in milliseconds with the pro's 0-20 s events in
  `data/telemetry/ljqj1ukg3mnpcxvl.json` (event list has `score` events with `alliance`, `points`, `at`).
- Start pose: the engine clamps starts, leaving about 0.44 m of offset to the pro's start.

Shared resources: one live run at a time on the debug Chrome (port 9222). Use a separate branch or worktree
for parallel work.

## Closed-loop auto (route follow, not open-loop replay)

`frsbot/tools/auto_closed.py`: the driver follows the aligned pro route for 0-20 s, with a Score goal at the
scoring spot when the pro sheds FUEL. Six cases (team 2910, both alliances, seeds 1-3): final mean 455.0,
score at 20 s mean 5.0. The route follow alone scored 438.7 (@20s 5.0). The open-loop mimic scored 499.5
(@20s 80.5). The closed-loop variant does not reproduce the pro's auto scoring, so it is not promoted.
