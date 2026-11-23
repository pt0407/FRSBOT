# ImAI controller tuning

## Scope and protocol

ImAI controls the robot from t=0, including the auto phase, with native `auto="none"`.
Built-in autonomous routines were dropped at the user's request. No climbing is attempted.
All final comparisons use the same 30 solo cases as baseline: teams 254/2910/1114/4414/60026,
both alliances, seeds 1/2/3, event rules, HP throws and unloads enabled, 166 s, 20 Hz control.

Scores include human-player FUEL; `robot_fuel` separately records engine-attributed robot points.
Score equals active FUEL in all final cases. The leaderboard is a reference, not a paired experiment:
strategies, seeds and starting poses differ. Native-auto none does NOT mean ImAI sits idle for 20 s.

## Before / after

Lists are seeds 1 / 2 / 3; goal times are three-seed means in seconds. Hopper peaks are observed at 20 Hz.
Baseline is commit `b386e26`, also preserved in `/home/ubuntu/imai-round0`.

### Before

| Team | Alliance | Score / FUEL | Max hopper | Collect / Score / Stage seconds |
|---:|---|---|---|---|
| 254 | blue | 589 / 626 / 683 | 72 / 72 / 72 | 58.3 / 61.5 / 46.2 |
| 254 | red | 454 / 562 / 590 | 72 / 72 / 72 | 47.6 / 72.8 / 45.6 |
| 2910 | blue | 551 / 422 / 509 | 50 / 48 / 50 | 58.2 / 58.8 / 49.0 |
| 2910 | red | 465 / 424 / 424 | 49 / 49 / 50 | 48.1 / 62.3 / 55.7 |
| 1114 | blue | 591 / 601 / 630 | 80 / 80 / 80 | 58.1 / 63.5 / 44.4 |
| 1114 | red | 582 / 591 / 611 | 80 / 80 / 80 | 58.2 / 65.2 / 42.6 |
| 4414 | blue | 359 / 391 / 517 | 115 / 115 / 115 | 28.8 / 99.0 / 38.2 |
| 4414 | red | 380 / 495 / 386 | 115 / 115 / 115 | 29.4 / 97.2 / 39.4 |
| 60026 | blue | 299 / 324 / 217 | 35 / 35 / 35 | 43.1 / 73.0 / 50.0 |
| 60026 | red | 310 / 329 / 306 | 35 / 35 / 35 | 53.5 / 68.6 / 43.9 |

### After

| Team | Alliance | Score / FUEL | Max hopper | Collect / Score / Stage seconds |
|---:|---|---|---|---|
| 254 | blue | 641 / 661 / 571 | 72 / 72 / 72 | 59.9 / 62.5 / 43.7 |
| 254 | red | 633 / 592 / 637 | 72 / 72 / 72 | 57.5 / 64.1 / 44.4 |
| 2910 | blue | 471 / 471 / 507 | 50 / 49 / 50 | 53.2 / 62.2 / 50.6 |
| 2910 | red | 523 / 496 / 525 | 49 / 51 / 49 | 54.8 / 62.6 / 48.6 |
| 1114 | blue | 629 / 632 / 619 | 80 / 80 / 80 | 59.0 / 64.5 / 42.5 |
| 1114 | red | 613 / 642 / 607 | 80 / 78 / 80 | 57.5 / 65.8 / 42.8 |
| 4414 | blue | 673 / 613 / 590 | 115 / 115 / 115 | 47.9 / 81.1 / 37.0 |
| 4414 | red | 662 / 668 / 684 | 115 / 115 / 115 | 47.3 / 78.8 / 39.9 |
| 60026 | blue | 324 / 246 / 305 | 35 / 36 / 35 | 50.0 / 71.7 / 44.3 |
| 60026 | red | 311 / 322 / 308 | 35 / 35 / 35 | 55.7 / 69.1 / 41.2 |

### Leaderboard reference

| Team | Before mean | After mean | After best | Leaderboard best | After mean / leaderboard |
|---:|---:|---:|---:|---:|---:|
| 254 | 584.0 | 622.5 | 661 | 1419 | 43.9% |
| 2910 | 465.8 | 498.8 | 525 | 1302 | 38.3% |
| 1114 | 601.0 | 623.7 | 642 | 1257 | 49.6% |
| 4414 | 421.3 | 648.3 | 684 | 1318 | 49.2% |
| 60026 | 297.5 | 302.7 | 324 | 620 | 48.8% |

Aggregate mean-score sum / leaderboard-best sum improved from 40.1% to
45.6%; aggregate mean score increased 13.8%.

## Progress recovery and final deadline

At a scoring spot, track net hopper drops, with an independent continuously-open-gate timer.
After 1 s open without a drop, or 2 s without progress overall: release score for 0.6 s and manually
re-aim, try a different valid arc point, then a 1.2 s alternating translation/turn wiggle.
After this fails at <=5 FUEL, collect for at least 2 s and target max(10, 0.3 capacity) FUEL,
with an 8 s escape cap. This avoids returning to the stalled Score goal before reaching new FUEL.
The absolute final deadline overrides another collection trip.

Time in Score with <=5 FUEL, seconds/match. Baseline is reconstructed from 10 Hz traces; final is
counted at the 20 Hz control rate. A little more low-hopper Score time on some robots is the cost
of actually draining near the final deadline rather than carrying a large load into settle.

| Team | Before low-hopper Score seconds | After | Hopper at 163 s, blue seeds then red seeds |
|---:|---:|---:|---|
| 254 | 5.87 | 6.69 | 0 / 0 / 0 / 0 / 0 / 0 |
| 2910 | 3.48 | 6.98 | 0 / 0 / 0 / 0 / 0 / 0 |
| 1114 | 5.50 | 8.13 | 0 / 0 / 0 / 0 / 0 / 0 |
| 4414 | 55.87 | 15.34 | 1 / 1 / 3 / 0 / 1 / 1 |
| 60026 | 5.95 | 15.58 | 0 / 0 / 0 / 0 / 0 / 0 |

Empty at 163 s: **7/30 before, 25/30 after**.
Total retained FUEL: **402 before, 7 after**.
**The zero-FUEL-at-163 goal is not fully solved:** five 4414 runs still retain one to three FUEL.
The code does not dump FUEL with outtake just to satisfy that metric.

## 2910 vision/tag investigation

Reference bundle `3gjhnn1vwqqgq.js` sets
`observation = tagSeen ? (visionPose ?? truePose) : null`, and
`gate.tag = observation === null`. Thus a nominally valid shooting range does not guarantee readiness.
In the reference limelight implementation (`2lnu-vtgpos5u.js`, class eR), a tag must face the camera;
all four corners need positive depth, fit inside the image, exceed minimum projected pixel size,
and have unobstructed raycasts excluding the robot's own colliders. There is no simple fixed range gate.

2910's bundle camera config is position [0.33, 0.388, 0] m, pitch 27 degrees, 1280x800 px,
960 px focal length, 10 px minimum tag height and 0.1651 m tag side. This implies approximately
67.4 degree horizontal / 45.2 degree vertical FOV. Rust source was not available; those geometric
details are the JS reference model, checked against real-engine visibility probes, not a claim
that the JS class executes in the WASM backend.

Real Rust/WASM probes held score at rear-side arc offsets -60/-35/+35/+60 degrees on both alliances
at radius 3.75 m: all eight poses had tag unblocked in all 12 samples over 4 s.
The centerline probe was not usable: the tower displaced the robot, confounding range and visibility.
The policy keeps the inner +/-35 degree pair for clearance, roughly blue (-6.717, +/-2.151),
red (+6.717, +/-2.151). Native `sense(0)` returned `tagSeen=true`, non-null `visionPose`
and `gate.tag=false` at the working pose.

A second native-engine experiment disproved the assumption that manual turn can re-aim while
score remains held: 2910 stayed at heading 0.6105 rad after 2 s of full turn + score, then rotated to
2.5503 rad after only 0.5 s of full turn without score. Recovery now briefly releases score.
This is why changing the heading command alone did not reliably recover the previous bot.

Across six matches, commanded-score tag-blocked time fell from **59.35 s to 27.50 s**.
These are command-duration diagnostics, not actual lost-shot counts; recovery includes score-button pauses.
Fixed arcs still do not guarantee continuous visibility after travel, beaching, or pose-estimation drift.

## Measured score timing

Reproduce with:
```
python3 -m frsbot.tools.measure_shooting --output /home/ubuntu/imai-timing.json
```

Three blue-alliance seeds per robot, no HP or unloads, then intake disabled while holding score
from the first loaded attempt. Startup is score-pressed to first net hopper drop. Drain uses
subsequent drops, excluding the first from the numerator. All 15 attempts produced measurements.
Calibration chooses maximum observed startup + 0.2 s, and minimum measured drain rate.

| Team | Startup observed, s | Startup used, s | Drain observed, FUEL/s | Drain used |
|---:|---|---:|---|---:|
| 254 | 1.133-1.200 | 1.400 | 25.714-26.225 | 25.714 |
| 2910 | 1.067-1.100 | 1.300 | 30.370-31.622 | 30.370 |
| 1114 | 0.567-1.083 | 1.283 | 19.904-20.190 | 19.904 |
| 4414 | 0.300 | 0.500 | 14.009-14.017 | 14.009 |
| 60026 | 0.917-0.933 | 1.133 | 9.017-10.196 | 9.017 |

T_score includes route length through bump portals, 0.5 s arrival allowance when travelling,
measured startup and hopper / measured drain rate. Active-window rules use T_score + 1 s;
the final 163 s deadline reserves 3 s for recovery. Late scoring disables opportunistic intake.
An initial extra 2 s per crossing was too conservative and reduced every fast robot's mean;
that version was rejected. Startup/rate estimates do not predict every feed jam or flight time.

## Shoot while collecting

Enabled only by calibrated policy for 4414 among the evaluated robots (the only one with
maxShotSpeed >=2 m/s). It requires own-zone valid range, an active hub, FUEL, and safe speed;
pursuit is capped at 85% of maxShotSpeed while attempting this. The turret aims independently,
so collection keeps its chassis-heading command and intake rather than setting turn to zero.

The final paired six-case 4414 ablation was **637.0 without vs 648.3 with**, +11.3 points/match
(+1.8%). Collection score was held for 29.1 s across the final six cases;
the traces record exact `collect_score_seconds` for this revised recovery policy.
An over-restricted outward-only policy made no difference, and suppressing chassis steering
regressed the mean to 566.7; neither variant was kept. Keeping score held during manual recovery
also regressed, so that experiment was discarded despite 4414 supporting independent steering.

```
python3 -m frsbot.tools.evaluate
python3 -m frsbot.tools.evaluate --teams 4414 --no-moving-score --output /path/to/ablation
```

## Sequential evaluation checkpoints

All entries below are full 30-case no-native-auto evaluations; final rows include follow-up
correctness and deadline refinements, so this is cumulative development, not a clean factorial ablation.

| Policy checkpoint | 254 mean | 2910 | 1114 | 4414 | 60026 |
|---|---:|---:|---:|---:|---:|
| Baseline | 584.0 | 465.8 | 601.0 | 421.3 | 297.5 |
| Progress recovery | 620.7 | 507.7 | 610.3 | 653.8 | 283.7 |
| Visible arcs | 620.7 | 516.3 | 610.3 | 653.8 | 283.7 |
| Measured timing, first pass | 626.3 | 486.0 | 614.3 | 655.3 | 294.0 |
| Final, all native auto none | 622.5 | 498.8 | 623.7 | 648.3 | 302.7 |

Console logs/checkpoints remain under `/home/ubuntu/imai-round*`, final log
`/home/ubuntu/imai-final-noauto-fresh.txt`, final ablation `/home/ubuntu/imai-fresh-nomoving`,
and raw startup/drain observations `/home/ubuntu/imai-timing.json`.

## Remaining weaknesses

1. **Residual feed jams:** five 4414 cases still carry 1-3 FUEL into settle. Recovery removes the long
   low-hopper camps but does not guarantee the last pocket clears before the hub shuts down.
2. **2910 visibility is still intermittent:** 27.5 commanded-score seconds remain tag-blocked across six
   matches. Static visible arcs plus releasing score to re-aim help, but do not model loaded/dynamic camera
   pose and line of sight continuously.
3. **Collection/return duty cycle:** greedy cluster targets and fixed range-circle spots still require
   repeated travel, spin-up and staging. Simultaneous collection/shooting is useful but infrequent;
   the planner does not yet learn replenishment routes or coordinate roles with other robots.

## Checks and artifacts

```text
361 passed in 14.47s
```

Final 30 traces/summary: `data/sim/imai/` (gitignored). The rebuilt viewer contains the final compact
runs. Viewer compatibility is checked programmatically; no browser visual check was performed.
No downloaded engine files are committed. No climbing or built-in auto logic was added.
