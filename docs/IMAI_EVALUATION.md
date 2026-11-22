# ImAI initial real-engine evaluation

Command: `python3 -m frsbot.tools.evaluate`.

30 full solo matches: 254, 2910, 1114, 4414, 60026; blue/red; seeds 1, 2, 3.
No auto, event rules, site-default HP throws and unloads enabled. 166 s including settle;
20 Hz control (3 physics ticks), 10 Hz trace samples. Scores include HP contributions and
are not directly comparable to leaderboard runs with robot autos. No climbing is attempted.

Isolated HP-disabled smoke tests score on both alliances within 40 s and observe hub-contained
FUEL at negative x for blue and positive x for red, verifying the scoring-hub association.
Native score commands outside the alliance zone pass instead; scoring spots must lie on the
own-zone portion of the range circle. Native auto-aim points the shooter, opposite the intake.

## Results

Each list is ordered seeds 1 / 2 / 3. Score equals active FUEL scored in every run.
Goal durations below are means across the three seeds, in seconds (Collect / Score / Stage).

| Team | Alliance | Scores / FUEL | Max hopper | Mean goal seconds C / S / T |
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

The planner uses the observation's hopper capacity for its storage counter, falling back to
the spec's feed-inclusive total capacity if the observation lacks a capacity. Transport/feed
transients can put observed counts slightly above the nominal storage capacity.

## Gate diagnostics

The engine exposes boolean blocked reasons, not actual lost-shot events. The evaluator counts
overlapping blocked command-ticks and estimates blocked fire opportunities as blocked seconds
times configured fireRate. These cannot be summed into an actual number of lost shots.

Total blocked-command seconds over all six matches for each team:

| Team | Hood | Flywheel | Range | Heading | Speed | Tag |
|---:|---:|---:|---:|---:|---:|---:|
| 254 | 9.20 | 10.85 | 0 | 36.65 | 0 | 0 |
| 2910 | 11.30 | 0.75 | 0 | 43.55 | 2.50 | 59.35 |
| 1114 | 4.65 | 21.90 | 0 | 46.65 | 0 | 0 |
| 4414 | 4.30 | 0 | 0 | 0.45 | 0 | 0 |
| 60026 | 45.20 | 4.80 | 0 | 32.60 | 0 | 0 |

## Leaderboard comparison

Best recorded points from board.json and telemetry, excluding driver 404. The highest for
each requested robot is currently in board.json. Bot means cover both alliances and all seeds.

| Team | Bot mean | Bot best | Leaderboard best | Mean / leaderboard | Leaderboard driver |
|---:|---:|---:|---:|---:|---|
| 254 | 584.0 | 683 | 1419 | 41.2% | next |
| 2910 | 465.8 | 551 | 1302 | 35.8% | EinSteiner3 |
| 1114 | 601.0 | 630 | 1257 | 47.8% | EinSteiner3 |
| 4414 | 421.3 | 517 | 1318 | 32.0% | TickleMonster |
| 60026 | 297.5 | 329 | 620 | 48.0% | EinSteiner3 |

## Biggest weaknesses visible in traces

1. **Vision recovery is incomplete.** 2910 accumulates 59.35 tag-blocked command-seconds
   across six matches. Red seed 1 has a stationary Score period from 23 to 33 s near
   (5.31, 2.96), carrying 48 FUEL with tag failures visible. Moving toward the range
   midpoint and facing the shooter does not reliably recover visibility; alternate
   shooting angles are a next improvement to test.
2. **Score completion can wait too long on the last few balls.** 4414 blue seeds 1/2 spend
   68.1/66.2 trace-sampled seconds in Score goals with at most five FUEL remaining.
   Blue seed 1 still has one ball and goal Score at 162.9 s. The sticky unload goal can
   therefore suppress a new collection cycle. Gate failures alone do not explain this:
   4414 has only 4.3 hood-blocked and 0.45 heading-blocked command-seconds across six runs.
   Progress-based recovery for residual/jammed balls is needed in addition to closed-gate recovery.
3. **Spin-up and late-window planning lose throughput.** Heading/flywheel settling costs
   1114 46.65/21.9 blocked-command seconds across six matches; 60026 loses 45.2 s to hood
   settling. T_score uses nominal travel and fire rate, not these delays or flight time.
   60026 blue seed 3 still carries 14 FUEL at 162.9 s (score then 199, final 217).
   More anticipatory positioning and shooting readiness would help.

## Artifacts and checks

- Per-run exact metrics and estimated blocked fire opportunities: `data/sim/imai/summary.json`.
- 30 viewer-format traces: `data/sim/imai/imai-{team}-{alliance}-{seed}.json`.
  Generated traces are gitignored; rerun evaluate to regenerate them.
- `python3 -m frsbot.tools.build_viewer` includes these runs, converting the engine score
  breakdown to a numeric viewer score. The rebuilt bundle has 76 total runs.
- `python3 -m pytest -q`: `346 passed in 15.72s` (NVM loaded), including direct arrival,
  routing around a hub without stopping at its intermediate corners, and HP-disabled scoring.
- Bundler check: 30 ImAI sim runs, 1661 samples each, numeric scores and hopper counts.
- Viewer compatibility is programmatically checked; no browser visual verification was performed.
