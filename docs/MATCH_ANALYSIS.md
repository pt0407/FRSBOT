# Observational multiplayer role and defense analysis

## Run it

```sh
python3 -m frsbot.analysis.matches data/raw_mp/ranked-FJ5S-1795384599586.0.rbrp --output data/raw_mp/FJ5S-analysis.json
```

Pass multiple RBRP paths to aggregate matches by **retrospective scoring rank within each
alliance**. The Python API is `analyze(states, events=..., metas=..., partial=...)`; the CLI
uses the existing decoder/converter. JSON includes robot specs, attribution, exclusive role
seconds, pairwise pressure/contact time, score timeline, hub windows, defense-start contexts,
response measurements, fouls/blockades and outcome. Raw captures and generated reports stay
in gitignored `data/raw_mp/`. This is analysis, not a trained strategy or a causal experiment.

## Passive batch recording

From the repository root, this records successive live ranked matches for **60 minutes**:

```sh
~/.nvm/versions/node/v24.22.2/bin/node frsbot/scrape/spectate.mjs --minutes 60 --concurrency 2 --decode --output data/raw_mp/batch
```

Without `--mode`, all 1v1/2v2/3v3 matches are eligible; add `--mode 3v3` (or `1v1`/`2v2`)
to restrict it. `--concurrency` accepts only 1 or 2. There is one discovery connection and
at most two watch connections. Discovery polls no faster than every 5 s while slots are
available; recently started games are preferred, and `(code, since, game)` is watched once.
Each watch is capped at 210 s and the remaining batch wall-clock budget. The deadline closes
active watches, retaining partial recordings; it does not wait for their matches to finish.
Already scheduled decoding/archive cleanup can finish after recording stops. A `batch-*.json`
manifest records each result, reason and mode. Empty lists keep waiting until the deadline.
Listing/transport/server errors stop the batch rather than reconnecting in a loop.

Only `rankedLive` and `rankedWatch` are sent, never participant join, queue, seat, input or
match-control commands. Announced public replays are fetched read-only; archive failure does
not discard the live recording. Batch decoding is asynchronous and serialized, so it does
not block another watch's snapshot stream. Ranked identities are anonymized: this data must
not be promoted to identity-filtered training data without a policy for excluding owner `404`.

The only live batch run performed for this change was **0.05 minutes (3 seconds)**. It watched
two real 1v1s concurrently, captured 149 frames/504 FUEL each, and decoded both as partial.
No long batch was run. Fake-transport tests cover successive mixed modes, deduplication,
concurrency, deadline cancellation, empty lists and errors.

## Definitions and attribution

- Times are normalized to auto start. Role/exposure integrals cover `[0,163)` (163 playing
  seconds), not countdown or settle. Last-frame duration is zero; partial clips retain only
  their observed exposure. No unobserved play is filled in.
- FUEL identity is stable. Intake events anchor possession; otherwise ownership is inferred
  from bumper-size-bounded nearest-robot geometry (15 cm separation from the next candidate).
  A carried identity remains assigned until release, unless a newer intake event overrides it.
  Initial carried FUEL is **not** collection. Collection is a transition into `where==2`.
- Shots follow carried `2` → airborne `1` → hub `3`, allowing floor bounces. The latest
  launch is retained for at most 8 s to entry, reset by another intake or human-player/off-field
  states. A score event must match the same FUEL and hub, within 5 s after entry (50 ms
  early tolerance for snapshot/event ordering). Hub processing delays are real: e.g. FUEL
  471 enters at 2.05 s but its score event is at 3.333 s. This ball is a human-player throw,
  correctly not assigned to a robot. Repeated hub entries cannot credit the same launch twice.
  Without score events, unique active hub entries are an explicitly approximate fallback.
- Opponent-hub scores are separate from own-alliance FUEL. Human-player/unresolved points
  remain unattributed rather than being assigned to the nearest robot.
- A pass-launch proxy is a release outside the robot's own alliance zone. The zone boundary
  is `8.270494 - 4.028694 = 4.2418 m`, with any bumper overlap qualifying. These constants are
  module 2960's half-field length and alliance-zone depth in `~/game/worker/2c3zgga2f0b2g.js`;
  the judge's `launched(...corners...)` / `P()` checks min/max bumper corners. Native `fed`
  tallies are also reported. A completed ally transfer requires another ally to acquire that
  FUEL within 10 s without an intervening hub entry/intake/reset. A miss subsequently recovered
  by an ally can qualify; neither metric proves an intentional pass.
- Exclusive role priority is **scorer > feeder > collector > defender > transit**. A release
  in the own zone marks scorer, outside marks feeder, and pickup marks collector, each for the
  following 1 s. Defender means no higher-priority activity and either opponent proximity,
  being within 0.7 m of a loaded opponent's active-hub shooting segment, or occupying the
  opponent zone. These are behavioral proxies, not labels of player intent. Simultaneous
  feeding/defending is hidden by the exclusive priority but remains in contact/pressure data.
- Being-defended/pressure means any opponent center within 1.5 m. **Contact is a separate
  proxy**: oriented bumper rectangles overlap with 4 cm tolerance. It is not a native contact
  event or proof the opponent intended defense; both pairwise and union times are retained.
- Rate comparisons restrict exposure to active own hub, loaded hopper, own zone and spec
  range envelope. Points are assigned to the launch's preceding observation, not the delayed
  score event. No gate/aim/speed equivalence is assumed. The hypothetical loss is
  `(undefended rate - defended rate) * defended exposure`; it is **not causal score loss**.
- Defense starts require at least 1 continuous second; pressure-response episodes at least
  0.5 s. Route turns compare 2 s before onset with up to 2 s during pressure, requiring 0.4 m
  displacement on each side. Shooting-spot shifts compare mean launch positions in the prior
  5 s with those during the episode. Missing paired shots produce null, not zero movement.

## FJ5S: complete real ranked 2v2

Source: `data/raw_mp/ranked-FJ5S-1795384599586.0.rbrp`, SHA256
`0135e8771a1f2474994eb4324f0ddfbe3bc15119b510c342074ddfbb52fe3afa`.
10,140 frames, four robots, 504 FUEL, -2.983 through 166 s, `partial=False`.
**Red won 1242–1108**. This is one match, not a representative multiplayer corpus.

### FUEL and roles

All four tracked own-alliance FUEL totals **exactly match the engine's autoFuel+teleFuel
tallies**. Possession geometry had zero ambiguous carried observations in this recording.
Each robot began with 8 FUEL, excluded from the collection column.

| Alliance / ID | Team | Own FUEL | Collected | Native fed | Pass proxy | Scorer s | Collector s | Feeder s | Defender s | Transit s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Blue / 25427 | 1690 | 582 | 1102 | 373 | 371 | 69.30 | 35.98 | 42.37 | 5.15 | 10.20 |
| Blue / 25553 | 2910 | 505 | 703 | 28 | 29 | 34.85 | 68.93 | 2.85 | 26.28 | 30.08 |
| Red / 24924 | 1690 | 728 | 1178 | 321 | 321 | 80.82 | 24.68 | 45.95 | 4.93 | 6.62 |
| Red / 24921 | 2910 | 468 | 769 | 187 | 187 | 32.25 | 62.30 | 10.68 | 37.58 | 20.18 |

The small pass-proxy/native differences are retained as measurement uncertainty; their exact
cause has not been localized and proxy launches are not replaced with native totals.
Completed ally transfers (outgoing, excluding hub recycling) were respectively **17, 11, 5,
58** in the row order above. Native `fed` is clearly not a count of delivered ally passes.
Native team 1690 specs: max speed 4.4 m/s,
fire rate 13/s, intake 40/s, total capacity 51. Team 2910: max speed 5.1 m/s, fire rate 32/s,
intake 50/s, total capacity 55; these are configured limits, not achieved rates.

There is **one blue 1690 opponent-hub point**: FUEL 168, released at 112.333 s from
`(7.372,1.711)`, enters the red hub at 114.100 s. Counting it as blue scoring would incorrectly
produce 583 instead of 582. Unattributed active points are blue 21/red 15, consistent with
human-player contributions. Full reconciliation: blue `582+505+21=1108`; red
`728+468+15+1 opponent-hub point+30 foul=1242`.

### Who experienced pressure, and the descriptive rate gap

| Robot | Near opponent s | Contact proxy s | Eligible defended s | Defended FUEL/s | Clear FUEL/s | Hypothetical loss FUEL |
|---|---:|---:|---:|---:|---:|---:|
| Blue 1690 | 29.92 | 8.80 | 2.27 | 4.41 | 8.73 | 9.8 |
| Blue 2910 | 45.93 | 17.53 | 9.62 | 6.97 | 13.43 | 62.2 |
| Red 1690 | 28.37 | 4.38 | 3.97 | 6.05 | 9.50 | 13.7 |
| Red 2910 | 44.37 | 21.62 | 10.72 | 7.74 | 12.71 | 53.3 |

The 2910 pair spent **33.02 s** within 1.5 m and **15.05 s** in bumper-contact proxy, more
than any other pair. That is mutual proximity, not proof of which player initiated defense.
Stronger scorers were not the most pressured by this metric. Rate gaps are confounded by
movement, shooting readiness, route, fuel availability and hub windows; small defended
exposures make causal or policy conclusions especially unsafe.

### Rank, defense switching and responses

Both 1690s were rank 1 on their alliance. Pooled role percentages:

| Retrospective rank | Scorer | Collector | Feeder | Defender | Transit |
|---|---:|---:|---:|---:|---:|
| 1 (1690) | 46.05% | 18.61% | 27.09% | 3.09% | 5.16% |
| 2 (2910) | 20.58% | 40.26% | 4.15% | 19.59% | 15.42% |

Rank-2 robots had 20 sustained defense starts: 12 while leading, 8 while trailing, 9 with
their own hub inactive; mean time left 84.03 s. Rank-1 robots had only two: one leading and
one trailing, both with own hub inactive. These are contexts at proxy-role onset, not an
estimated decision rule. Example: red 2910's 89.50 s onset lasted 12.90 s, with 73.50 s left,
red down 17 and red hub inactive. Blue 2910 began 4.45 s at 60.183 s, blue ahead 138 and
blue hub inactive. Thus neither "defend only while leading" nor "only while inactive" fits.

Observed route turns of at least 45 degrees: blue 1690 **8/15** measurable pressure episodes,
blue 2910 **15/17**, red 1690 **15/19**, red 2910 **15/20**. Passing occurred in 6/16 blue
1690 episodes, 0/32 blue 2910, 11/21 red 1690 and 1/26 red 2910. Paired shooting-spot data
were sparse: blue 1690 one measurable shift of 4.13 m; red 1690 two, median 2.52 m; neither
2910 had a qualifying paired sample. These are concomitant behavior, not demonstrated
responses caused by defense. Do not interpret missing shot pairs as no adaptation.

### Schedule and penalties

Both hubs active at 0–20 and 23–33 s; both inactive 20–23 and 163+ s. Red is first inactive:
blue active/red inactive 33–58 and 83–108; red active/blue inactive 58–83 and 108–133;
both active 133–163. JSON retains every observed hub transition and score metadata entry.

Last observed alliance totals at selected boundaries (including in-flight processing):

| Time s | Blue | Red |
|---:|---:|---:|
| 20 | 233 | 224 |
| 23 | 259 | 279 |
| 33 | 357 | 365 |
| 58 | 510 | 367 |
| 83 | 521 | 568 |
| 108 | 720 | 591 |
| 133 | 752 | 814 |
| 163 | 1056 | 1205 |
| 166 | 1108 | 1242 |

Final points can arrive during settle from previously accepted shots; the 163 s snapshot
is not the final result.

At 61.0 s, the native G419 event awards red 30 points, offender blue robot 25427, detail
`blocked an opponent, trapped with a partner`. Native blockade events are retained separately.
They explicitly name blue 25553 as the partner.
This confirms an actual trapping penalty, unlike our general proximity/contact proxies;
the foul event itself does not name the trapped opponent. It must not be used to label every
nearby episode intentional defense.

## Limits and checks

One complete 2v2 cannot establish general role assignments, defense effectiveness or causal
score loss. Retrospective scoring rank is for stratification, not an online feature. Anonymized
identities prevent guaranteed account exclusions; no model training occurs here. Larger passive
captures can populate all three modes, but partial-match and sampling biases must remain visible.

Tests include synthetic shot/transfer/recycle/own-goal cases, absent/invalid inputs, the checked-in
real multiplayer excerpt, Node transport batches, and exact FJ5S reconciliation when the full
gitignored recording is cached. See the task report for the final test output.
