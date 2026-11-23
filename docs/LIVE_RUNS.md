# Solo bridge latency runs

Only Singleplayer, team 254, blue, native auto `none`, no climb. Separate random seeds;
these are functional checks, not controlled score comparisons. Software SwiftShader,
low graphics, 320x180 driving viewport. Raw JSONL, full summaries and screenshots are
ignored under `data/live/`.

## Python driver, auto compensation

Command: `/usr/bin/python3 -m frsbot.live.bridge --team 254 --driver python --latency auto --calibrate 6`

Run: `imai-254-blue-20261123T085339429216Z`, seed 2136950285.
Frame metadata and visually inspected results screen both show blue **589**, red **0**.
Screenshot: `data/live/imai-254-blue-20261123T085339429216Z-end.png`.

| Measurement | Median ms | p90 ms | Median 60-Hz tick equivalents | p90 ticks |
|---|---:|---:|---:|---:|
| Physical response (6 steps) | 70.25 | 108.95 | 4.215 | 6.537 |
| Write → poll (2224 sampled writes) | 18.10 | 32.20 | 1.086 | 1.932 |
| Poll → input | 0 | 0.10 | 0 | 0.006 |
| Input → first received frame | 35.40 | 37.60 | 2.124 | 2.256 |
| Write → first received frame | 41.60 | 66.50 | 2.496 | 3.990 |

Physical effect-frame tick deltas (not wall-time equivalents): median 5, p90 6.5.
4284 completed write traces, 2224 actually sampled (the others were superseded before
the next poll). Transport percentiles exclude superseded commands; all traces are logged.
The first received frame after sending is a transport boundary, **not** proof that the
worker applied that input. Only the step-response test detects physical motion.

Rates: frames 60.258/s, polls 17.125/s, sends 16.019/s, bot acts 25.601/s.
Final running compensation estimate: 39.35 ms.

The merged upstream `view_to_state` still accepts tapped views unchanged; the official
merged `driver.mjs` is byte-identical to the initially imported port.
Post-merge `/usr/bin/python3 -m pytest -q`: **423 passed, 5 skipped in 35.53s**.
