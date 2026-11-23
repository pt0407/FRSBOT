# Passive REBUILT multiplayer capture

The public ranked spectator path works without signing in. No room join, queue, robot input,
or match-control request is sent. The recorder permits only `rankedLive` and `rankedWatch`.
Single mode watches one match at a time; fresh-match discovery requests the list at most once every 5 s.

## Continuous local recorder and GitHub submission

From the repository root, with `FRSBOT_GITHUB_TOKEN` already exported (a token with repository
contents/release write access to `pt0407/FRSBOT`), the single launch command is:

```sh
python3 -m frsbot.scrape.record_forever
```

This runs four passive ranked watches across 1v1/2v2/3v3 on the REBUILT endpoint only.
It finds Node on PATH or under `~/.nvm`, needs the project's Python dependencies, and runs locally,
not in GitHub Actions. Leave it in a persistent terminal or your process supervisor.
Listing polls are five seconds apart. Disconnected watches are saved as partial matches and their
slots reused. A listing disconnect, watch error or Node crash restarts the worker with exponential
backoff (5–300 s); healthy workers rotate every 5.5 h. Only one recorder can own an output directory.

Each finished match prints one `MATCH ...` status line. The recorder copies original wire frames
to one gzip-compressed RBRP file under `data/raw_mp/matches/`, validates the FUEL stream, writes an
upload outbox entry and fsyncs a local `manifest.jsonl` before removing its raw JSONL. The manifest
contains id, mode, team numbers, robot IDs/alliances, last observed score, partial flag and frame count,
plus asset/hash/release metadata. Match identity is a hash of the server's room code, room birth time
and game number, so restarts do not duplicate already saved matches. The first saved capture wins;
a partial match is not rewatched to manufacture a complete one. Older captures without server
identity fields use their capture timestamp as a fallback.

Interrupted raw files are recovered on startup/worker exit; an incomplete final JSON line is trimmed.
Undecodable captures are retained in `failed/` for inspection, not silently deleted. Ctrl-C/SIGTERM
stops discovery and active watches, finishes local conversion, and leaves any pending uploads on
disk. Compact files are retained even after upload. If stopping forces worker termination after
a five-minute grace period, its raw captures are recovered as well.

The upload worker reads **only `FRSBOT_GITHUB_TOKEN`**, creates daily UTC releases tagged
`mp-data-YYYYMMDD` if absent, uploads one `.rbrp.gz` asset per match and replaces that release's
`manifest.jsonl` asset. A durable `upload_queue/` holds unacknowledged entries, with retry backoff;
requests are at least one second apart. A local `upload_manifests/` cache allows interrupted manifest
replacement to recover without losing earlier entries. There is one uploader per local directory;
do not point independent machines at the same release concurrently (GitHub asset replacement has
no compare-and-swap). Release limits/API errors leave matches queued instead of dropping them.
Without a token, capture continues offline and warns once; restart with the token to drain the queue.
All runtime files remain under gitignored `data/raw_mp/` (or `--output DIR`).

Download **all `mp-data` / `mp-data-*` releases** into per-release directories, with SHA256 and frame
count validation and resumable local reuse:

```sh
python3 -m frsbot.scrape.fetch_mp
```

The same token variable is optional for public downloads. Load either local or downloaded files
directly as multi-robot `GameState` sequences:

```python
from frsbot.scrape.record_forever import load_match
states = load_match("data/raw_mp/matches/ranked-EXAMPLE.rbrp.gz")
```

For a short live smoke test (stops cleanly via the same SIGINT shutdown path):

```sh
python3 -m frsbot.scrape.record_forever --smoke-seconds 5 --output data/raw_mp/forever-smoke
```

The real upload verifier needs a queued compact match and `FRSBOT_GITHUB_TOKEN` exported from the
personal `GITHUB_TOKENCLASSIC` secret. It uploads/downloads the asset, verifies its hash and manifest,
then deletes the throwaway `mp-data-test` release and tag. It refuses to overwrite an existing test
release and keeps the real daily-release queue entry:

```sh
FRSBOT_GITHUB_TOKEN="$GITHUB_TOKENCLASSIC" python3 -m frsbot.scrape.mp_release --test-upload data/raw_mp/forever-smoke
```

For successive captures with at most two watches, use:

```sh
~/.nvm/versions/node/v24.22.2/bin/node frsbot/scrape/spectate.mjs --minutes 60 --concurrency 2 --decode --output data/raw_mp/batch
```

This includes 1v1/2v2/3v3 by default; `--mode` restricts the mode. See
[MATCH_ANALYSIS.md](MATCH_ANALYSIS.md) for deadline behavior, manifests, methodology and FJ5S findings.

## Record and decode

From the repository root:

```sh
~/.nvm/versions/node/v24.22.2/bin/node frsbot/scrape/spectate.mjs --fresh --mode 2v2 --seconds 420 --decode
```

This waits for a newly listed 2v2, records it, and invokes the Python decoder. The deadline includes
both discovery and recording; no match in the allotted time is an explicit error, not a fabricated sample.
The JSONL source and `.0.rbrp` output are under gitignored `data/raw_mp/`.

Other read-only operations:

```sh
~/.nvm/versions/node/v24.22.2/bin/node frsbot/scrape/spectate.mjs --list
~/.nvm/versions/node/v24.22.2/bin/node frsbot/scrape/spectate.mjs --code CODE --seconds 210 --decode
~/.nvm/versions/node/v24.22.2/bin/node frsbot/scrape/spectate.mjs --replay PUBLIC_REPLAY_ID
python3 -m frsbot.replay.spectator data/raw_mp/CAPTURE.jsonl
```

`--mode` also accepts 1v1/3v3. Without it, the recorder prefers 3v3, then 2v2, then 1v1.
There is a 128 MiB received-binary cap and an explicit time cap. In these bounded commands there are no automatic reconnects
or archive-download retries; HTTP errors and server errors are surfaced.

## Protocol and fidelity

- WebSocket: `wss://reefscape-ws.frsim.com/rebuilt-ws`.
- `rankedLive` replies with `matches` including a public `code`, mode, rating and `since`.
- `rankedWatch` replies with spectator `joined`/`room`, then `start.match`, `meta`, `ev`, binary
  snapshots and `over`. The word `joined` in the reply does not mean a competitor was added.
- `start.match.robots` supplies all IDs, teams, alliances and state-block sizes. Ranked names are
  anonymized (`Blue 1`, etc.); the recorder does not try to identify players.
- Snapshots are the existing protocol-v3 type 3 keyframes / type 4 deltas. Positions, quaternion,
  velocities, hopper counts and all 504 FUEL position/where values use the existing RBRP codecs.
  The observed live stream is 60 Hz, not the 30 Hz assumed by historical solo metadata timing.
- The wrapper copies snapshot bytes unchanged into an RBRP v2 container, preserving events and
  metadata. `autoStartTick` is derived from the snapshot tick and elapsed time, or countdown clock
  while elapsed is still zero. Conversion therefore handles mid-match attachment and countdowns
  without inventing missing auto frames or assuming a 30 Hz metadata index.
- `meta` includes alliance score breakdowns, hub activity, `firstInactive`, and per-robot `drivers`
  counters. These are retained for subsequent attribution/analysis checks.

The first observed `over.replay` ID (`gzogppgqjlnfmzke`) returned HTTP 404 at the public replay URL.
The tool reports this and retains the usable raw spectator data. A shared archive is not required.
The completed event API match inspected exposed scores/breakdowns, not a replay ID; event watching
has not yet been implemented or tested.

## Recorded sample

First capture: ranked **KJLR**, 2v2, last 64.433 s through settle:

```text
frames=3867 robots=4 fuel=504 t=101.567..166.000 partial=True
blue: 24924/4414, 24921/1323; red: 24745/1690, 23228/4414
final alliance total: blue 1452, red 1228
```

This is explicitly a partial recording, not a complete match. The first three real snapshots are
stored in `tests/fixtures/multiplayer_excerpt.rbrp.gz` with source metadata. Tests check all four
identities, hopper counts, absolute time, hub activity and 504 FUEL; they also exercise countdown
normalization, empty/invalid captures and a fake transport that rejects any non-passive request.

Fresh ranked 2v2 **FJ5S** was recorded from countdown through done and decoded successfully:

```text
data/raw_mp/ranked-FJ5S-1795384599586.0.rbrp: frames=10140 robots=4 fuel=504 t=-2.983..166.000 partial=False
blue: 25427/1690, 25553/2910; red: 24924/1690, 24921/2910
final alliance total: blue 1108, red 1242
```

Re-run the recorded source conversion with:

```sh
python3 -m frsbot.replay.spectator data/raw_mp/ranked-FJ5S-1795384599586.jsonl
```

The decoded full-replay SHA256 is
`0135e8771a1f2474994eb4324f0ddfbe3bc15119b510c342074ddfbb52fe3afa`.
Its announced archive ID `4qvsw4rf4mvhqal0` also returned HTTP 404; the live recording is independent
of archive availability. Two matches are now recorded: one complete and one partial.

Checks after the final code changes:

```text
366 passed in 13.02s
```

Role/defense analysis is now available; see [MATCH_ANALYSIS.md](MATCH_ANALYSIS.md). No training
has been implemented. Since ranked names
are anonymized, account-level identity exclusions cannot be guaranteed from these captures alone;
do not promote them to identity-filtered training data without a provenance policy.
