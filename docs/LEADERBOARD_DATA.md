# Public leaderboard replay inventory

Inspected every `rebuiltBoard` / `rankedBoard` occurrence in the downloaded `~/game/*.js`
bundle and traced the REBUILT `fetchBoards` adapter and generic leaderboard UI.

| Request | Supported parameters / interpretation |
|---|---|
| `{"t":"rebuiltBoard"}` | No parameters. Returns `entries`, `duos`, `trios`, 200 each. |
| `{"t":"rankedBoard","mode":"1v1"}` | Ranked rating table, 100 rows. |
| Same, `mode:"2v2"` | 100 rows. |
| Same, `mode:"3v3"` | 100 rows. |

Evidence: `1nq0fplgks03z.js` character offsets 17542/17637 send the bare solo request;
module 44468 exports local `t.filter(t=>t.team===e)` and crew team filtering/sorting with
`.slice(0,10)`. `2-q9_slh_091j.js` around offset 62915 implements `fetchBoards` by fetching
once and selecting `a.entries`, `a.duos`, `a.trios` locally. It supplies no variant options.
`3d6x1usqguumu.js` around offset 45079 sends `rankedBoard` with only `mode`.

**No per-robot network query, toggle/mode variants for solo, period/season selection,
pagination, limit or offset is exposed by this client.** Solo/2v0/3v0 and robot selectors
are local views of the same response. Non-default rules are marked off-board by the UI.
Ranked seasons are displayed in account history, not passed to a leaderboard request.
We did not invent unsupported fields or claim to enumerate a private server database.

## Reproduce

```sh
~/.nvm/versions/node/v24.22.2/bin/node frsbot/scrape/fetch_board.mjs data/board_queries.json --all
python3 -m frsbot.scrape.harvest data/board.json data/board_queries.json
```

The snapshot preserves all four exact queries, response timestamps and complete replies.
They were issued serially with 1 s gaps. Solo returned 46 unique shared replay IDs versus
45 in the original board. Duos/trios and all three ranked responses had **zero replay IDs**.
The union of original and refreshed snapshots contains **46 unique shared IDs**, all solo,
none belonging to `404`. This is the entire observed client-accessible listing, not a claim
about every replay stored on the server.

Harvesting deduplicates IDs across inputs and excludes an ID if **any** associated driver
is `404` (including crew members). Cached telemetry/raw files are reused; downloads are
serial with at least 1 s between starts. Raw blobs are cached under gitignored `data/raw/`.
Rate-limit/service failures stop the loop without automatic retry.

Result: **45 usable cached replays, zero new successful downloads, one HTTP 404**.
The new listing is team 9785 / AB_6328 / `3owlc02fjznvznc0`; its public archive GET returned
`HTTP Error 404: Not Found`. The board's shared flag does not guarantee archive retention.

| Team | Harvested |
|---:|---:|
| 254 | 6 |
| 971 | 2 |
| 1114 | 5 |
| 1323 | 0 |
| 1360 | 3 |
| 1678 | 2 |
| 1690 | 1 |
| 1706 | 2 |
| 1833 | 3 |
| 2056 | 2 |
| 2481 | 2 |
| 2910 | 4 |
| 4414 | 1 |
| 4946 | 3 |
| 5000 | 2 |
| 6329 | 0 |
| 7558 | 2 |
| 7769 | 0 |
| 9785 | 2 |
| 60026 | 3 |
