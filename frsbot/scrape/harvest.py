"""Download shared leaderboard replays, decode to telemetry JSON, and retain the raw blob.

Usage:
  python -m frsbot.scrape.harvest data/board.json
  (data/board.json comes from frsbot/scrape/fetch_board.mjs)
"""
import argparse
import gzip
import json
import re
import time
from collections import Counter
import urllib.request
from pathlib import Path

from frsbot.replay.rbrp import parse, telemetry

REPLAY_HOST = "https://reefscape-ws.frsim.com/rebuilt-replays"
DATA = Path(__file__).resolve().parents[2] / "data"
TEL = DATA / "telemetry"
RAW = DATA / "raw"
EXCLUDE_NAMES = {"404"}  # the user's own driver name; their runs are never used


def fetch_blob(rid: str) -> bytes:
    req = urllib.request.Request(f"{REPLAY_HOST}/{rid}", headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
    return raw


def process(entry: dict, retain_raw: bool = False) -> dict:
    rid = entry["replay"]
    out = TEL / f"{rid}.json"
    raw = RAW / f"{rid}.rbrp"
    if out.exists() and (not retain_raw or raw.exists()):
        return {"replay": rid, "status": "cached"}
    try:
        blob = raw.read_bytes() if raw.exists() else fetch_blob(rid)
        replay = parse(blob)
        if any(str(robot.get("name", "")).strip() in EXCLUDE_NAMES for robot in replay.robots):
            return {"replay": rid, "status": "excluded"}
        tel = telemetry(replay)
        tel.update({"leaderboard": {k: entry[k] for k in ("name", "team", "teams", "names", "points", "scored", "against", "at") if k in entry}})
        RAW.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(blob)
        out.write_text(json.dumps(tel, separators=(",", ":")))
        return {"replay": rid, "status": "ok", "samples": len(tel["samples"])}
    except Exception as ex:  # keep going; report at the end
        return {"replay": rid, "status": "error", "error": str(ex)}


def shared_entries(boards):
    candidates = []
    for board in boards:
        if "responses" in board:
            candidates.extend(shared_entries([r["data"] for r in board["responses"]]))
        else:
            for key in ("entries", "duos", "trios", "rows"):
                candidates.extend(e for e in board.get(key, []) if e.get("shared") and e.get("replay"))
    return candidates


def select_entries(boards):
    candidates = shared_entries(boards)
    excluded = {e["replay"] for e in candidates if any(str(n).strip().casefold() in EXCLUDE_NAMES
                for n in e.get("names", [e.get("name", "")]))}
    unique = {}
    for entry in candidates:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", entry["replay"]):
            raise ValueError("invalid replay id")
        if entry["replay"] not in excluded:
            unique[entry["replay"]] = entry
    return list(unique.values()), excluded


def main(board_paths, interval=1, retain_raw=False):
    if interval < 1:
        raise ValueError("request interval must be at least 1 second")
    boards = [json.loads(Path(p).read_text()) for p in board_paths]
    keep, excluded = select_entries(boards)
    shared = {e["replay"] for e in shared_entries(boards)}
    print(f"shared_unique={len(shared)} after_exclusion={len(keep)} excluded={len(excluded)}")
    TEL.mkdir(parents=True, exist_ok=True)
    results = []
    next_request = 0
    for entry in keep:
        if (retain_raw or not (TEL / f"{entry['replay']}.json").exists()) and not (RAW / f"{entry['replay']}.rbrp").exists():
            time.sleep(max(0, next_request - time.monotonic()))
            next_request = time.monotonic() + interval
        result = {**process(entry, retain_raw), "teams": entry.get("teams", [entry.get("team")])}
        results.append(result)
        if result["status"] == "error" and any(s in result["error"] for s in ("429", "503", "Connection refused")):
            break
    ok = [r for r in results if r["status"] in ("ok", "cached")]
    for r in results:
        if r["status"] == "error":
            print("ERR", r["replay"], r["error"])
    index = [r for r in results if r["status"] != "error"]
    (DATA / "harvest_index.json").write_text(json.dumps(index, indent=1))
    print(f"ok={len(ok)} errors={len(results) - len(ok)}")
    counts = Counter(team for result in ok for team in set(result["teams"]))
    for team, count in sorted(counts.items()):
        print(f"team={team} harvested={count}")
    if len(results) != len(keep):
        raise RuntimeError("harvest stopped on rate-limit/service failure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("boards", nargs="*", default=[str(DATA / "board.json")])
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--retain-raw", action="store_true", help="backfill raw files for cached telemetry")
    args = parser.parse_args()
    main(args.boards, args.interval, args.retain_raw)
