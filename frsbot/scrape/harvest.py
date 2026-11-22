"""Download shared leaderboard replays; optionally retain raw frames for imitation.

Usage:
  python -m frsbot.scrape.harvest data/board.json
  (data/board.json comes from frsbot/scrape/fetch_board.mjs)
"""
import argparse
import concurrent.futures as cf
import gzip
import json
import urllib.request
from pathlib import Path

from frsbot.replay.rbrp import parse, telemetry

REPLAY_HOST = "https://reefscape-ws.frsim.com/rebuilt-replays"
DATA = Path(__file__).resolve().parents[2] / "data"
TEL = DATA / "telemetry"
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
    raw_path = DATA / "raw" / f"{rid}.rbrp"
    if out.exists() and (not retain_raw or raw_path.exists()):
        return {"replay": rid, "status": "cached"}
    try:
        blob = raw_path.read_bytes() if retain_raw and raw_path.exists() else fetch_blob(rid)
        replay = parse(blob)
        if any(r["name"].strip() in EXCLUDE_NAMES for r in replay.robots):
            return {"replay": rid, "status": "excluded"}
        if retain_raw and not raw_path.exists():
            raw_path.write_bytes(blob)
        if not out.exists():
            tel = telemetry(replay)
            tel.update({"leaderboard": {k: entry[k] for k in ("name", "team", "points", "scored", "against", "at")}})
            out.write_text(json.dumps(tel, separators=(",", ":")))
        return {"replay": rid, "status": "ok", "samples": len(replay.frames), "driver": entry["name"]}
    except Exception as ex:  # keep going; report at the end
        return {"replay": rid, "status": "error", "error": str(ex)}


def main(board_path: str, retain_raw: bool = False):
    board = json.loads(Path(board_path).read_text())
    shared = [e for e in board["entries"] if e.get("replay") and e.get("shared")]
    keep = list({e["replay"]: e for e in shared if e["name"].strip() not in EXCLUDE_NAMES}.values())
    print(f"shared={len(shared)} after_exclusion={len(keep)} excluded={len(shared) - len(keep)}")
    TEL.mkdir(parents=True, exist_ok=True)
    if retain_raw:
        (DATA / "raw").mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda entry: process(entry, retain_raw), keep))
    ok = [r for r in results if r["status"] in ("ok", "cached")]
    for r in results:
        if r["status"] == "error":
            print("ERR", r["replay"], r["error"])
    index = ok
    (DATA / "harvest_index.json").write_text(json.dumps(index, indent=1))
    print(f"ok={len(ok)} errors={sum(r['status'] == 'error' for r in results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", nargs="?", default=str(DATA / "board.json"))
    parser.add_argument("--retain-raw", action="store_true")
    args = parser.parse_args()
    main(args.board, args.retain_raw)
