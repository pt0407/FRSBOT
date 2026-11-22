"""Download shared leaderboard replays, decode to telemetry JSON, discard the raw blob.

Usage:
  python -m frsbot.scrape.harvest data/board.json
  (data/board.json comes from frsbot/scrape/fetch_board.mjs)
"""
import concurrent.futures as cf
import gzip
import json
import sys
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


def process(entry: dict) -> dict:
    rid = entry["replay"]
    out = TEL / f"{rid}.json"
    if out.exists():
        return {"replay": rid, "status": "cached"}
    try:
        blob = fetch_blob(rid)
        tel = telemetry(parse(blob))
        tel.update({"leaderboard": {k: entry[k] for k in ("name", "team", "points", "scored", "against", "at")}})
        out.write_text(json.dumps(tel, separators=(",", ":")))
        return {"replay": rid, "status": "ok", "samples": len(tel["samples"]), "driver": entry["name"]}
    except Exception as ex:  # keep going; report at the end
        return {"replay": rid, "status": "error", "error": str(ex)}


def main(board_path: str):
    board = json.loads(Path(board_path).read_text())
    shared = [e for e in board["entries"] if e.get("replay") and e.get("shared")]
    keep = [e for e in shared if e["name"] not in EXCLUDE_NAMES]
    print(f"shared={len(shared)} after_exclusion={len(keep)} excluded={len(shared) - len(keep)}")
    TEL.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(8) as ex:
        results = list(ex.map(process, keep))
    ok = [r for r in results if r["status"] in ("ok", "cached")]
    for r in results:
        if r["status"] == "error":
            print("ERR", r["replay"], r["error"])
    index = [r for r in results if r["status"] != "error"]
    (DATA / "harvest_index.json").write_text(json.dumps(index, indent=1))
    print(f"ok={len(ok)} errors={len(results) - len(ok)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DATA / "board.json"))
