"""Passive four-watch REBUILT recorder with durable compact storage and upload outbox."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from frsbot.replay.convert import convert
from frsbot.replay.rbrp import parse
from frsbot.replay.spectator import capture_replays
from frsbot.scrape.mp_release import GitHub, atomic_write, flush_queue, json_lines
from frsbot.sim.wasm_env import find_node

ROOT = Path(__file__).resolve().parents[2]


def repair_tail(path):
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return
    with path.open("rb+") as stream:
        data = stream.read()
        if data.endswith(b"\n"):
            return
        last = data.rfind(b"\n") + 1
        try:
            json.loads(data[last:])
        except (ValueError, UnicodeDecodeError):
            stream.seek(last)
            stream.truncate()
        else:
            stream.seek(0, os.SEEK_END)
            stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_match(path):
    return convert(parse(Path(path).read_bytes(), fuel=True))


def compact_capture(path):
    path = Path(path).resolve()
    root = path.parent
    with (root / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        repair_tail(root / "manifest.jsonl")
        repair_tail(path)
        known = {r["id"]: r for r in json_lines(root / "manifest.jsonl")}
        rows = []
        for index, raw in enumerate(capture_replays(path)):
            rep = parse(raw, fuel=True)
            source = rep.meta["source"]
            key = source.get("match_key") or f"{source.get('code')}:{source.get('since', source.get('at', rep.frames[0][0]))}:{source.get('game', 1)}"
            rid = "ranked-" + hashlib.sha256(f"{key}:{index}".encode()).hexdigest()[:24]
            asset = f"{rid}.rbrp.gz"
            target = root / "matches" / asset
            if rid in known:
                if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != known[rid]["sha256"]:
                    raise ValueError("manifest references a missing or corrupt compact match; raw retained")
                rows.append(known[rid])
                continue
            # Validate the full FUEL stream before deleting the source capture.
            if not rep.frames or len(rep.fuel) != len(rep.frames):
                raise ValueError("incomplete FUEL stream")
            payload = gzip.compress(raw, compresslevel=6, mtime=0)
            atomic_write(target, payload)
            robots = [{k: r[k] for k in ("id", "team", "alliance")} for r in rep.robots]
            blue = sum(r["alliance"] == "blue" for r in robots)
            red = len(robots) - blue
            date = datetime.fromtimestamp(source.get("at", time.time() * 1000) / 1000, timezone.utc)
            row = {"id": rid, "match_key": key, "mode": source.get("mode") or f"{blue}v{red}",
                   "teams": [r["team"] for r in robots], "robots": robots, "score": rep.meta.get("score"),
                   "partial": rep.meta["partial"], "frames": len(rep.frames), "asset": asset,
                   "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                   "tag": "mp-data-" + date.strftime("%Y%m%d"), "game": "rebuilt"}
            atomic_write(root / "upload_queue" / f"{rid}.json", json.dumps(row).encode())
            with (root / "manifest.jsonl").open("a") as manifest:
                manifest.write(json.dumps(row, separators=(",", ":")) + "\n")
                manifest.flush()
                os.fsync(manifest.fileno())
            known[rid] = row
            rows.append(row)
            print(f"MATCH id={rid} mode={row['mode']} frames={row['frames']} partial={row['partial']} bytes={len(payload)} upload=queued", flush=True)
        if not rows:
            raise ValueError("no decodable match; raw retained")
        path.unlink()
        return rows


def recover(root):
    for path in sorted(Path(root).glob("ranked-*.jsonl")):
        try:
            compact_capture(path)
        except Exception as error:
            failed = Path(root) / "failed"
            failed.mkdir(exist_ok=True)
            path.replace(failed / path.name)
            print(f"RECOVERY retained={failed / path.name} error={error}", file=sys.stderr, flush=True)


def upload_loop(root, stop):
    client = GitHub()
    if not client.token:
        print("UPLOAD disabled: FRSBOT_GITHUB_TOKEN missing; compact matches remain queued locally", file=sys.stderr)
        return
    backoff = 5
    while not stop.is_set():
        try:
            flush_queue(root, client)
            backoff = 5
            stop.wait(10)
        except Exception as error:
            print(f"UPLOAD retry_in={backoff}s error={error}", file=sys.stderr, flush=True)
            stop.wait(backoff)
            backoff = min(300, backoff * 2)


def run(root, smoke_seconds=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".recorder.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another recorder is already using this directory") from None
        stop = threading.Event()
        child = None

        def shutdown(signum, frame):
            stop.set()
            if child and child.poll() is None:
                child.send_signal(signal.SIGINT)

        previous = {sig: signal.signal(sig, shutdown) for sig in (signal.SIGINT, signal.SIGTERM)}
        timer = threading.Timer(smoke_seconds, lambda: shutdown(None, None)) if smoke_seconds else None
        if timer:
            timer.start()
        uploader = threading.Thread(target=upload_loop, args=(root, stop), daemon=True)
        uploader.start()
        backoff = 5
        try:
            while not stop.is_set():
                recover(root)
                if stop.is_set():
                    break
                repair_tail(root / "manifest.jsonl")
                (root / "manifest.jsonl").touch(exist_ok=True)
                started = time.monotonic()
                child = subprocess.Popen([find_node(), str(ROOT / "frsbot/scrape/spectate.mjs"),
                                          "--minutes", "330", "--concurrency", "4", "--compact",
                                          "--seen-file", str(root / "manifest.jsonl"), "--output", str(root)],
                                         cwd=ROOT, start_new_session=True)
                while child.poll() is None:
                    if stop.wait(.25):
                        if child.poll() is None:
                            child.send_signal(signal.SIGINT)
                        try:
                            child.wait(timeout=300)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGTERM)
                            child.wait()
                        break
                recover(root)
                if stop.is_set():
                    break
                if time.monotonic() - started >= 300:
                    backoff = 5
                print(f"RECORDER restart exit={child.returncode} backoff={backoff}s", file=sys.stderr, flush=True)
                stop.wait(backoff)
                backoff = min(300, backoff * 2)
        finally:
            stop.set()
            if timer:
                timer.cancel()
            uploader.join(timeout=2)
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw_mp")
    parser.add_argument("--process", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-seconds", type=float)
    args = parser.parse_args()
    if args.smoke_seconds is not None and args.smoke_seconds <= 0:
        parser.error("smoke duration must be positive")
    if args.process:
        compact_capture(args.process)
    else:
        run(args.output, args.smoke_seconds)


if __name__ == "__main__":
    main()
