"""Package passive spectator snapshots as RBRP, preserving absolute match time."""
import argparse
import base64
import json
import struct
from pathlib import Path

from frsbot.replay.convert import convert
from frsbot.replay.rbrp import parse


def capture_replays(path):
    """Yield one RBRP byte string per start message in a spectator JSONL capture."""
    record = None
    source = {}
    for line in Path(path).open():
        entry = json.loads(line)
        if entry["type"] == "capture":
            source = entry
        elif entry["type"] == "text":
            message = entry["message"]
            if message["t"] == "start":
                if record:
                    yield package(record)
                record = {"start": message["match"], "frames": [], "metas": [], "events": [], "source": source}
            elif record and message["t"] == "meta":
                record["metas"].append([len(record["frames"]), message["meta"]])
            elif record and message["t"] == "ev":
                record["events"].extend([len(record["frames"]), event] for event in message["events"])
        elif entry["type"] == "binary":
            if record is None:
                raise ValueError("snapshot received before match start")
            frame = base64.b64decode(entry["data"], validate=True)
            if len(frame) < 22 or frame[0] not in (2, 3, 4) or struct.unpack_from("<H", frame, 2)[0] != 3:
                raise ValueError("unsupported spectator snapshot wire format")
            record["frames"].append(frame)
    if record:
        yield package(record)


def package(record):
    frames = record["frames"]
    if not frames or frames[0][0] != 3:
        raise ValueError("capture requires an initial keyframe")
    first, last = frames[0], frames[-1]
    first_tick = struct.unpack_from("<i", first, 4)[0]
    first_elapsed = struct.unpack_from("<f", first, 16)[0]
    last_elapsed = struct.unpack_from("<f", last, 16)[0]
    countdown = bool(record["metas"] and record["metas"][0][1]["phase"] == "countdown")
    origin = (first_tick + round(struct.unpack_from("<f", first, 8)[0] * 60) if countdown
              else first_tick - round(first_elapsed * 60))
    source = record["source"]
    metadata = {"id": f"ranked-{source.get('code', 'unknown')}-{first_tick}", "kind": "room",
                "source": source, "autoStartTick": origin,
                "durationMs": round((struct.unpack_from("<i", last, 4)[0] - first_tick) / 60 * 1000),
                "partial": first_elapsed > 0 or last_elapsed < 166,
                "score": record["metas"][-1][1].get("score") if record["metas"] else None}
    head = json.dumps({"meta": metadata, "start": record["start"], "metas": record["metas"],
                       "events": record["events"]}, separators=(",", ":")).encode()
    return (b"RBRP\x02" + struct.pack("<I", len(head)) + head + struct.pack("<I", len(frames))
            + b"".join(struct.pack("<I", len(f)) + f for f in frames))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    found = False
    for index, raw in enumerate(capture_replays(args.capture)):
        found = True
        output = args.capture.with_suffix(f".{index}.rbrp")
        rep = parse(raw, fuel=True)
        states = convert(rep)
        output.write_bytes(raw)
        print(f"{output}: frames={len(states)} robots={len(states[0].robots)} "
              f"fuel={len(states[0].fuel.where)} t={states[0].t:.3f}..{states[-1].t:.3f} "
              f"partial={rep.meta['partial']}")
    if not found:
        raise ValueError("no matches in capture")


if __name__ == "__main__":
    main()
