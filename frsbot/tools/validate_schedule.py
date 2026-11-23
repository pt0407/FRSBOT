"""Compare the match model with cached replay header metadata, without decoding frames."""
import json
import struct
from pathlib import Path

from frsbot.core.field import ALLIANCES, is_active, phase_at
from frsbot.replay.convert import auto_start_seconds
from frsbot.replay.rbrp import MAGIC, gunzip_if_needed

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def read_head(path: Path) -> dict:
    raw = gunzip_if_needed(path.read_bytes())
    if raw[:4] != MAGIC:
        raise ValueError("not an RBRP replay")
    size = struct.unpack_from("<I", raw, 5)[0]
    return json.loads(raw[9:9 + size])


def validate_metas(metas: list) -> list[dict]:
    origin = auto_start_seconds(metas)
    mismatches = []
    for index, meta in metas:
        t = index / 30 - origin
        if phase_at(t).value != meta["phase"]:
            mismatches.append({"frame": index, "t": t, "field": "phase", "observed": meta["phase"]})
        for alliance in ALLIANCES:
            observed = meta["hubs"][alliance]["active"]
            if is_active(alliance, t, meta.get("firstInactive")) != observed:
                mismatches.append({"frame": index, "t": t, "field": alliance, "observed": observed})
    return mismatches


def main():
    paths = sorted(RAW.glob("*.rbrp"))
    if not paths:
        raise FileNotFoundError(f"no cached replays in {RAW}")
    entries = countdowns = 0
    mismatches = []
    for path in paths:
        metas = read_head(path)["metas"]
        entries += len(metas)
        countdowns += auto_start_seconds(metas) > 0
        mismatches.extend({"run": path.stem, **m} for m in validate_metas(metas))
    print(f"runs={len(paths)} countdown_runs={countdowns} metadata_entries={entries} "
          f"hub_checks={entries * 2} phase_checks={entries} mismatches={len(mismatches)}")
    for mismatch in mismatches:
        print(json.dumps(mismatch))
    return mismatches


if __name__ == "__main__":
    if main():
        raise SystemExit(1)
