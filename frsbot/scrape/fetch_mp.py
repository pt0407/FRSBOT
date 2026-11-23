"""Download compact REBUILT matches from all mp-data daily GitHub releases."""
import argparse
import hashlib
import json
from pathlib import Path

from frsbot.replay.rbrp import parse
from frsbot.scrape.mp_release import GitHub, atomic_write

ROOT = Path(__file__).resolve().parents[2]


def fetch_all(output, client=None, tag=None):
    output = Path(output)
    client = client or GitHub()
    count = 0
    for release in client.pages("/releases"):
        name = release["tag_name"]
        if tag is not None:
            if name != tag:
                continue
        elif name != "mp-data" and not name.startswith("mp-data-"):
            continue
        if Path(name).name != name:
            raise ValueError("unsafe multiplayer release tag")
        assets = {a["name"]: a for a in client.assets(release)}
        if "manifest.jsonl" not in assets:
            raise ValueError(f"{name}: no manifest; retry after uploader finishes")
        manifest = client.download(assets["manifest.jsonl"])
        rows = [json.loads(line) for line in manifest.decode().splitlines() if line]
        for row in rows:
            filename = row["asset"]
            if Path(filename).name != filename or not filename.endswith(".rbrp.gz"):
                raise ValueError("unsafe multiplayer asset name")
            target = output / name / filename
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]:
                continue
            data = client.download(assets[filename])
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"{filename}: hash mismatch")
            replay = parse(data, fuel=True)
            if len(replay.frames) != row["frames"]:
                raise ValueError(f"{filename}: frame count mismatch")
            atomic_write(target, data)
            count += 1
        atomic_write(output / name / "manifest.jsonl", manifest)
        print(f"FETCH tag={name} listed={len(rows)}")
    print(f"FETCH downloaded={count}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw_mp")
    parser.add_argument("--tag", help="restrict to one release (including mp-data-test)")
    args = parser.parse_args()
    fetch_all(args.output, tag=args.tag)
