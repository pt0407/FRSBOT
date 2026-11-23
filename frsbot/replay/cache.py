"""Parallel replay decode cache (docs/PERF.md step 4): one uncompressed .npz per match under data/cache/.

Each .npz holds the converted GameState arrays: t (F,), robots (F, R, 6: x, z, a, vx, vz, hopper),
fuel_pos (F, N, 3) float32 metres, fuel_where (F, N) uint8, plus a JSON `meta` (ids, teams, alliances, capacities,
first_inactive, source size/mtime). Members are stored uncompressed, so `load()` memory-maps them in place.

    python3 -m frsbot.replay.cache                      # decode data/raw/*.rbrp into data/cache/ on every core
    python3 -m frsbot.replay.cache --benchmark          # serial decode vs parallel build vs cached load
"""
import argparse
import json
import os
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from frsbot.core.field import ALLIANCES, is_active, phase_at, seconds_to_toggle
from frsbot.core.state import FuelState, GameState, HubState, RobotState

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
VERSION = 1
ROBOT_FIELDS = ("x", "z", "a", "vx", "vz", "hopper")


def decode(path: Path) -> dict:
    """Parse and convert one replay with the reference decoder; returns the arrays stored in its cache file."""
    from frsbot.replay.convert import convert
    from frsbot.replay.rbrp import parse
    path = Path(path)
    rep = parse(path.read_bytes(), fuel=True)
    states = convert(rep)
    if not states:
        raise ValueError(f"{path.name} has no frames")
    metas = getattr(rep, "metas", [])
    first = next((m["firstInactive"] for _, m in metas if m.get("firstInactive") is not None), None)
    stat = path.stat()
    meta = dict(version=VERSION, source=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns, first_inactive=first,
                ids=[r.id for r in states[0].robots], teams=[r.team for r in states[0].robots],
                alliances=[r.alliance for r in states[0].robots], capacities=[r.capacity for r in states[0].robots])
    return dict(
        t=np.array([s.t for s in states], np.float64),
        robots=np.array([[[getattr(r, f) for f in ROBOT_FIELDS] for r in s.robots] for s in states], np.float64),
        fuel_pos=np.stack([s.fuel.pos for s in states]).astype(np.float32),
        fuel_where=np.stack([s.fuel.where for s in states]).astype(np.uint8),
        meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
    )


def cache_path(source: Path, cache_dir: Path = CACHE) -> Path:
    return Path(cache_dir) / f"{Path(source).stem}.npz"


def fresh(source: Path, cache_dir: Path = CACHE) -> bool:
    target = cache_path(source, cache_dir)
    if not target.is_file():
        return False
    try:
        meta = load(target, mmap=False)["meta"]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False
    stat = Path(source).stat()
    return meta.get("version") == VERSION and meta.get("size") == stat.st_size and meta.get("mtime_ns") == stat.st_mtime_ns


def _build_one(job) -> tuple[str, str | None]:
    source, cache_dir = job
    try:
        arrays = decode(source)
        target = cache_path(source, cache_dir)
        partial = target.with_name(f".{target.stem}.{os.getpid()}.npz")
        np.savez(partial, **arrays)
        partial.replace(target)
        return Path(source).name, None
    except Exception as error:  # noqa: BLE001 - one bad replay must not stop the batch
        return Path(source).name, f"{type(error).__name__}: {error}"


def build(sources, cache_dir: Path = CACHE, workers: int | None = None, force: bool = False) -> dict[str, str | None]:
    """Decode every stale source in parallel; returns {source name: error or None} for the ones it decoded."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(Path(s), cache_dir) for s in sources if force or not fresh(s, cache_dir)]
    if not jobs:
        return {}
    workers = min(workers or os.cpu_count() or 1, len(jobs))
    if workers == 1:
        return dict(map(_build_one, jobs))
    with Pool(workers) as pool:
        return dict(pool.imap_unordered(_build_one, jobs))


def load(path: Path, mmap: bool = True) -> dict:
    """Open a cache file; arrays are read-only memory maps into the .npz unless mmap=False."""
    path, out = Path(path), {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename.removesuffix(".npy")
            if not mmap or info.compress_type != zipfile.ZIP_STORED or name == "meta":
                with archive.open(info) as member:
                    out[name] = np.lib.format.read_array(member)
                continue
            with path.open("rb") as raw:
                raw.seek(info.header_offset)
                header = raw.read(30)
                start = info.header_offset + 30 + int.from_bytes(header[26:28], "little") + int.from_bytes(header[28:30], "little")
                raw.seek(start)
                version = np.lib.format.read_magic(raw)
                read_header = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}[version]
                shape, fortran, dtype = read_header(raw)
                offset = raw.tell()
            out[name] = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")
    out["meta"] = json.loads(bytes(out["meta"]).decode())
    return out


def states(cached: dict) -> list[GameState]:
    """Rebuild the GameState list that frsbot.replay.convert.convert returns for the same replay."""
    meta, first = cached["meta"], cached["meta"]["first_inactive"]
    out = []
    for k, t in enumerate(cached["t"].tolist()):
        robots = [RobotState(id=i, team=team, alliance=alliance, x=r[0], z=r[1], a=r[2], vx=r[3], vz=r[4],
                             hopper=int(r[5]), capacity=cap)
                  for i, team, alliance, cap, r in zip(meta["ids"], meta["teams"], meta["alliances"], meta["capacities"],
                                                       cached["robots"][k].tolist())]
        out.append(GameState(t=t, phase=phase_at(t),
                             hubs={a: HubState(is_active(a, t, first), seconds_to_toggle(a, t, first)) for a in ALLIANCES},
                             robots=robots, fuel=FuelState(np.asarray(cached["fuel_pos"][k]), np.asarray(cached["fuel_where"][k])),
                             me=0))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=RAW)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="time serial decode, parallel build and cached load")
    args = parser.parse_args()
    sources = sorted(args.src.glob("*.rbrp"))
    if not args.benchmark:
        results = build(sources, args.cache, args.workers, args.force)
        failed = {k: v for k, v in results.items() if v}
        print(f"decoded {len(results) - len(failed)} of {len(sources)} replays into {args.cache} ({len(sources) - len(results)} already fresh)")
        for name, error in sorted(failed.items()):
            print(f"failed {name}: {error}")
        return
    from frsbot.replay.convert import convert
    from frsbot.replay.rbrp import parse
    before = time.perf_counter()
    frames = sum(len(convert(parse(p.read_bytes(), fuel=True))) for p in sources)
    serial = time.perf_counter() - before
    before = time.perf_counter()
    errors = [e for e in build(sources, args.cache, args.workers, force=True).values() if e]
    parallel = time.perf_counter() - before
    before = time.perf_counter()
    for p in sources:
        cached = load(cache_path(p, args.cache))
        float(cached["fuel_pos"][-1].sum())
    loaded = time.perf_counter() - before
    print(f"replays={len(sources)} frames={frames} workers={args.workers} errors={len(errors)}")
    print("stage wall_s replays_per_min")
    for stage, wall in (("serial_decode", serial), ("parallel_build", parallel), ("cached_load_mmap", loaded)):
        print(f"{stage} {wall:.2f} {60 * len(sources) / wall:.0f}")


if __name__ == "__main__":
    main()
