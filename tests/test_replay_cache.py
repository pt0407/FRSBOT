import gzip
import os
from pathlib import Path

import numpy as np

from frsbot.replay import cache
from frsbot.replay.convert import convert
from frsbot.replay.rbrp import parse

FIXTURE = Path(__file__).parent / "fixtures" / "multiplayer_excerpt.rbrp.gz"


def test_parallel_cache_round_trips_convert_and_memory_maps(tmp_path):
    raw = gzip.decompress(FIXTURE.read_bytes())
    sources = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.rbrp"
        path.write_bytes(raw)
        sources.append(path)
    (tmp_path / "bad.rbrp").write_bytes(b"not a replay")
    out = tmp_path / "cache"
    results = cache.build([*sources, tmp_path / "bad.rbrp"], out, workers=2)
    assert {k for k, v in results.items() if v is None} == {"a.rbrp", "b.rbrp", "c.rbrp"} and results["bad.rbrp"]
    assert cache.build(sources, out, workers=2) == {}
    expected = convert(parse(raw, fuel=True))
    loaded = cache.load(cache.cache_path(sources[0], out))
    assert isinstance(loaded["fuel_pos"], np.memmap) and not loaded["fuel_pos"].flags.writeable
    for a, b in zip(expected, cache.states(loaded)):
        assert (a.t, a.phase, a.hubs, a.robots, a.me) == (b.t, b.phase, b.hubs, b.robots, b.me)
        np.testing.assert_array_equal(a.fuel.pos, b.fuel.pos)
        np.testing.assert_array_equal(a.fuel.where, b.fuel.where)
    os.utime(sources[1], ns=(1, 1))
    assert not cache.fresh(sources[1], out) and cache.fresh(sources[0], out)
    assert set(cache.build(sources, out, workers=1)) == {"b.rbrp"}
