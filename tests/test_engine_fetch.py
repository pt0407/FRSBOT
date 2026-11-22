import json
import subprocess

import pytest

from frsbot.sim.engine import fetch


FACTORY = ('sim.worker.changed.ts");factory.f("static/chunks/turbopack-worker-new.js",'
           '["static/chunks/sim-new.js","static/chunks/runtime-new.js"])')


def test_discover_worker_without_fixed_hashes():
    assert fetch.discover_worker(["unrelated", FACTORY]) == (
        "/_next/static/chunks/turbopack-worker-new.js",
        ["/_next/static/chunks/runtime-new.js", "/_next/static/chunks/sim-new.js"])
    for sources in ([], ["unrelated"], [FACTORY.replace("sim.worker", "remote.worker")]):
        with pytest.raises(RuntimeError, match="could not discover"):
            fetch.discover_worker(sources)


def test_fetch_manifest_and_asset_cache(tmp_path, monkeypatch):
    wasm = '/_next/static/media/frs_match.changed.wasm'
    contents = {
        fetch.SITE: b'<script src="/_next/static/chunks/page.js"></script>',
        fetch.SITE + '_next/static/chunks/page.js': FACTORY.encode(),
        fetch.SITE + '_next/static/chunks/turbopack-worker-new.js': b'bootstrap',
        fetch.SITE + '_next/static/chunks/runtime-new.js': b'runtime',
        fetch.SITE + '_next/static/chunks/sim-new.js': (fetch.SIM_CONSTRUCTOR + f'"{wasm}"').encode(),
        fetch.SITE.rstrip('/') + wasm: b'\x00asm',
    }
    monkeypatch.setattr(fetch, "download", contents.__getitem__)
    destination = tmp_path / "engine"
    fetch.fetch(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["sim_chunk"] == "sim-new.js"
    assert len(manifest["assets"]) == 4
    assert (destination / "frs_match.changed.wasm").read_bytes() == b'\x00asm'
    assert not list(destination.glob("*.part"))


def test_download_reports_url_and_404(monkeypatch):
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/curl")
    monkeypatch.setattr(fetch.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(
        a, 22, b"", b"curl: (22) The requested URL returned error: 404"))
    with pytest.raises(RuntimeError, match=r"https://example.com/missing.js.*404"):
        fetch.download("https://example.com/missing.js")
