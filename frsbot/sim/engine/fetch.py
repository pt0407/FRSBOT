"""Discover and cache the public game's worker assets; never vendor game code."""
import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urljoin, urlparse

SITE = "https://rebuilt.frsim.com/"
ENGINE_DIR = Path(__file__).resolve().parents[3] / "data" / "engine"
WORKER = re.compile(r'\.f\("(static/chunks/turbopack-worker-[^"]+\.js)",\s*(\[[^\]]+\])\)')
SIM_CONSTRUCTOR = "constructor(t,e,i){this.e=t,this.p=e,this.ids=i,"


def download(url: str) -> bytes:
    if not shutil.which("curl"):
        raise RuntimeError("curl is required to fetch the engine assets")
    result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                             "--max-time", "60", url], capture_output=True)
    if result.returncode:
        raise RuntimeError(f"engine download failed: {url}: {result.stderr.decode().strip()}")
    return result.stdout


def discover_worker(sources: list[str]) -> tuple[str, list[str]]:
    for source in sources:
        for match in WORKER.finditer(source):
            if "sim.worker." in source[max(0, match.start() - 300):match.start()]:
                # Turbopack's factory reverses this list for its worker bootstrap.
                return "/_next/" + match[1], ["/_next/" + p for p in reversed(json.loads(match[2]))]
    raise RuntimeError("could not discover sim.worker bootstrap/chunks; the site's bundling changed")


def fetch(destination: Path = ENGINE_DIR, site: str = SITE):
    html = download(site).decode()
    scripts = list(dict.fromkeys(re.findall(r'<script[^>]+src="([^"]+\.js(?:\?[^"]*)?)"', html)))
    if not scripts:
        raise RuntimeError("no client scripts found on the game page")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        sources = list(pool.map(lambda p: download(urljoin(site, p)).decode(), scripts))
    bootstrap, chunks = discover_worker(sources)
    urls = [urljoin(site, p) for p in [bootstrap, *chunks]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        assets = dict(zip(urls, pool.map(download, urls)))
    sim_chunks = [url for url, raw in assets.items() if SIM_CONSTRUCTOR.encode() in raw]
    if len(sim_chunks) != 1:
        raise RuntimeError("expected one WASM sim constructor; engine adapter needs updating")
    wasm_paths = list(dict.fromkeys(re.findall(r'"(/_next/static/media/frs_match[^"/]*\.wasm)"',
                                             assets[sim_chunks[0]].decode())))
    if not wasm_paths:
        raise RuntimeError("no frs_match WASM URLs found in the simulation chunk")
    for path in wasm_paths:
        url = urljoin(site, path)
        assets[url] = download(url)
    names = [Path(urlparse(url).path).name for url in assets]
    if len(set(names)) != len(names):
        raise RuntimeError("engine assets have colliding filenames")
    destination.mkdir(parents=True, exist_ok=True)
    for url, raw in assets.items():
        target = destination / Path(urlparse(url).path).name
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(raw)
        temporary.replace(target)
    manifest = {"site": site, "bootstrap": bootstrap, "chunks": chunks,
                "sim_chunk": Path(urlparse(sim_chunks[0]).path).name,
                "assets": {urlparse(url).path: Path(urlparse(url).path).name for url in assets}}
    temporary = destination / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "manifest.json")
    print(f"cached {len(assets)} engine assets in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ENGINE_DIR)
    args = parser.parse_args()
    fetch(args.destination)
