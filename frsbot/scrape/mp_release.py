"""Daily REBUILT replay releases and a durable, retryable upload outbox."""
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "pt0407/FRSBOT"


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def json_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("refusing insecure GitHub download redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


class GitHub:
    def __init__(self, token=None, interval=1):
        self.token = token if token is not None else os.environ.get("FRSBOT_GITHUB_TOKEN")
        self.base = f"https://api.github.com/repos/{REPO}"
        self.interval = interval
        self.next_request = 0
        self.opener = urllib.request.build_opener(SafeRedirect())

    def request(self, url, method="GET", data=None, content_type="application/json", binary=False):
        if url.startswith("/"):
            url = self.base + url
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in ("api.github.com", "uploads.github.com"):
            raise ValueError("refusing GitHub credentials outside GitHub API hosts")
        time.sleep(max(0, self.next_request - time.monotonic()))
        self.next_request = time.monotonic() + self.interval
        headers = {"Accept": "application/octet-stream" if binary else "application/vnd.github+json",
                   "User-Agent": "FRSBOT-passive-recorder", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if data is not None:
            headers["Content-Type"] = content_type
            if not isinstance(data, bytes):
                data = json.dumps(data).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with self.opener.open(req, timeout=90) as response:
            payload = response.read()
        return payload if binary else json.loads(payload) if payload else None

    def pages(self, path):
        page = 1
        while True:
            rows = self.request(f"{path}?per_page=100&page={page}")
            yield from rows
            if len(rows) < 100:
                return
            page += 1

    def release(self, tag):
        try:
            return self.request("/releases/tags/" + urllib.parse.quote(tag, safe=""))
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            return self.request("/releases", "POST", {"tag_name": tag, "name": tag,
                                "body": "Passive REBUILT multiplayer recordings. Ranked identities are anonymized."})

    def assets(self, release):
        return list(self.pages(f"/releases/{release['id']}/assets"))

    def download(self, asset):
        return self.request(asset["url"], binary=True)

    def upload(self, release, name, payload):
        url = release["upload_url"].split("{")[0] + "?" + urllib.parse.urlencode({"name": name})
        return self.request(url, "POST", payload, "application/octet-stream")

    def delete_asset(self, asset):
        self.request(asset["url"], "DELETE")


def publish_one(root, pending, client, tag_override=None):
    root, pending = Path(root), Path(pending)
    row = json.loads(pending.read_text())
    tag = tag_override or row["tag"]
    if not (tag == "mp-data-test" or len(tag) == 16 and tag.startswith("mp-data-") and tag[8:].isdigit()):
        raise ValueError("invalid multiplayer release tag")
    payload = (root / "matches" / row["asset"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError("queued replay hash mismatch")
    release = client.release(tag)
    assets = {a["name"]: a for a in client.assets(release)}
    existing = assets.get(row["asset"])
    if existing:
        digest = existing.get("digest")
        if digest != "sha256:" + row["sha256"]:
            if hashlib.sha256(client.download(existing)).hexdigest() != row["sha256"]:
                raise ValueError("remote match id already has different content")
    else:
        client.upload(release, row["asset"], payload)
    cache = root / "upload_manifests" / f"{tag}.jsonl"
    merged = {r["id"]: r for r in json_lines(cache)}
    manifest = assets.get("manifest.jsonl")
    if manifest:
        merged.update({r["id"]: r for r in map(json.loads, client.download(manifest).decode().splitlines()) if r})
    merged[row["id"]] = row
    data = b"".join((json.dumps(r, separators=(",", ":")) + "\n").encode() for r in merged.values())
    atomic_write(cache, data)
    if manifest:
        client.delete_asset(manifest)
    client.upload(release, "manifest.jsonl", data)
    pending.unlink()
    return tag


def flush_queue(root, client):
    if not client.token:
        return 0
    count = 0
    for pending in sorted((Path(root) / "upload_queue").glob("*.json")):
        publish_one(root, pending, client)
        count += 1
    return count


def test_upload(root):
    """Exercise a real throwaway release; never deletes a pre-existing release."""
    root = Path(root)
    client = GitHub()
    if not client.token:
        raise RuntimeError("FRSBOT_GITHUB_TOKEN is required for the real upload test")
    tag = "mp-data-test"
    for endpoint in ("/releases/tags/", "/git/ref/tags/"):
        try:
            client.request(endpoint + tag)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        else:
            raise RuntimeError("mp-data-test release/tag already exists; refusing to modify it")
    pending = next(iter(sorted((root / "upload_queue").glob("*.json"))), None)
    if pending is None:
        raise ValueError("upload test needs a compact match in upload_queue")
    row = json.loads(pending.read_text())
    (root / "upload_manifests" / f"{tag}.jsonl").unlink(missing_ok=True)
    release = client.release(tag)
    try:
        # Keep the production outbox entry for its real daily release.
        test_pending = root / "test-pending.json"
        atomic_write(test_pending, pending.read_bytes())
        publish_one(root, test_pending, client, tag_override=tag)
        assets = {a["name"]: a for a in client.assets(release)}
        downloaded = client.download(assets[row["asset"]])
        assert hashlib.sha256(downloaded).hexdigest() == row["sha256"]
        manifest = [json.loads(line) for line in client.download(assets["manifest.jsonl"]).decode().splitlines()]
        assert any(r["id"] == row["id"] for r in manifest)
        print(f"UPLOAD_TEST release={release['id']} asset={assets[row['asset']]['id']} sha256={row['sha256']} manifest=verified")
    finally:
        client.request(f"/releases/{release['id']}", "DELETE")
        try:
            client.request("/git/refs/tags/" + tag, "DELETE")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        print(f"UPLOAD_TEST deleted={tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-upload", type=Path, required=True, metavar="RECORDING_DIRECTORY")
    test_upload(parser.parse_args().test_upload)
