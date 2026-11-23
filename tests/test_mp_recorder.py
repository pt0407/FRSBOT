import base64
import hashlib
import json
import urllib.request
from types import SimpleNamespace

import pytest

from frsbot.scrape import record_forever
from frsbot.scrape.fetch_mp import fetch_all
from frsbot.scrape.mp_release import GitHub, SafeRedirect, flush_queue, json_lines
from frsbot.scrape.record_forever import compact_capture, load_match, recover, repair_tail
from test_spectator import excerpt


def capture(path, key="KJLR:123:1"):
    record = excerpt()
    source = {**record["source"], "match_key": key, "mode": "2v2"}
    lines = [source, {"type": "text", "message": {"t": "start", "match": record["start"]}}]
    lines.extend({"type": "text", "message": {"t": "meta", "meta": meta}} for _, meta in record["metas"])
    lines.extend({"type": "binary", "data": base64.b64encode(frame).decode()} for frame in record["frames"])
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


class FakeGitHub:
    token = "test-token"

    def __init__(self):
        self.releases = {}
        self.payloads = {}
        self.uploads = []
        self.fail_manifest = False

    def release(self, tag):
        if tag not in self.releases:
            self.releases[tag] = {"id": len(self.releases) + 1, "tag_name": tag}
            self.payloads[tag] = {}
        return self.releases[tag]

    def pages(self, path):
        assert path == "/releases"
        return iter(self.releases.values())

    def assets(self, release):
        return [{"name": name, "url": (release["tag_name"], name),
                 "digest": "sha256:" + hashlib.sha256(data).hexdigest()}
                for name, data in self.payloads[release["tag_name"]].items()]

    def upload(self, release, name, payload):
        if name == "manifest.jsonl" and self.fail_manifest:
            self.fail_manifest = False
            raise OSError("offline during manifest upload")
        data = self.payloads[release["tag_name"]]
        assert name not in data, "GitHub rejects duplicate asset names"
        data[name] = payload
        self.uploads.append(name)

    def delete_asset(self, asset):
        tag, name = asset["url"]
        del self.payloads[tag][name]

    def download(self, asset):
        tag, name = asset["url"]
        return self.payloads[tag][name]


def test_compact_real_sample_manifest_and_match_dedupe(tmp_path):
    raw = capture(tmp_path / "ranked-KJLR.jsonl")
    original_size = raw.stat().st_size
    row, = compact_capture(raw)
    assert not raw.exists()
    target = tmp_path / "matches" / row["asset"]
    assert row["bytes"] == target.stat().st_size < original_size
    states = load_match(target)
    assert len(states) == row["frames"] == 3
    assert len(states[0].robots) == len(row["robots"]) == 4
    assert states[0].fuel.pos.shape == (504, 3)
    assert row["partial"] and row["score"] and row["mode"] == "2v2"
    assert row["teams"] == [4414, 1323, 1690, 4414]
    assert len(list((tmp_path / "upload_queue").glob("*.json"))) == 1
    assert compact_capture(capture(raw)) == [row]
    assert json_lines(tmp_path / "manifest.jsonl") == [row]


def test_empty_invalid_and_crash_interrupted_capture(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    with pytest.raises(ValueError, match="no decodable"):
        compact_capture(empty)
    assert empty.exists()
    valid = capture(tmp_path / "ranked-partial.jsonl")
    with valid.open("ab") as stream:
        stream.write(b'{"type":"binary"')
    assert compact_capture(valid)[0]["frames"] == 3
    bad = tmp_path / "ranked-invalid.jsonl"
    bad.write_text("not json\n")
    recover(tmp_path)
    assert (tmp_path / "failed" / bad.name).read_text() == "not json\n"
    manifest = tmp_path / "manifest.jsonl"
    rows = json_lines(manifest)
    with manifest.open("ab") as stream:
        stream.write(b'{"id":')
    repair_tail(manifest)
    assert json_lines(manifest) == rows
    target = tmp_path / "matches" / rows[0]["asset"]
    target.write_bytes(b"damaged")
    raw = capture(tmp_path / "ranked-again.jsonl")
    with pytest.raises(ValueError, match="corrupt"):
        compact_capture(raw)
    assert raw.exists()


def test_offline_retry_preserves_daily_manifest_and_does_not_duplicate_assets(tmp_path):
    client = FakeGitHub()
    first, = compact_capture(capture(tmp_path / "ranked-1.jsonl"))
    assert flush_queue(tmp_path, client) == 1
    second, = compact_capture(capture(tmp_path / "ranked-2.jsonl", "ABCD:123:1"))
    client.fail_manifest = True
    with pytest.raises(OSError, match="offline"):
        flush_queue(tmp_path, client)
    assert (tmp_path / "upload_queue" / f"{second['id']}.json").exists()
    assert "manifest.jsonl" not in client.payloads[first["tag"]]
    assert flush_queue(tmp_path, client) == 1
    remote = [json.loads(line) for line in client.payloads[first["tag"]]["manifest.jsonl"].decode().splitlines()]
    assert {r["id"] for r in remote} == {first["id"], second["id"]}
    assert client.uploads.count(second["asset"]) == 1
    assert flush_queue(tmp_path, client) == 0
    third, = compact_capture(capture(tmp_path / "ranked-3.jsonl", "ABCD:123:2"))
    client.token = ""
    assert flush_queue(tmp_path, client) == 0
    assert (tmp_path / "upload_queue" / f"{third['id']}.json").exists()


def test_download_all_releases_and_hash_checked_resume(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    client = FakeGitHub()
    first, = compact_capture(capture(source / "ranked-1.jsonl"))
    assert flush_queue(source, client) == 1
    # Legacy mp-data releases are included too; unrelated releases are not.
    legacy = client.release("mp-data")
    for name, data in list(client.payloads[first["tag"]].items()):
        client.upload(legacy, name, data)
    client.release("unrelated")
    output = tmp_path / "downloaded"
    assert fetch_all(output, client) == 2
    assert fetch_all(output, client) == 0
    assert len(load_match(output / first["tag"] / first["asset"])) == 3
    client.payloads[first["tag"]][first["asset"]] = b"corrupt"
    with pytest.raises(ValueError, match="hash mismatch"):
        fetch_all(tmp_path / "bad", client)
    row = {**first, "asset": "../escape.rbrp.gz"}
    client.payloads[first["tag"]]["manifest.jsonl"] = json.dumps(row).encode()
    with pytest.raises(ValueError, match="unsafe"):
        fetch_all(tmp_path / "unsafe", client)
    assert fetch_all(tmp_path / "empty", FakeGitHub()) == 0


def test_api_pagination_and_credential_host_guard():
    client = GitHub(token="not-a-real-token", interval=0)
    with pytest.raises(ValueError, match="outside"):
        client.request("https://example.com/secret")
    calls = []

    def request(path):
        calls.append(path)
        return list(range(100)) if "page=1" == path.split("&")[-1] else [100]

    client.request = request
    assert len(list(client.pages("/releases"))) == 101
    assert len(calls) == 2
    request = urllib.request.Request("https://api.github.com/assets/1", headers={"Authorization": "Bearer test"})
    redirected = SafeRedirect().redirect_request(request, None, 302, "Found", {}, "https://objects.githubusercontent.com/asset")
    assert redirected.get_header("Authorization") is None
    with pytest.raises(ValueError, match="insecure"):
        SafeRedirect().redirect_request(request, None, 302, "Found", {}, "http://api.github.com/asset")


def test_supervisor_restarts_with_backoff_and_four_passive_watchers(tmp_path, monkeypatch):
    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, timeout):
            waits.append(timeout)
            return self.stopped

    waits, commands = [], []
    stop = Stop()

    def spawn(command, **kwargs):
        commands.append(command)
        if len(commands) == 3:
            stop.set()
        return SimpleNamespace(poll=lambda: 9, returncode=9)

    monkeypatch.setattr(record_forever.threading, "Event", lambda: stop)
    monkeypatch.setattr(record_forever.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: None, join=lambda **kw: None))
    monkeypatch.setattr(record_forever.subprocess, "Popen", spawn)
    record_forever.run(tmp_path)
    assert waits == [5, 10]
    assert len(commands) == 3
    assert all(command[command.index("--concurrency") + 1] == "4" and "--compact" in command for command in commands)
