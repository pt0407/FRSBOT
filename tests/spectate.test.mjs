import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { capture, fetchReplay } from '../frsbot/scrape/spectate.mjs';

function server(matches) {
  return class {
    static sent = [];
    constructor() { queueMicrotask(() => this.onopen()); }
    close() { queueMicrotask(() => this.onclose()); }
    emit(data) { queueMicrotask(() => this.onmessage({ data: JSON.stringify(data) })); }
    send(raw) {
      const message = JSON.parse(raw);
      this.constructor.sent.push(message);
      if (message.t === 'rankedLive') this.emit({ t: 'rankedLive', matches });
      else if (message.t === 'rankedWatch') {
        this.emit({ t: 'start', match: { robots: [{}, {}, {}, {}] } });
        queueMicrotask(() => this.onmessage({ data: new Uint8Array([3, 0, 3, 0]).buffer }));
        this.emit({ t: 'over', replay: 'missing-public-replay' });
      } else throw new Error('non-passive request');
    }
  };
}

test('records using only list/watch requests even when archive is unavailable', async () => {
  const Socket = server([{ code: 'AB12', mode: '2v2', rating: 1000 }]);
  const output = mkdtempSync(path.join(os.tmpdir(), 'spectator-test-'));
  const result = await capture({ output, Socket, replayFetch: async () => { throw new Error('HTTP 404'); } });
  assert.equal(result.frames, 1);
  assert.equal(result.reason, 'over');
  assert.deepEqual(Socket.sent.map(m => m.t), ['rankedLive', 'rankedWatch']);
  const lines = readFileSync(result.capture, 'utf8').trim().split('\n').map(JSON.parse);
  assert(lines.some(r => r.type === 'binary'));
  assert(lines.some(r => r.type === 'replay-unavailable'));
});

test('no matches does not watch or join anything', async () => {
  const Socket = server([]);
  const result = await capture({ output: mkdtempSync(path.join(os.tmpdir(), 'spectator-test-')), Socket });
  assert.equal(result.reason, 'no-live-matches');
  assert.deepEqual(Socket.sent.map(m => m.t), ['rankedLive']);
});

test('invalid options and replay errors are explicit', async t => {
  assert.throws(() => capture({ seconds: NaN }), /seconds/);
  assert.throws(() => capture({ code: '../escape' }), /code/);
  await assert.rejects(fetchReplay('../escape', '.'), /invalid replay/);
  t.mock.method(globalThis, 'fetch', async () => ({ ok: false, status: 429 }));
  await assert.rejects(fetchReplay('unavailable', '.'), /HTTP 429/);
});

test('public replay GET saves RBRP and rejects non-replays', async t => {
  const output = mkdtempSync(path.join(os.tmpdir(), 'spectator-test-'));
  t.mock.method(globalThis, 'fetch', async url => {
    assert.equal(url, 'https://reefscape-ws.frsim.com/rebuilt-replays/publicid');
    return { ok: true, arrayBuffer: async () => Buffer.from('RBRP\x02fixture') };
  });
  const target = await fetchReplay('publicid', output);
  assert.equal(readFileSync(target).subarray(0, 4).toString(), 'RBRP');
  t.mock.method(globalThis, 'fetch', async () => ({ ok: true, arrayBuffer: async () => Buffer.from('<html>') }));
  await assert.rejects(fetchReplay('badbody', output), /neither RBRP nor gzip/);
});
