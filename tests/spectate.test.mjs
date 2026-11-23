import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { batch, capture, fetchReplay } from '../frsbot/scrape/spectate.mjs';

function server(matches) {
  return class {
    static sent = [];
    constructor() { this.readyState = 1; queueMicrotask(() => this.onopen()); }
    close() { this.readyState = 3; queueMicrotask(() => this.onclose()); }
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

test('batch takes successive mixed modes once, never more than two watches', async () => {
  const matches = ['1v1', '2v2', '3v3', '1v1'].map((mode, i) => ({ code: `ABC${i}`, mode, since: i, game: 1 }));
  const Socket = server(matches);
  const output = mkdtempSync(path.join(os.tmpdir(), 'batch-test-'));
  let active = 0, peak = 0;
  const watched = [], decoded = [];
  const watch = options => new Promise(resolve => {
    watched.push(options.code);
    assert(options.signal);
    peak = Math.max(peak, ++active);
    setTimeout(() => { active--; resolve({ capture: options.code, frames: 1 }); }, 10);
  });
  const result = await batch({ output, minutes: .002, Socket, watch, pollMs: 10, decode: async f => decoded.push(f) });
  assert.equal(result.results.length, 4);
  assert.equal(new Set(watched).size, 4);
  assert.equal(decoded.length, 4);
  assert.equal(peak, 2);
  assert(Socket.sent.every(m => m.t === 'rankedLive'));
  assert.throws(() => batch({ minutes: 1, concurrency: 5 }), /concurrency/);
  assert.throws(() => batch({ minutes: 0 }), /minutes/);
});

test('batch deadline aborts active watches and empty lists terminate cleanly', async () => {
  const output = mkdtempSync(path.join(os.tmpdir(), 'batch-test-'));
  let aborted = 0;
  const result = await batch({ output, minutes: .001, Socket: server([{ code: 'ABC1', since: 1 }]),
    watch: o => new Promise(resolve => o.signal.addEventListener('abort', () => { aborted++; resolve({ frames: 0 }); })) });
  assert.equal(aborted, 1);
  assert.equal(result.results.length, 1);
  const empty = await batch({ output, minutes: .001, Socket: server([]) });
  assert.equal(empty.results.length, 0);
});

test('batch stops on listing error without retrying', async () => {
  class Broken {
    constructor() { queueMicrotask(() => this.onerror()); }
    close() {}
  }
  await assert.rejects(batch({ output: mkdtempSync(path.join(os.tmpdir(), 'batch-test-')), minutes: 1, Socket: Broken }), /not retrying/);
});

test('four-watch mode honors persistent dedupe and external shutdown', async () => {
  const output = mkdtempSync(path.join(os.tmpdir(), 'batch-four-'));
  const matches = Array.from({ length: 6 }, (_, i) => ({ code: `ABC${i}`, since: i, game: 1 }));
  let active = 0, peak = 0;
  const result = await batch({ output, minutes: .002, concurrency: 4, Socket: server(matches),
    seenKeys: ['ABC0:0:1'], pollMs: 10,
    watch: o => new Promise(resolve => {
      assert.equal(o.context.match_key, `${o.code}:${o.context.since}:1`);
      peak = Math.max(peak, ++active);
      setTimeout(() => { active--; resolve({ frames: 0 }); }, 10);
    }) });
  assert.equal(peak, 4);
  assert.equal(result.results.length, 5);
  const controller = new AbortController();
  controller.abort();
  const stopped = await batch({ output, minutes: 10, Socket: server(matches), signal: controller.signal });
  assert.equal(stopped.results.length, 0);
});

test('failed watch still queues captured frames for compact recovery', async () => {
  let decoded = false;
  await assert.rejects(batch({ output: mkdtempSync(path.join(os.tmpdir(), 'batch-error-')), minutes: 1,
    Socket: server([{ code: 'ABC1', since: 1, game: 1 }]),
    watch: async () => { throw Object.assign(new Error('disconnect'), { captureResult: { capture: 'partial', frames: 1 } }); },
    decode: async target => { assert.equal(target, 'partial'); decoded = true; } }), /disconnect/);
  assert(decoded);
});
