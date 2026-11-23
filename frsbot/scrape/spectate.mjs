// Passive ranked spectator capture; never joins a room, queues, or sends robot inputs.
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, spawnSync } from 'node:child_process';

const endpoint = 'wss://reefscape-ws.frsim.com/rebuilt-ws';
const replayBase = 'https://reefscape-ws.frsim.com/rebuilt-replays';

export async function fetchReplay(id, output) {
  if (!/^[A-Za-z0-9_-]+$/.test(id)) throw new Error('invalid replay id');
  const response = await fetch(`${replayBase}/${encodeURIComponent(id)}`, { signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`replay ${id}: HTTP ${response.status}`);
  const data = Buffer.from(await response.arrayBuffer());
  if (data.subarray(0, 4).toString() !== 'RBRP' && !(data[0] === 31 && data[1] === 139)) {
    throw new Error('replay response is neither RBRP nor gzip');
  }
  mkdirSync(output, { recursive: true });
  const target = path.join(output, `${id}.rbrp`);
  writeFileSync(target, data);
  console.log(`REPLAY ${target} bytes=${data.length}`);
  return target;
}

export function capture({ output, code, seconds = 210, list = false, fresh = false, mode, Socket = WebSocket, replayFetch = fetchReplay, signal, context = {}, archive = true }) {
  if (!Number.isFinite(seconds) || seconds <= 0 || seconds > 3600) throw new Error('seconds must be in (0, 3600]');
  if (code && !/^[A-Z0-9]{4,12}$/.test(code)) throw new Error('invalid ranked code');
  if (mode && !['1v1', '2v2', '3v3'].includes(mode)) throw new Error('invalid mode');
  mkdirSync(output, { recursive: true });
  return new Promise((resolve, reject) => {
    const socket = new Socket(endpoint);
    socket.binaryType = 'arraybuffer';
    let target, bytes = 0, frames = 0, ending = false, polling;
    let existing;
    const deadline = setTimeout(() => finish('timeout'), seconds * 1000);
    const save = record => {
      if (target) appendFileSync(target, JSON.stringify(record) + '\n');
    };
    const send = message => {
      if (!['rankedLive', 'rankedWatch'].includes(message.t)) throw new Error('non-spectator message refused');
      socket.send(JSON.stringify(message));
    };
    const watch = room => {
      code = room;
      target = path.join(output, `ranked-${room}-${Date.now()}.jsonl`);
      save({ type: 'capture', version: 1, source: 'ranked', code, at: Date.now(), ...context });
      console.log(`WATCH code=${room} capture=${target}`);
      send({ t: 'rankedWatch', code: room, name: 'ImAI observer', team: 254, tab: crypto.randomUUID() });
    };
    async function finish(reason, replay) {
      if (ending) return;
      ending = true;
      clearTimeout(deadline);
      clearTimeout(polling);
      signal?.removeEventListener('abort', aborted);
      socket.close();
      save({ type: 'end', reason, frames, bytes, replay: replay ?? null, at: Date.now() });
      try {
        let downloaded = null;
        if (replay && archive) {
          try { downloaded = await replayFetch(replay, output); }
          catch (error) {
            save({ type: 'replay-unavailable', replay, error: error.message });
            console.warn(`REPLAY_UNAVAILABLE ${error.message}; raw spectator capture retained`);
          }
        }
        if (reason === 'socket-error' || reason === 'server-error') throw new Error(`spectator ${reason}`);
        console.log(`CAPTURE_END reason=${reason} frames=${frames} bytes=${bytes} replay=${downloaded ?? 'none'}`);
        resolve({ capture: target, replay: downloaded, frames, bytes, reason });
      } catch (error) { error.captureResult = { capture: target, frames, bytes, reason }; reject(error); }
    }
    socket.onopen = () => { if (!ending) code ? watch(code) : send({ t: 'rankedLive' }); };
    socket.onmessage = event => {
      if (ending) return;
      if (typeof event.data !== 'string') {
        const data = Buffer.from(event.data);
        bytes += data.length;
        frames++;
        save({ type: 'binary', at: Date.now(), data: data.toString('base64') });
        if (bytes > 128 * 1024 * 1024) finish('byte-limit');
        return;
      }
      const message = JSON.parse(event.data);
      save({ type: 'text', at: Date.now(), message });
      if (message.t === 'rankedLive') {
        if (list) { console.log(JSON.stringify(message, null, 2)); finish('listed'); return; }
        const modes = { '3v3': 0, '2v2': 1, '1v1': 2 };
        if (fresh && !existing) existing = new Set(message.matches.map(m => m.code));
        const matches = message.matches.filter(m => (!mode || m.mode === mode) && (!fresh || !existing.has(m.code)))
          .sort((a, b) => (modes[a.mode] ?? 9) - (modes[b.mode] ?? 9) || b.rating - a.rating);
        if (fresh && !matches.length) {
          polling = setTimeout(() => send({ t: 'rankedLive' }), 5000);
          return;
        }
        if (!matches.length) { finish('no-live-matches'); return; }
        watch(matches[0].code);
      } else if (message.t === 'over') {
        finish(message.why || 'over', message.replay);
      } else if (message.t === 'error' || message.t === 'rankedError') {
        console.error('SPECTATOR_ERROR', message.msg ?? message.message);
        finish('server-error');
      } else {
        if (message.t === 'start') console.log(`START robots=${message.match?.robots?.length}`);
      }
    };
    socket.onerror = () => finish('socket-error');
    socket.onclose = () => finish('socket-closed');
    const aborted = () => finish('batch-deadline');
    signal?.addEventListener('abort', aborted, { once: true });
    if (signal?.aborted) aborted();
  });
}

export function decodeCapture(target) {
  return new Promise((resolve, reject) => {
    const child = spawn('python3', ['-m', 'frsbot.replay.spectator', target],
      { stdio: 'inherit', cwd: fileURLToPath(new URL('../../', import.meta.url)) });
    child.on('error', reject);
    child.on('exit', code => code === 0 ? resolve() : reject(new Error(`decode exited ${code}: ${target}`)));
  });
}

export function compactCapture(target) {
  return new Promise((resolve, reject) => {
    const child = spawn('python3', ['-m', 'frsbot.scrape.record_forever', '--process', target],
      { stdio: 'inherit', cwd: fileURLToPath(new URL('../../', import.meta.url)) });
    child.on('error', reject);
    child.on('exit', code => code === 0 ? resolve() : reject(new Error(`compact exited ${code}: ${target}`)));
  });
}

export function batch({ output, minutes, concurrency = 2, mode, Socket = WebSocket,
  watch = capture, decode, pollMs = 5000, signal, seenKeys = [], archive = true }) {
  if (!Number.isFinite(minutes) || minutes <= 0) throw new Error('minutes must be positive');
  if (![1, 2, 3, 4].includes(concurrency)) throw new Error('concurrency must be between 1 and 4');
  if (mode && !['1v1', '2v2', '3v3'].includes(mode)) throw new Error('invalid mode');
  mkdirSync(output, { recursive: true });
  return new Promise((resolve, reject) => {
    const started = Date.now(), until = started + minutes * 60000;
    const seen = new Set(seenKeys), active = new Map(), results = [];
    const controller = new AbortController();
    const browser = new Socket(endpoint);
    let ending = false, decoding = Promise.resolve();
    const timer = setTimeout(() => stop(), minutes * 60000);
    const poll = setInterval(() => {
      if (!ending && active.size < concurrency && browser.readyState === 1) list();
    }, pollMs);
    const list = () => browser.send(JSON.stringify({ t: 'rankedLive' }));
    const aborted = () => { void stop(); };
    signal?.addEventListener('abort', aborted, { once: true });
    async function stop(error) {
      if (ending) return;
      ending = true;
      signal?.removeEventListener('abort', aborted);
      clearTimeout(timer);
      clearInterval(poll);
      browser.close();
      controller.abort();
      await Promise.allSettled([...active.values()]);
      try { await decoding; } catch (e) { error ??= e; }
      const report = { started, stopped: Date.now(), minutes, concurrency, results, error: error?.message ?? null };
      const manifest = path.join(output, `batch-${started}.json`);
      writeFileSync(manifest, JSON.stringify(report, null, 2));
      console.log(`BATCH_END captures=${results.length} manifest=${manifest}`);
      if (error) reject(error); else resolve(report);
    }
    browser.onopen = () => { if (!ending) list(); };
    browser.onmessage = event => {
      if (ending) return;
      try {
        const message = JSON.parse(event.data);
        if (['error', 'rankedError'].includes(message.t)) throw new Error(message.msg ?? 'ranked list error');
        if (message.t !== 'rankedLive') return;
        const matches = message.matches.filter(m => !mode || m.mode === mode)
          .sort((a, b) => b.since - a.since);
        for (const match of matches) {
          if (active.size >= concurrency || Date.now() >= until) break;
          const key = `${match.code}:${match.since}:${match.game}`;
          if (seen.has(key)) continue;
          seen.add(key);
          const job = Promise.resolve().then(() => watch({ output, code: match.code,
            seconds: Math.min(210, (until - Date.now()) / 1000), signal: controller.signal,
            archive, context: { mode: match.mode, rating: match.rating, game: match.game, since: match.since, match_key: key } }))
            .then(result => {
              results.push({ key, mode: match.mode, ...result });
              if (decode && result.capture && result.frames) {
                decoding = decoding.then(() => decode(result.capture));
                decoding.catch(error => { void stop(error); });
              }
            }).catch(error => {
              if (decode && error.captureResult?.capture && error.captureResult.frames) {
                decoding = decoding.then(() => decode(error.captureResult.capture));
                decoding.catch(() => {});
              }
              void stop(error);
            })
            .finally(() => active.delete(key));
          active.set(key, job);
        }
      } catch (error) { void stop(error); }
    };
    browser.onerror = () => stop(new Error('batch list socket error; not retrying'));
    browser.onclose = () => { if (!ending) void stop(new Error('batch list socket closed; not retrying')); };
    if (signal?.aborted) aborted();
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const value = key => args.includes(key) ? args[args.indexOf(key) + 1] : undefined;
  const output = path.resolve(value('--output') ?? fileURLToPath(new URL('../../data/raw_mp', import.meta.url)));
  const id = value('--replay');
  if (args.includes('--compact')) console.log = () => {};
  try {
    if (value('--minutes') !== undefined) {
      const controller = new AbortController();
      const stop = () => controller.abort();
      process.on('SIGINT', stop);
      process.on('SIGTERM', stop);
      const seenKeys = value('--seen-file') ? readFileSync(value('--seen-file'), 'utf8').split('\n')
        .filter(Boolean).map(line => JSON.parse(line).match_key) : [];
      await batch({ output, minutes: Number(value('--minutes')), concurrency: Number(value('--concurrency') ?? 2),
        mode: value('--mode'), signal: controller.signal, seenKeys, archive: !args.includes('--compact'),
        decode: args.includes('--compact') ? compactCapture : args.includes('--decode') ? decodeCapture : undefined });
      process.removeListener('SIGINT', stop);
      process.removeListener('SIGTERM', stop);
    }
    else if (id) await fetchReplay(id, output);
    else {
      const result = await capture({ output, code: value('--code'), seconds: Number(value('--seconds') ?? 210), list: args.includes('--list'), fresh: args.includes('--fresh'), mode: value('--mode') });
      if (!args.includes('--list') && !result.capture) throw new Error(`no match captured (${result.reason})`);
      if (args.includes('--decode') && result.capture) {
        const converted = spawnSync('python3', ['-m', 'frsbot.replay.spectator', result.capture],
          { stdio: 'inherit', cwd: fileURLToPath(new URL('../../', import.meta.url)) });
        if (converted.status !== 0) throw new Error(converted.error?.message ?? 'spectator replay conversion failed');
      }
    }
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
