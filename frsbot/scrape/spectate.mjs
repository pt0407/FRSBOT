// Passive ranked spectator capture; never joins a room, queues, or sends robot inputs.
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

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

export function capture({ output, code, seconds = 210, list = false, fresh = false, mode, Socket = WebSocket, replayFetch = fetchReplay }) {
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
      save({ type: 'capture', version: 1, source: 'ranked', code, at: Date.now() });
      console.log(`WATCH code=${room} capture=${target}`);
      send({ t: 'rankedWatch', code: room, name: 'ImAI observer', team: 254, tab: crypto.randomUUID() });
    };
    async function finish(reason, replay) {
      if (ending) return;
      ending = true;
      clearTimeout(deadline);
      clearTimeout(polling);
      socket.close();
      save({ type: 'end', reason, frames, bytes, replay: replay ?? null, at: Date.now() });
      try {
        let downloaded = null;
        if (replay) {
          try { downloaded = await replayFetch(replay, output); }
          catch (error) {
            save({ type: 'replay-unavailable', replay, error: error.message });
            console.warn(`REPLAY_UNAVAILABLE ${error.message}; raw spectator capture retained`);
          }
        }
        if (reason === 'socket-error' || reason === 'server-error') throw new Error(`spectator ${reason}`);
        console.log(`CAPTURE_END reason=${reason} frames=${frames} bytes=${bytes} replay=${downloaded ?? 'none'}`);
        resolve({ capture: target, replay: downloaded, frames, bytes, reason });
      } catch (error) { reject(error); }
    }
    socket.onopen = () => code ? watch(code) : send({ t: 'rankedLive' });
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
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const value = key => args.includes(key) ? args[args.indexOf(key) + 1] : undefined;
  const output = path.resolve(value('--output') ?? fileURLToPath(new URL('../../data/raw_mp', import.meta.url)));
  const id = value('--replay');
  try {
    if (id) await fetchReplay(id, output);
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
