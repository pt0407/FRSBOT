import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const directory = path.resolve(process.argv[2] ?? fileURLToPath(new URL('../../../data/engine', import.meta.url)));
const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'manifest.json'), 'utf8'));
const teams = new Set(JSON.parse(fs.readFileSync(new URL('../../../data/robot_feel.json', import.meta.url), 'utf8')).map(r => r.team));
const constructor = 'constructor(t,e,i){this.e=t,this.p=e,this.ids=i,';
let sim = null, started = false;

function validateConfig(config) {
  if (!teams.has(config.team)) throw new Error(`unknown team ${config.team}`);
  if (!['blue', 'red'].includes(config.alliance)) throw new Error('alliance must be blue or red');
  if (!['event', 'districtChampionship', 'championship'].includes(config.event)) throw new Error('invalid event');
  if (!Number.isInteger(config.seed) || config.seed < 0 || config.seed > 0xffffffff) throw new Error('seed must be a uint32');
  if (!config.start || !['fx', 'fy', 'headingDeg'].every(k => Number.isFinite(config.start[k]))) throw new Error('start requires finite fx, fy, headingDeg');
  if (typeof config.auto !== 'string') throw new Error('auto must be a name or "none"');
  for (const key of ['hp', 'unload', 'freeplay']) if (typeof config[key] !== 'boolean') throw new Error(`${key} must be boolean`);
}

async function create(config) {
  const defaults = { alliance: 'blue', start: { fx: 0, fy: 0, headingDeg: 0 }, seed: 1,
    auto: 'none', event: 'event', hp: true, unload: true, freeplay: false };
  config = { ...defaults, ...config };
  validateConfig(config);
  sim?.dispose(); sim = null; started = false;
  let ready, fail;
  const prepared = new Promise((resolve, reject) => { ready = resolve; fail = reject; });
  const timer = setTimeout(() => fail(new Error('engine prepare timed out')), 30000);
  const context = vm.createContext({ URL, Response, WebAssembly, performance, TextEncoder, TextDecoder, crypto,
    setTimeout, clearTimeout, setInterval, clearInterval, structuredClone, atob, btoa,
    console: { ...console, log: (...a) => console.error(...a), error: (...a) => {
      console.error(...a); fail(new Error(a.map(String).join(' ')));
    } }, __sims: [], __listeners: [],
  });
  const local = url => {
    const key = new URL(String(url), manifest.site).pathname;
    const name = manifest.assets[key];
    if (!name || path.basename(name) !== name) throw new Error(`uncached engine asset ${key}; run python3 -m frsbot.sim.engine.fetch`);
    return path.join(directory, name);
  };
  context.fetch = async url => new Response(fs.readFileSync(local(url)), {
    headers: { 'content-type': String(url).includes('.wasm') ? 'application/wasm' : 'text/javascript' },
  });
  context.importScripts = (...urls) => {
    for (const url of urls) {
      const file = local(url);
      let source = fs.readFileSync(file, 'utf8');
      if (path.basename(file) === manifest.sim_chunk) {
        if (source.split(constructor).length !== 2) throw new Error('WASM sim constructor changed; update engine adapter');
        source = source.replace(constructor, 'constructor(t,e,i){globalThis.__sims.push(this);this.e=t,this.p=e,this.ids=i,');
      }
      vm.runInContext(source, context, { filename: file });
    }
  };
  context.addEventListener = (kind, listener) => { if (kind === 'message') context.__listeners.push(listener); };
  context.postMessage = message => {
    if (message.t === 'preview') ready(context.__sims.at(-1));
    if (message.t === 'error') fail(new Error(message.message));
  };
  try {
    vm.runInContext('globalThis.self=globalThis;globalThis.WorkerGlobalScope=function(){};Object.setPrototypeOf(globalThis,WorkerGlobalScope.prototype)', context);
    context.location = new URL('/_next/static/chunks/worker.js#params=' + encodeURIComponent(JSON.stringify([
      manifest.chunks, '', '/_next/', null, null,
    ])), manifest.site);
    context.importScripts(new URL(manifest.bootstrap, manifest.site));
    await new Promise(resolve => setImmediate(resolve));
    if (!context.onmessage && !context.__listeners.length) throw new Error('worker installed no message handler; update engine adapter');
    const event = { data: { ...config, t: 'prepare', builderAuto: null, practice: null, troll: false, engine: 'wasm' } };
    context.onmessage?.(event);
    for (const listener of context.__listeners) listener(event);
    sim = await prepared;
    if (!sim) throw new Error('worker prepared without a captured WASM simulation');
    return sim.views();
  } finally { clearTimeout(timer); }
}

function inputsFor(raw = {}) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('inputs must map robot ids to inputs');
  return new Map(Object.entries(raw).map(([key, input]) => {
    const id = Number(key);
    if (!sim.ids.includes(id)) throw new Error(`unknown robot id ${key}`);
    if (!input || typeof input !== 'object') throw new Error('invalid robot inputs');
    for (const axis of ['x', 'y', 'turn']) {
      if (input[axis] !== undefined && (!Number.isFinite(input[axis]) || Math.abs(input[axis]) > 1)) throw new Error(`${axis} must be in [-1, 1]`);
    }
    for (const button of ['intake', 'score', 'outtake', 'climb', 'block', 'stow', 'fieldRelative', 'hood']) {
      if (input[button] !== undefined && typeof input[button] !== 'boolean') throw new Error(`${button} must be boolean`);
    }
    return [id, { fieldRelative: true, ...input }];
  }));
}

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
try {
  for await (const line of lines) {
    let closing = false;
    try {
      const request = JSON.parse(line);
      let view;
      if (request.cmd === 'new') view = await create(request.config ?? Object.fromEntries(Object.entries(request).filter(([k]) => k !== 'cmd')));
      else if (request.cmd === 'close') { sim?.dispose(); sim = null; closing = true; }
      else {
        if (!sim) throw new Error('call new before using the simulation');
        if (request.cmd === 'start') {
          if (started) throw new Error('simulation already started; call new to reset');
          if (request.countdown !== undefined && typeof request.countdown !== 'boolean') throw new Error('countdown must be boolean');
          sim.start({ countdown: request.countdown ?? false }); started = true;
        } else if (request.cmd === 'step') {
          if (!started) throw new Error('call start before stepping');
          const n = request.n ?? 1;
          if (!Number.isInteger(n) || n < 1 || n > 60000) throw new Error('n must be an integer in [1, 60000]');
          const inputs = inputsFor(request.inputs);
          for (let i = 0; i < n; i++) sim.step(inputs);
        } else if (request.cmd !== 'state') throw new Error(`unknown command ${request.cmd}`);
        view = sim.views();
      }
      process.stdout.write(JSON.stringify({ ok: true, view, engine: sim?.engine, build: sim?.engineBuild },
        (_key, value) => ArrayBuffer.isView(value) ? Array.from(value) : value) + '\n');
    } catch (error) { process.stdout.write(JSON.stringify({ ok: false, error: error.message }) + '\n'); }
    if (closing) break;
  }
} finally { sim?.dispose(); lines.close(); }
