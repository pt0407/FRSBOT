// Loads the game's match worker once per process and exposes its WASM engine for many concurrent matches.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

export const DEFAULT_DIR = fileURLToPath(new URL('../../../data/engine', import.meta.url));
export const TEAMS = new Set(JSON.parse(fs.readFileSync(new URL('../../../data/robot_feel.json', import.meta.url), 'utf8')).map(r => r.team));
const SIM_CONSTRUCTOR = 'constructor(t,e,i){this.e=t,this.p=e,this.ids=i,';
const TEAM_LOOKUP = 'e=(0,s.teamByNumber)(t.team)';
const EVENTS = ['event', 'districtChampionship', 'championship'];

export function validateMatch(config) {
  if (!Array.isArray(config.robots) || config.robots.length < 1 || config.robots.length > 6) throw new Error('robots must list 1 to 6 robots');
  for (const robot of config.robots) {
    if (!TEAMS.has(robot.team)) throw new Error(`unknown team ${robot.team}`);
    if (!['blue', 'red'].includes(robot.alliance)) throw new Error('alliance must be blue or red');
    if (!robot.start || !['fx', 'fy', 'headingDeg'].every(k => Number.isFinite(robot.start[k]))) throw new Error('start requires finite fx, fy, headingDeg');
  }
  if (!Number.isInteger(config.ego) || config.ego < 0 || config.ego >= config.robots.length) throw new Error('ego must index robots');
  if (!EVENTS.includes(config.event)) throw new Error('invalid event');
  if (!Number.isInteger(config.seed) || config.seed < 0 || config.seed > 0xffffffff) throw new Error('seed must be a uint32');
  if (typeof config.auto !== 'string') throw new Error('auto must be a name or "none"');
  for (const key of ['hp', 'unload', 'freeplay']) if (typeof config[key] !== 'boolean') throw new Error(`${key} must be boolean`);
}

export function normalizeMatch(config) {
  const defaults = { alliance: 'blue', start: { fx: 0, fy: 0, headingDeg: 0 }, seed: 1, auto: 'none', event: 'event',
    hp: true, unload: true, freeplay: false, ego: 0 };
  config = { ...defaults, ...config };
  if (!config.robots) config.robots = [{ team: config.team, alliance: config.alliance, start: config.start }];
  config.robots = config.robots.map(r => ({ alliance: 'blue', start: { fx: 0, fy: 0, headingDeg: 0 }, ...r }));
  validateMatch(config);
  return config;
}

export async function loadEngine(directory = DEFAULT_DIR) {
  const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'manifest.json'), 'utf8'));
  let ready, fail, patchedLookup = false;
  const prepared = new Promise((resolve, reject) => { ready = resolve; fail = reject; });
  const timer = setTimeout(() => fail(new Error('engine prepare timed out')), 30000);
  const context = vm.createContext({ URL, Response, WebAssembly, performance, TextEncoder, TextDecoder, crypto,
    setTimeout, clearTimeout, setInterval, clearInterval, structuredClone, atob, btoa,
    console: { ...console, log: (...a) => console.error(...a), error: (...a) => {
      console.error(...a); fail(new Error(a.map(String).join(' ')));
    } }, __sims: [], __listeners: [], __teamByNumber: null,
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
        if (source.split(SIM_CONSTRUCTOR).length !== 2) throw new Error('WASM sim constructor changed; update engine adapter');
        source = source.replace(SIM_CONSTRUCTOR, 'constructor(t,e,i){globalThis.__sims.push(this);this.e=t,this.p=e,this.ids=i,');
      }
      if (source.split(TEAM_LOOKUP).length === 2) {
        source = source.replace(TEAM_LOOKUP, `${TEAM_LOOKUP},globalThis.__teamByNumber=s.teamByNumber`);
        patchedLookup = true;
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
    if (!patchedLookup) throw new Error('worker team lookup changed; update engine adapter');
    const event = { data: { team: 254, alliance: 'blue', start: { fx: 0, fy: 0, headingDeg: 0 }, seed: 1, auto: 'none',
      event: 'event', hp: true, unload: true, freeplay: false, t: 'prepare', builderAuto: null, practice: null, troll: false, engine: 'wasm' } };
    context.onmessage?.(event);
    for (const listener of context.__listeners) listener(event);
    const boot = await prepared;
    if (!boot) throw new Error('worker prepared without a captured WASM simulation');
    if (typeof context.__teamByNumber !== 'function') throw new Error('worker did not expose its team roster');
    const engine = boot.e, teamByNumber = context.__teamByNumber;
    const name = boot.engine, build = boot.engineBuild;
    boot.dispose();
    return {
      name, build,
      // Mirrors the game's solo config builder; several robots make a non-solo match on the same field.
      create(raw) {
        const config = normalizeMatch(raw);
        const sim = engine.create({
          robots: config.robots.map((r, id) => ({ id, team: teamByNumber(r.team), alliance: r.alliance, start: r.start,
            preload: r.preload ?? 8, local: id === config.ego })),
          seed: config.seed, auto: config.auto, event: config.event, solo: config.robots.length === 1,
          hpThrows: config.hp, hpUnloads: config.unload, freeplay: config.freeplay,
        });
        return { sim, config };
      },
    };
  } finally { clearTimeout(timer); }
}
