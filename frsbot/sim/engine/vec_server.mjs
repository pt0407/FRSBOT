// Many matches per Node process over a length-prefixed binary protocol (frsbot/sim/ipc.py is the other end).
// Frames both ways: u32 little-endian payload length, then payload. Requests start with a u8 opcode.
import fs from 'node:fs';
import path from 'node:path';
import { DEFAULT_DIR, loadEngine } from './loader.mjs';
import { Driver, GOAL_KINDS, viewToState } from './driver.mjs';

export const OP = { NEW: 1, STEP: 2, STATE: 3, CLOSE: 4, EXIT: 5 };
export const HEADER = 24, ROBOT = 24, MAX_ROBOTS = 6;
export const CMD = { INPUTS: 0, GOAL: 1, IDLE: 2 };
const PHASES = ['countdown', 'auto', 'gap', 'transition', 'shift1', 'shift2', 'shift3', 'shift4', 'endgame', 'settle', 'done'];
const SCORE = ['autoFuel', 'teleFuel', 'inactiveFuel', 'towerAuto', 'towerTele', 'foulPoints', 'total'];
const GATE = ['open', 'hood', 'flywheel', 'range', 'heading', 'speed', 'tag'];
const BUTTONS = ['intake', 'score', 'outtake', 'climb', 'block', 'stow', 'fieldRelative', 'hood'];
const IDLE = new Map();

const strings = new Map();
let fresh = [];
function intern(s) {
  if (s == null) return -1;
  let code = strings.get(s);
  if (code === undefined) { code = strings.size; strings.set(s, code); fresh.push([code, s]); }
  return code;
}

function blockSize(count) { return (HEADER + ROBOT * MAX_ROBOTS) * 8 + count * 13 + ((8 - count * 13 % 8) % 8); }

// Header f64[24]: slot, elapsed, clock, phase, firstInactive(-1 none, 0 blue, 1 red), blue active, red active,
// robots, blue score[7], red score[7], ego, fuel count. Robot f64[24] each (unused rows are zero); FUEL f32 pos, u8 where.
function encode(match, view, out = Buffer.alloc(blockSize(view.fuel.count)), offset = 0) {
  const count = view.fuel.count;
  const f = new Float64Array(out.buffer, out.byteOffset + offset, HEADER + ROBOT * MAX_ROBOTS);
  f.fill(0);
  f[0] = match.slot; f[1] = view.elapsed; f[2] = view.clock; f[3] = PHASES.indexOf(view.phase);
  f[4] = view.firstInactive == null ? -1 : view.firstInactive === 'blue' ? 0 : 1;
  f[5] = view.hubs.blue.active ? 1 : 0; f[6] = view.hubs.red.active ? 1 : 0; f[7] = view.robots.length;
  for (let i = 0; i < 7; i++) { f[8 + i] = view.score.blue[SCORE[i]]; f[15 + i] = view.score.red[SCORE[i]]; }
  f[22] = match.config.ego; f[23] = count;
  const fed = new Map((view.drivers ?? []).map(d => [d.robot, d.fuel]));
  view.robots.forEach((r, i) => {
    const o = HEADER + ROBOT * i, robot = match.robots[i];
    f[o] = r.id; f[o + 1] = r.team; f[o + 2] = r.alliance === 'red' ? 1 : 0; f[o + 3] = r.local ? 1 : 0;
    f.set(r.pos, o + 4); f.set(r.quat, o + 7); f.set(r.vel, o + 11);
    f[o + 14] = r.hopperFuel; f[o + 15] = r.hopperCapacity;
    f[o + 16] = r.gate ? GATE.reduce((bits, key, b) => bits | (r.gate[key] ? 1 << b : 0), 0) : -1;
    f[o + 17] = intern(r.state ?? null);
    f[o + 18] = robot.abandoned ? 1 : 0; f[o + 19] = robot.driver?.recovering ? 1 : 0;
    f[o + 20] = robot.driver?.recoveryStage ?? 0; f[o + 21] = fed.get(r.id) ?? 0; f[o + 22] = robot.kind; f[o + 23] = r.drumRpm ?? 0;
  });
  const base = offset + (HEADER + ROBOT * MAX_ROBOTS) * 8;
  new Float32Array(out.buffer, out.byteOffset + base, count * 3).set(view.fuel.pos.subarray(0, count * 3));
  out.set(view.fuel.where.subarray(0, count), base + count * 12);
  return out;
}

class Match {
  constructor(engine, slot, request) {
    const { sim, config } = engine.create(request.config ?? {});
    this.slot = slot; this.sim = sim; this.config = config; this.freeplay = config.freeplay;
    this.delay = request.delay ?? 0; this.hold = request.hold ?? 1; this.tick = 0;
    for (const [key, v, hi] of [['delay', this.delay, 600], ['hold', this.hold, 60]])
      if (!Number.isInteger(v) || v < (key === 'hold' ? 1 : 0) || v > hi) throw new Error(`${key} must be an integer tick count`);
    const drivers = request.drivers ?? {};
    this.robots = config.robots.map((_, i) => ({ kind: CMD.IDLE, inputs: null, goal: null, abandoned: false,
      driver: drivers[i] ? new Driver(drivers[i].spec, drivers[i].profile ?? {}) : null, pending: null }));
    for (const [i, d] of Object.entries(drivers)) if (d.shootWhileCollecting != null && this.robots[i]) this.robots[i].driver.shootWhileCollecting = d.shootWhileCollecting;
    sim.start({ countdown: request.countdown ?? false });
    this.ring = [];
    if (this.delay) { const block = encode(this, sim.views()); for (let i = 0; i <= this.delay; i++) this.ring.push(block); }
  }
  command(robot, cmd) {
    const r = this.robots[robot];
    if (!r) throw new Error(`unknown robot ${robot} in match ${this.slot}`);
    if (cmd.kind === CMD.GOAL && !r.driver) throw new Error(`robot ${robot} has no driver config for goals`);
    if (this.hold > 1) r.pending = cmd; else this.apply(r, cmd);
  }
  apply(r, cmd) {
    r.kind = cmd.kind; r.inputs = cmd.inputs ?? null;
    if (cmd.kind === CMD.GOAL) { r.goal = cmd.goal; r.abandoned = false; } else r.goal = null;
  }
  step(n) {
    const sim = this.sim, robots = this.robots;
    for (let k = 0; k < n; k++) {
      if (this.hold > 1 && this.tick % this.hold === 0) for (const r of robots) if (r.pending) { this.apply(r, r.pending); r.pending = null; }
      let inputs = IDLE, state = null;
      for (let i = 0; i < robots.length; i++) {
        const r = robots[i];
        if (r.kind === CMD.IDLE) continue;
        if (inputs === IDLE) inputs = new Map();
        if (r.kind === CMD.INPUTS) inputs.set(i, r.inputs);
        else {
          state ??= viewToState(sim.views(), this.freeplay);
          state.me = i;
          inputs.set(i, r.driver.act(r.goal, state));
          if (r.driver.abandon) r.abandoned = true;
        }
      }
      sim.step(inputs);
      this.tick++;
      if (this.delay) { this.ring.push(encode(this, sim.views())); this.ring.shift(); }
    }
  }
  observe(out, offset) {
    if (this.delay) { this.ring[0].copy(out, offset); return; }
    encode(this, this.sim.views(), out, offset);
  }
  size() { return this.delay ? this.ring[0].length : blockSize(this.sim.views().fuel.count); }
  dispose() { this.sim.dispose(); }
}

function readCommands(buf, at, match) {
  const count = buf.readUInt8(at++);
  for (let c = 0; c < count; c++) {
    const robot = buf.readUInt8(at), kind = buf.readUInt8(at + 1); at += 2;
    if (kind === CMD.INPUTS) {
      const x = buf.readFloatLE(at), y = buf.readFloatLE(at + 4), turn = buf.readFloatLE(at + 8), bits = buf.readUInt16LE(at + 12); at += 14;
      for (const [axis, v] of [['x', x], ['y', y], ['turn', turn]]) if (!Number.isFinite(v) || Math.abs(v) > 1) throw new Error(`${axis} must be in [-1, 1]`);
      const inputs = { x, y, turn };
      BUTTONS.forEach((b, i) => { inputs[b] = !!(bits & (1 << i)); });
      match.command(robot, { kind, inputs });
    } else if (kind === CMD.GOAL) {
      const g = buf.readUInt8(at), px = buf.readFloatLE(at + 1), pz = buf.readFloatLE(at + 5); at += 9;
      if (!GOAL_KINDS[g] || !Number.isFinite(px) || !Number.isFinite(pz)) throw new Error('invalid goal');
      match.command(robot, { kind, goal: { kind: GOAL_KINDS[g], point: [px, pz] } });
    } else if (kind === CMD.IDLE) match.command(robot, { kind });
    else throw new Error(`unknown command kind ${kind}`);
  }
  return at;
}

export async function serve(directory = DEFAULT_DIR, input = process.stdin, output = process.stdout) {
  if (!fs.existsSync(path.join(directory, 'manifest.json'))) throw new Error('engine assets missing; run python3 -m frsbot.sim.engine.fetch');
  const engine = await loadEngine(directory);
  const matches = new Map();
  const reply = (status, body) => {
    const extra = fresh.length ? Buffer.from(JSON.stringify(fresh)) : Buffer.alloc(0);
    fresh = [];
    const head = Buffer.alloc(9);
    head.writeUInt32LE(5 + extra.length + body.length, 0); head.writeUInt8(status, 4); head.writeUInt32LE(extra.length, 5);
    output.write(Buffer.concat([head, extra, body]));
  };
  const observe = slots => {
    const list = slots.map(s => { const m = matches.get(s); if (!m) throw new Error(`no match in slot ${s}`); return m; });
    const sizes = list.map(m => m.size());
    const out = Buffer.alloc(8 + sizes.reduce((a, b) => a + b, 0));
    out.writeUInt16LE(list.length, 0);
    let at = 8;
    list.forEach((m, i) => { m.observe(out, at); at += sizes[i]; });
    return out;
  };
  const handle = buf => {
    const op = buf.readUInt8(0);
    if (op === OP.NEW) {
      const request = JSON.parse(buf.subarray(1).toString('utf8'));
      const slot = request.slot ?? 0;
      if (!Number.isInteger(slot) || slot < 0 || slot > 0xffff) throw new Error('slot must be a u16');
      const match = new Match(engine, slot, request);
      matches.get(slot)?.dispose();
      matches.set(slot, match);
      return observe([slot]);
    }
    if (op === OP.STEP) {
      const n = buf.readUInt16LE(1), count = buf.readUInt16LE(3);
      if (n < 1) throw new Error('n must be an integer in [1, 65535]');
      let at = 5; const slots = [];
      for (let i = 0; i < count; i++) {
        const slot = buf.readUInt16LE(at), match = matches.get(slot);
        if (!match) throw new Error(`no match in slot ${slot}`);
        at = readCommands(buf, at + 2, match); slots.push(slot);
      }
      for (const slot of slots) matches.get(slot).step(n);
      return observe(slots);
    }
    if (op === OP.STATE) {
      const count = buf.readUInt16LE(1);
      return observe(Array.from({ length: count }, (_, i) => buf.readUInt16LE(3 + 2 * i)));
    }
    if (op === OP.CLOSE) { const slot = buf.readUInt16LE(1); matches.get(slot)?.dispose(); matches.delete(slot); return Buffer.alloc(0); }
    if (op === OP.EXIT) return null;
    throw new Error(`unknown opcode ${op}`);
  };
  reply(0, Buffer.from(JSON.stringify({ engine: engine.name, build: engine.build })));
  let pending = Buffer.alloc(0);
  try {
    for await (const chunk of input) {
      pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
      while (pending.length >= 4 && pending.length >= 4 + pending.readUInt32LE(0)) {
        const size = pending.readUInt32LE(0), frame = pending.subarray(4, 4 + size);
        pending = pending.subarray(4 + size);
        let body;
        try { body = handle(frame); } catch (error) { reply(1, Buffer.from(String(error.message))); continue; }
        if (body === null) { reply(0, Buffer.alloc(0)); return; }
        reply(0, body);
      }
    }
  } finally { for (const m of matches.values()) m.dispose(); }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  serve(path.resolve(process.argv[2] ?? DEFAULT_DIR)).then(() => process.exit(0), error => { console.error(error); process.exit(1); });
}
