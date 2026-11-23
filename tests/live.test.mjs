import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const listeners = {}, documentListeners = {};
class Worker {
  constructor() { this.listeners = {}; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  postMessage(message) { this.sent = message; }
  receive(message) { this.listeners.message({ data: message }); }
}
globalThis.window = { Worker, addEventListener: (kind, fn) => { listeners[kind] = fn; } };
Object.defineProperty(globalThis, 'navigator', { value: {}, configurable: true });
globalThis.document = { hidden: false, addEventListener: (kind, fn) => { documentListeners[kind] = fn; } };
vm.runInThisContext(fs.readFileSync(new URL('../frsbot/live/inject.js', import.meta.url), 'utf8'));
const api = window.__imai;
const worker = new window.Worker('solo');
const meta = n => ({ phase: 'auto', clock: 19, elapsed: 1, matchTimer: 19, fuelCount: n, firstInactive: null,
  robots: [{ id: 0, team: 254, alliance: 'blue', local: true, pos: new Float32Array([1, 0, 2]),
    quat: [0, 0, 0, 1], vel: [0, 0, 0], hopperFuel: 3, hopperCapacity: 72, partsLen: 2, modulesLen: 1 }],
  prev: [], hubs: { blue: { active: true, net: new Float32Array([1]) }, red: { active: true } },
  score: { blue: { total: 4 }, red: { total: 0 } } });
const buffer = new ArrayBuffer(77 * 2 + 12);
new Float32Array(buffer, 0, 6).set([1, 2, 3, 4, 5, 6]);
new Uint8Array(buffer, buffer.byteLength - 2).set([0, 1]);
worker.postMessage({ t: 'create', team: 254, seed: 4, start: { fx: 1, fy: 2, headingDeg: 90 }, alliance: 'blue' });
worker.receive({ t: 'ready', match: 1, engine: 'WASM' });
worker.receive({ t: 'frame', meta: meta(2), buf: buffer, stepTime: 123 });
const frame = api.poll(0).frames[0];
assert.deepEqual(frame.view.fuel, { count: 2, pos: [1, 2, 3, 4, 5, 6], where: [0, 1] });
assert.deepEqual(frame.view.robots[0].pos, [1, 0, 2]);
assert.deepEqual(frame.view.hubs.blue.net, [1]);
assert.equal(frame.view.score.blue.total, 4);
assert.equal(api.create.seed, 4);
assert.equal(api.ready.engine, 'WASM');
assert.ok(frame.received >= 0);
new Uint8Array(buffer).fill(0);
assert.deepEqual(frame.view.fuel.pos, [1, 2, 3, 4, 5, 6]);
assert.equal(api.poll(1).frames.length, 0);
for (let i = 0; i < 125; i++) worker.receive({ t: 'frame', meta: meta(0), buf: new ArrayBuffer(0) });
assert.equal(api.frames.length, 120);
assert.equal(api.seq, 126);
assert.equal(api.poll(0).frames.length, 1);
assert.equal(api.poll(0, { latestOnly: false }).frames.length, 120);
assert.deepEqual(api.decode(meta(0), new ArrayBuffer(0)).fuel, { count: 0, pos: [], where: [] });
assert.throws(() => api.decode(meta(2), new ArrayBuffer(0)), /invalid frame/);
assert.throws(() => api.decode({ fuelCount: -1 }, buffer), /invalid frame/);
const pad = { axes: [0, -1, 0, 0], buttons: Array.from({ length: 17 }, () => ({ pressed: true })) };
const neutral = () => {
  assert.deepEqual(api.pad.axes, [0, 0, 0, 0]);
  assert.ok(api.pad.buttons.every(b => !b.pressed && b.value === 0 && !b.touched));
};
api.writePad(pad);
assert.equal(navigator.getGamepads()[0].mapping, 'standard');
assert.equal(api.pollCount, 1);
assert.equal(api.pad.buttons[3].pressed, false);
assert.equal(api.pad.buttons[6].value, 1);
const stamp = api.pad.timestamp;
api.writePad(pad);
assert.ok(api.pad.timestamp > stamp);
listeners.blur(); neutral();
api.writePad(pad); neutral();
listeners.focus(); api.writePad(pad);
assert.equal(api.pad.axes[1], -1);
document.hidden = true; documentListeners.visibilitychange(); neutral();
document.hidden = false; documentListeners.visibilitychange();
assert.throws(() => api.writePad({ axes: [], buttons: [] }), /invalid pad/); neutral();
worker.postMessage({ t: 'input', input: { fieldRelative: true, y: 1 } });
assert.equal(api.inputs[0].input.y, 1);
worker.receive({ t: 'frame', meta: { ...meta(0), phase: 'done' }, buf: new ArrayBuffer(0) });
assert.equal(api.ended, true); neutral();
api.writePad(pad); neutral();
worker.postMessage({ t: 'create', team: 254 });
worker.postMessage({ t: 'input', input: { fieldRelative: false } });
assert.equal(api.fieldRelativeError, true); neutral();
worker.receive({ t: 'frame', meta: meta(2), buf: new ArrayBuffer(0) });
assert.match(api.error, /invalid frame/); neutral();
console.log(JSON.stringify({ view: frame.view, empty: api.decode(meta(0), new ArrayBuffer(0)), result: 'live injection checks passed' }));
