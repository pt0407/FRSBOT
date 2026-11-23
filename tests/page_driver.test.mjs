import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const config = JSON.parse(fs.readFileSync(0, 'utf8'));
const events = {};
class Worker {
  constructor() { this.listeners = {}; }
  addEventListener(t, f) { this.listeners[t] = f; }
  postMessage() {}
  receive(data) { this.listeners.message({data}); }
}
globalThis.window = {Worker, addEventListener:(t,f)=>{events[t]=f;}};
Object.defineProperty(globalThis, 'navigator', {value:{}, configurable:true});
globalThis.document = {hidden:false, addEventListener() {}};
vm.runInThisContext(fs.readFileSync(new URL('../frsbot/live/inject.js', import.meta.url), 'utf8'));
vm.runInThisContext(config.source);
vm.runInThisContext(fs.readFileSync(new URL('../frsbot/live/page_driver.js', import.meta.url), 'utf8'));
const api = window.__imai, worker = new window.Worker();
api.installDriver({spec:config.spec, profile:config.profile, horizon:0});
const results = [];
for (const {view, goal} of config.calls) {
  worker.receive({t:'frame', meta:{...view, fuelCount:0}, buf:new ArrayBuffer(0)});
  api.setGoal({goal, horizon:0});
  navigator.getGamepads();
  assert.equal(api.error, null);
  assert.equal(api.pad.buttons[3].pressed, false);
  results.push(api.driverStatus);
}
const predicted = api.predictState(api.frames.at(-1), {x:.2,y:.7,turn:.3}, .1, config.spec);
api.setGoal({goal:null});
navigator.getGamepads();
assert.ok(api.pad.axes.every(x => x === 0));
events.blur();
api.setGoal({goal:config.calls[0].goal});
navigator.getGamepads();
assert.ok(api.pad.axes.every(x => x === 0));
console.log(JSON.stringify({results, predicted:{t:predicted.t, robots:predicted.robots, hubs:predicted.hubs}}));
