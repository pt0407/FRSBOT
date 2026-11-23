import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

export function loadBundle(bundlePath) {
  const context = { TURBOPACK: [] };
  vm.runInNewContext(fs.readFileSync(bundlePath, 'utf8'), context);
  const modules = new Map(), cache = new Map(), executed = new Set();
  for (const registration of context.TURBOPACK) {
    let ids = [];
    for (const value of registration.slice(1)) {
      if (typeof value === 'number') ids.push(value);
      else if (typeof value === 'function') {
        for (const id of ids) modules.set(id, value);
        ids = [];
      }
    }
  }
  function load(id) {
    if (!cache.has(id)) cache.set(id, {});
    const fn = modules.get(id);
    if (!fn) throw new Error(`missing bundle module ${id}`);
    if (!executed.has(fn)) {
      executed.add(fn);
      fn({
        i: load,
        s(exports, target = id) {
          if (!cache.has(target)) cache.set(target, {});
          for (let j = 0; j < exports.length;) {
            const key = exports[j++], type = exports[j++];
            Object.defineProperty(cache.get(target), key, type === 0
              ? { value: exports[j++], enumerable: true, configurable: true }
              : { get: type, enumerable: true, configurable: true });
          }
        },
      });
    }
    return cache.get(id);
  }
  for (const [id, fn] of modules) {
    if (fn.toString().includes('brain:{')) load(id);
  }
  const robots = new Map();
  for (const exports of cache.values()) {
    for (const robot of Object.values(exports)) {
      if (robot?.brain) robots.set(robot.number, robot);
    }
  }
  return { robots, load };
}

export function enrich(records, bundle) {
  return records.map(record => {
    const robot = bundle.robots.get(record.team);
    assert.ok(robot, `no bundle robot for team ${record.team}`);
    assert.equal(robot.robotName, record.robotName);
    const brain = {};
    for (const key of ['kind', 'minRange', 'maxRange', 'headingToleranceDeg', 'maxShotSpeed', 'hubShot', 'autos']) {
      if (robot.brain[key] !== undefined) brain[key] = robot.brain[key];
    }
    const result = {
      ...record, brain, fireRate: robot.shooter.fireRate, intakeRate: robot.intake.rate,
      hopperStorageCapacity: robot.hopper.capacity,
      feedLanes: robot.feed.lanes.length, feedPlaces: robot.feed.places.length,
      totalCapacity: robot.hopper.capacity + robot.feed.lanes.length * robot.feed.places.length,
    };
    if (brain.kind === 'reblitz') {
      const config = bundle.load(1721);
      result.brainEquivalents = {
        sourceModule: 1721,
        minRange: config.E.minShootDistance,
        headingToleranceDeg: config.E.scoreHeadingTol * 180 / Math.PI,
        hubShot: config.z,
        flightTime: config.S,
        scoreSpeedTol: config.E.scoreSpeedTol,
        autoStationaryHeadingToleranceDeg: config.E.autoStationaryHeadingTol * 180 / Math.PI,
      };
    }
    return result;
  });
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const args = process.argv.slice(2).filter(arg => arg !== '--check');
  const bundlePath = args[0] ?? path.join(os.homedir(), 'game/2c3zgga2f0b2g.js');
  const specsPath = args[1] ?? fileURLToPath(new URL('../../data/robot_feel.json', import.meta.url));
  const records = JSON.parse(fs.readFileSync(specsPath, 'utf8'));
  const result = enrich(records, loadBundle(bundlePath));
  const serialized = JSON.stringify(result, null, 1) + '\n';
  if (process.argv.includes('--check')) assert.deepEqual(records, JSON.parse(serialized));
  else fs.writeFileSync(specsPath, serialized);
  console.log(`${process.argv.includes('--check') ? 'verified' : 'extracted'} ${result.length} robot shooting/intake configurations from ${bundlePath}`);
}
