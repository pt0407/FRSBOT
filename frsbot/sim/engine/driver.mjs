// Line-for-line port of frsbot/bot/driver.py (and the planner helpers it calls). Python stays the reference;
// tests/test_driver_parity.py compares both on recorded states. Keep the two in sync.
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

export const HUB_CENTRES = { blue: [-3.645, 0.0, 1.829], red: [3.645, 0.0, 1.829] };
const HUBS = [HUB_CENTRES.blue, HUB_CENTRES.red];
export const FIELD_X = 8.270494, FIELD_Z = 4.034663, SCORING_X = 4.3;
const F32_02 = Math.fround(.2);
const TIMELINE = [[0, 'auto'], [20, 'gap'], [23, 'transition'], [33, 'shift1'], [58, 'shift2'], [83, 'shift3'],
  [108, 'shift4'], [133, 'endgame'], [163, 'settle'], [166, 'done']];
export const GOAL_KINDS = ['Hold', 'Collect', 'Score', 'Stage'];

// Python float semantics.
function pymod(x, m) { let r = x % m; if (r !== 0 && (m < 0) !== (r < 0)) r += m; return r === 0 ? 0 : r; }
export function angleError(target, actual) { return pymod(target - actual + Math.PI, 2 * Math.PI) - Math.PI; }
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const copysign = (m, s) => (s < 0 || Object.is(s, -0)) ? -Math.abs(m) : Math.abs(m);
function minBy(items, key) { let best, bestKey = Infinity, first = true; for (const item of items) { const k = key(item); if (first || k < bestKey) { best = item; bestKey = k; first = false; } } if (first) throw new Error('min() arg is an empty sequence'); return best; }

export function phaseAt(t) { let phase = 'countdown'; for (const [start, p] of TIMELINE) { if (t < start) break; phase = p; } return phase; }
export function isActive(alliance, t, firstInactive) {
  const phase = phaseAt(t);
  if (phase === 'auto' || phase === 'transition' || phase === 'endgame') return true;
  if (phase.startsWith('shift')) {
    if (firstInactive == null) throw new Error('first_inactive is required during shifts');
    const firstOff = phase === 'shift1' || phase === 'shift3';
    return firstOff ? alliance !== firstInactive : alliance === firstInactive;
  }
  return false;
}
export function secondsToToggle(alliance, t, firstInactive) {
  const active = isActive(alliance, t, firstInactive);
  for (const [start] of TIMELINE) if (start > t && isActive(alliance, start, firstInactive) !== active) return start - t;
  return Infinity;
}

// Same fields as frsbot.sim.wasm_env.view_to_state, copied so later ticks don't alias WASM memory.
export function viewToState(view, freeplay = false, me = null) {
  const t = view.phase === 'countdown' ? -Number(view.clock) : Number(view.elapsed);
  const robots = view.robots.map(raw => ({
    id: raw.id, team: raw.team, alliance: raw.alliance, x: raw.pos[0], z: raw.pos[2],
    a: Math.PI - 2 * Math.atan2(raw.quat[1], raw.quat[3]), vx: raw.vel[0], vz: raw.vel[2],
    qy: raw.quat[1], qw: raw.quat[3],
    hopper: raw.hopperFuel, capacity: raw.hopperCapacity, gate: raw.gate ?? null, mode: raw.state ?? null,
  }));
  const hubs = {};
  for (const a of ['blue', 'red']) hubs[a] = { active: !!view.hubs[a].active, secondsToToggle: freeplay ? Infinity : secondsToToggle(a, t, view.firstInactive) };
  const local = view.robots.findIndex(r => r.local);
  return { t, phase: view.phase, elapsed: view.elapsed, clock: view.clock, firstInactive: view.firstInactive ?? null,
    hubs, robots, fuelPos: Float32Array.from(view.fuel.pos), fuelWhere: Uint8Array.from(view.fuel.where),
    score: view.score, me: me ?? (local < 0 ? 0 : local) };
}

export function inverseDeadzone(value) { return value ? copysign(.12 + .88 * Math.min(1, Math.abs(value)), value) : 0.0; }

export function intersects(a, b, box) {
  let lower = 0.0, upper = 1.0;
  const ps = [[a[0], b[0], box[0], box[1]], [a[1], b[1], box[2], box[3]]];
  for (const [p, q, lo, hi] of ps) {
    const d = q - p;
    if (Math.abs(d) < 1e-9) { if (!(lo < p && p < hi)) return false; }
    else {
      let x = (lo - p) / d, y = (hi - p) / d;
      if (y < x) [x, y] = [y, x];
      lower = Math.max(lower, x); upper = Math.min(upper, y);
    }
  }
  return lower < upper && lower < 1 && upper > 0;
}

class Heap {
  constructor(less) { this.items = []; this.less = less; }
  get length() { return this.items.length; }
  push(v) { const h = this.items; h.push(v); let i = h.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (!this.less(h[i], h[p])) break; [h[i], h[p]] = [h[p], h[i]]; i = p; } }
  pop() { const h = this.items, top = h[0], last = h.pop(); if (h.length) { h[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < h.length && this.less(h[l], h[m])) m = l; if (r < h.length && this.less(h[r], h[m])) m = r; if (m === i) break; [h[i], h[m]] = [h[m], h[i]]; i = m; } } return top; }
}
function tupleLess(a, b) {
  if (a[0] !== b[0]) return a[0] < b[0];
  if (a[1] !== b[1]) return a[1] < b[1];
  const x = a[2], y = b[2];
  for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] < y[i];
  return x.length < y.length;
}

export class Planner {
  constructor(spec, profile = {}) { this.spec = spec; this.profile = profile; }
  capacity(state) { return state.robots[state.me].capacity || this.spec.total_capacity; }
  bounds() { const d = this.spec.hub_shot.distance ?? []; return [this.spec.min_range || Math.min(...d), this.spec.max_range || Math.max(...d)]; }
  hub(state) { return HUB_CENTRES[state.robots[state.me].alliance].slice(0, 2); }
  scoreSpot(state, angleOffset = 0) {
    const robot = state.robots[state.me];
    const [cx, cz] = this.hub(state);
    const [low, high] = this.bounds();
    const radius = Math.max(low + .3, Math.min(high - .5, (low + high) / 2));
    const heading = Math.atan2(robot.z - cz, robot.x - cx) + angleOffset;
    const margin = Math.hypot(this.spec.bumper_length, this.spec.bumper_width) / 2 + .1;
    let candidates = [];
    const deltas = [0]; for (let i = 1; i < 37; i++) for (const s of [-1, 1]) deltas.push(s * i * Math.PI / 36);
    for (const delta of deltas) {
      const x = cx + radius * Math.cos(heading + delta), z = cz + radius * Math.sin(heading + delta);
      const ownSide = robot.alliance === 'blue' ? x < -SCORING_X : x > SCORING_X;
      if (ownSide && Math.abs(x) <= FIELD_X - margin && Math.abs(z) <= FIELD_Z - margin) candidates.push([x, z]);
    }
    if (this.profile.score_arc_degrees && this.profile.score_arc_degrees.length) {
      const back = robot.alliance === 'blue' ? Math.PI : 0;
      candidates = this.profile.score_arc_degrees.map(deg => [cx + radius * Math.cos(back + deg * (Math.PI / 180)), cz + radius * Math.sin(back + deg * (Math.PI / 180))]);
    }
    const preferred = [cx + radius * Math.cos(heading), cz + radius * Math.sin(heading)];
    return minBy(candidates, p => dist(p, preferred));
  }
  fuelPoints(state) {
    const pos = state.fuelPos, where = state.fuelWhere, out = [];
    for (let i = 0; i < where.length; i++) if (where[i] === 0 && pos[3 * i + 1] < F32_02) out.push([pos[3 * i], pos[3 * i + 2]]);
    return out;
  }
  scoreTime(state) {
    const robot = state.robots[state.me];
    const target = this.scoreSpot(state), position = [robot.x, robot.z];
    let distance = dist(position, target);
    for (const [cx] of HUBS) {
      if ((position[0] - cx) * (target[0] - cx) < 0) {
        const side = position[0] < cx ? -1 : 1;
        const lane = minBy([-1.8, 1.8], z => Math.abs(position[1] - z) + Math.abs(target[1] - z));
        const entry = [cx + side * 1.65, lane], exit = [cx - side * 1.65, lane];
        distance = dist(position, entry) + dist(entry, exit) + dist(exit, target);
      }
    }
    let travel = distance / this.spec.max_speed;
    if (distance > .3) travel += .5;
    return travel + (this.profile.startup_seconds ?? 1.5) + robot.hopper / (this.profile.drain_rate ?? this.spec.fire_rate);
  }
}

export class Driver {
  constructor(spec, profile = {}) {
    this.spec = spec;
    this.planner = new Planner(spec, profile);
    this.recovering = false; this.crossing = null; this.progressAt = null; this.openSince = null;
    this.previousHopper = null; this.recoveryStage = 0; this.recoveryPoint = null; this.wiggleUntil = 0;
    this.abandon = false; this.shootWhileCollecting = profile.shoot_while_collecting ?? false; this.reaimUntil = 0;
  }

  progress(goal, state) {
    const robot = state.robots[state.me];
    this.abandon = false;
    const scoring = goal.kind === 'Score' && state.hubs[robot.alliance].active && robot.hopper > 0;
    if (!scoring) {
      this.progressAt = null; this.openSince = null; this.previousHopper = robot.hopper;
      this.recoveryStage = 0; this.recoveryPoint = null; this.recovering = false;
      return goal;
    }
    let point = this.recoveryPoint ?? goal.point;
    if (this.previousHopper === null || robot.hopper < this.previousHopper) { this.progressAt = state.t; this.openSince = null; this.recoveryStage = 0; }
    this.previousHopper = robot.hopper;
    if (dist([robot.x, robot.z], point) > .7) this.progressAt = state.t;
    if (this.progressAt === null) this.progressAt = state.t;
    const opened = robot.gate != null && !!(robot.gate.open ?? false);
    if (!opened) this.openSince = null;
    else if (this.openSince === null) this.openSince = state.t;
    const stalled = state.t - this.progressAt >= 2 || (this.openSince !== null && state.t - Math.max(this.openSince, this.progressAt) >= 1);
    if (state.t >= this.wiggleUntil && stalled) {
      this.recoveryStage += 1;
      this.progressAt = state.t;
      if (this.recoveryStage === 1) { this.reaimUntil = state.t + .6; this.progressAt = this.reaimUntil; }
      else if (this.recoveryStage === 2) {
        point = this.planner.scoreSpot(state, Math.PI / 6);
        if (dist(point, [robot.x, robot.z]) < .5) point = this.planner.scoreSpot(state, -Math.PI / 6);
        this.recoveryPoint = point; this.crossing = null;
      } else if (this.recoveryStage === 3) this.wiggleUntil = state.t + 1.2;
      else if (this.recoveryStage >= 4) {
        if (robot.hopper <= 5) this.abandon = true;
        else { this.recoveryPoint = this.planner.scoreSpot(state, -Math.PI / 4); this.crossing = null; }
        this.recoveryStage = 1;
      }
    }
    this.recovering = this.recoveryStage > 0;
    return { kind: 'Score', point: this.recoveryPoint ?? goal.point };
  }

  waypoint(start, target) {
    if (this.crossing === null) {
      for (const [cx] of HUBS) {
        if ((start[0] - cx) * (target[0] - cx) < 0) {
          const side = start[0] < cx ? -1 : 1;
          const lane = minBy([-1.8, 1.8], z => Math.abs(start[1] - z) + Math.abs(target[1] - z));
          this.crossing = [[cx + side * 1.65, lane], [cx - side * 1.65, lane]];
          break;
        }
      }
    }
    if (this.crossing && this.crossing.length) {
      if (dist(start, this.crossing[0]) < .4) this.crossing.shift();
      if (this.crossing.length) return this.crossing[0];
      this.crossing = null;
    }
    const pad = Math.hypot(this.spec.bumper_length, this.spec.bumper_width) / 2 + .12;
    const boxes = HUBS.map(([cx, cz]) => [cx - .597 - pad, cx + .597 + pad, cz - .597 - pad, cz + .597 + pad]);
    for (const [left, right, bottom, top] of boxes) {
      if (left < start[0] && start[0] < right && bottom < start[1] && start[1] < top)
        return minBy([[left - .08, start[1]], [right + .08, start[1]], [start[0], bottom - .08], [start[0], top + .08]], p => dist(start, p));
    }
    const nodes = [start, target];
    for (const [left, right, bottom, top] of boxes) for (const x of [left - .05, right + .05]) for (const z of [bottom - .05, top + .05]) nodes.push([x, z]);
    const queue = new Heap(tupleLess); queue.push([0, 0, []]);
    const best = new Map([[0, 0]]);
    while (queue.length) {
      const [cost, index, route] = queue.pop();
      if (index === 1) return route.length ? nodes[route[0]] : target;
      if (cost > (best.get(index) ?? Infinity)) continue;
      for (let j = 0; j < nodes.length; j++) {
        const point = nodes[j];
        if (j === index || boxes.some(box => intersects(nodes[index], point, box))) continue;
        const newCost = cost + dist(nodes[index], point);
        if (newCost < (best.get(j) ?? Infinity)) { best.set(j, newCost); queue.push([newCost, j, [...route, j]]); }
      }
    }
    return start;
  }

  act(goalIn, state) {
    const goal = this.progress(goalIn, state);
    const robot = state.robots[state.me], spec = this.spec, planner = this.planner;
    const active = state.hubs[robot.alliance].active;
    const hub = planner.hub(state);
    const [low, high] = planner.bounds();
    const distanceHub = Math.hypot(robot.x - hub[0], robot.z - hub[1]);
    const shooting = goal.kind === 'Score' && active && robot.hopper > 0;
    const ownSide = robot.alliance === 'blue' ? robot.x < -SCORING_X : robot.x > SCORING_X;
    const atSpot = dist([robot.x, robot.z], goal.point) < .7;
    const target = goal.point;
    const point = this.waypoint([robot.x, robot.z], target);
    const dx = point[0] - robot.x, dz = point[1] - robot.z;
    const distance = Math.hypot(dx, dz);
    const direction = Math.atan2(dz, dx);
    const movingShot = this.shootWhileCollecting && goal.kind === 'Collect' && active && ownSide && robot.hopper > 0
      && (spec.max_shot_speed || 0) >= 2 && low + .2 <= distanceHub && distanceHub <= high - .5;
    const arriving = distance < .3;
    const finalWaypoint = dist(point, target) < .01;
    const vmax = spec.max_speed;
    let speed = Math.min(vmax, 2 * distance, Math.sqrt(2 * spec.accel * .4 * Math.max(0, distance - .12)));
    if (goal.kind === 'Collect') {
      if (distance < 1) speed = Math.min(speed, vmax * .6);
      speed *= Math.max(.05, Math.cos(angleError(direction, robot.a)));
    }
    if (arriving && finalWaypoint) speed = 0;
    else if (arriving) speed = Math.max(speed, .35);
    const scoreSpeed = spec.max_shot_speed || (spec.brain_equivalents?.scoreSpeedTol ?? .15);
    if (movingShot) speed = Math.min(speed, scoreSpeed * .85);
    if (shooting && finalWaypoint && low <= distanceHub && distanceHub <= high && distance < .6) speed = Math.min(speed, scoreSpeed * .5);
    let vx = speed * Math.cos(direction), vz = speed * Math.sin(direction);
    const margin = Math.hypot(spec.bumper_length, spec.bumper_width) / 2 + .1;
    if (robot.x * vx > 0 && Math.abs(robot.x + vx * .3) > FIELD_X - margin) vx = 0;
    if (robot.z * vz > 0 && Math.abs(robot.z + vz * .3) > FIELD_Z - margin) vz = 0;
    const norm = Math.hypot(vx, vz) / vmax;
    const stick = inverseDeadzone(norm);
    const sign = robot.alliance === 'blue' ? 1 : -1;
    let x = norm ? sign * stick * vz / Math.hypot(vx, vz) : 0;
    let y = norm ? sign * stick * vx / Math.hypot(vx, vz) : 0;
    const aim = goal.kind !== 'Collect' ? Math.atan2(hub[1] - robot.z, hub[0] - robot.x) + Math.PI : direction;
    let turn = inverseDeadzone(Math.max(-1, Math.min(1, -4 * angleError(aim, robot.a) / spec.turn_rate)));
    if (shooting && !this.recovering) turn = 0;
    const points = planner.fuelPoints(state);
    let pathFuel = false;
    if (points.length) {
      const c = Math.cos(robot.a), s = Math.sin(robot.a), half = spec.bumper_width / 2;
      for (const [px, pz] of points) {
        const rx = px - robot.x, rz = pz - robot.z;
        const ahead = rx * c + rz * s, side = rx * -s + rz * c;
        if (ahead > 0 && ahead < 1.2 && Math.abs(side) < half) { pathFuel = true; break; }
      }
    }
    let intake = robot.hopper < planner.capacity(state) && (goal.kind === 'Collect' || pathFuel);
    if (shooting && state.hubs[robot.alliance].secondsToToggle <= planner.scoreTime(state) + 1) intake = false;
    if (shooting && this.recovering) intake = false;
    const v = Math.hypot(robot.vx, robot.vz);
    let score = shooting && ownSide && atSpot && low <= distanceHub && distanceHub <= high && v <= scoreSpeed;
    score = score || (movingShot && v <= scoreSpeed);
    if (shooting && this.recoveryStage === 1 && state.t < this.reaimUntil) { x = y = 0; score = false; }
    if (shooting && state.t < this.wiggleUntil) {
      const pulse = Math.trunc((this.wiggleUntil - state.t) / .3) % 2 ? 1 : -1;
      y = sign * inverseDeadzone(pulse / spec.max_speed);
      x = 0;
      turn = inverseDeadzone(copysign(.25, pulse));
      score = true;
    }
    return { x, y, turn, intake, score, outtake: false, climb: false, block: false, stow: false, hood: false, fieldRelative: true };
  }
}

// CLI used by the parity test: first line {spec, profile, shootWhileCollecting?}; then {goal, state} per line.
function stateFromJson(raw) {
  const hubs = {};
  for (const [a, h] of Object.entries(raw.hubs)) hubs[a] = { active: h.active, secondsToToggle: h.seconds_to_toggle ?? Infinity };
  return { t: raw.t, hubs, robots: raw.robots, me: raw.me, fuelPos: Float32Array.from(raw.fuel_pos), fuelWhere: Uint8Array.from(raw.fuel_where) };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  let driver = null;
  for await (const line of lines) {
    const msg = JSON.parse(line);
    if (msg.spec) {
      driver = new Driver(msg.spec, msg.profile ?? {});
      if (msg.shootWhileCollecting != null) driver.shootWhileCollecting = msg.shootWhileCollecting;
      process.stdout.write('{"ok":true}\n');
      continue;
    }
    const inputs = driver.act({ kind: msg.goal.kind, point: msg.goal.point }, stateFromJson(msg.state));
    process.stdout.write(JSON.stringify({ ...inputs, abandon: driver.abandon, recovering: driver.recovering, stage: driver.recoveryStage }) + '\n');
  }
}
