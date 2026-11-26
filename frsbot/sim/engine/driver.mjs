// Line-for-line port of frsbot/bot/driver.py (and the planner helpers it calls). Python stays the reference;
// tests/test_driver_parity.py compares both on recorded states. Keep the two in sync.
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

export const HUB_CENTRES = { blue: [-3.645, 0.0, 1.829], red: [3.645, 0.0, 1.829] };
const HUBS = [HUB_CENTRES.blue, HUB_CENTRES.red];
export const FIELD_X = 8.270494, FIELD_Z = 4.034663, SCORING_X = 4.3, ZONE_X = 4.2418, FEED_X = 3.9;
const F32_02 = Math.fround(.2);
const TIMELINE = [[0, 'auto'], [20, 'gap'], [23, 'transition'], [33, 'shift1'], [58, 'shift2'], [83, 'shift3'],
  [108, 'shift4'], [133, 'endgame'], [163, 'settle'], [166, 'done']];
const SHIFT_STARTS = [33, 58, 83, 108];
export const GOAL_KINDS = ['Hold', 'Collect', 'Score', 'Stage', 'Feed', 'ZoneMow', 'Seed'];

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

// Shipped tuning defaults used by Driver (frsbot/bot/tuning.py DEFAULTS subset).
const PARAMS = { driveDistGain: 2.0, driveBrakeFrac: .4,
  collectNearFrac: .6, collectCosFloor: .05, mowConeMin: 99, collectCruiseFrac: .68, passFrac: .7,
  mowSpeedFrac: .35, intakeNeedsFuel: 0, approachFire: 1, freeFire: 0, freeFireSpeed: .8,
  preFireS: 2.0, postFireS: .35, scoreRadiusFrac: .5 };

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
  constructor(spec, profile = {}) { this.spec = spec; this.profile = profile; this.scoreOffset = 0; this.avoidSlots = []; }
  capacity(state) { return state.robots[state.me].capacity || this.spec.total_capacity; }
  bounds() { const d = this.spec.hub_shot.distance ?? []; return [this.spec.min_range || Math.min(...d), this.spec.max_range || Math.max(...d)]; }
  hub(state) { return HUB_CENTRES[state.robots[state.me].alliance].slice(0, 2); }
  hubOpen(state) {
    // Effective fire window: active, pre-fire lead before activation, and the
    // short post-grace (~0.3 s measured) after deactivation.
    const hub = state.hubs[state.robots[state.me].alliance];
    if (hub.active) return true;
    if (hub.secondsToToggle <= PARAMS.preFireS) return true;
    let start = null;
    for (const s of SHIFT_STARTS) if (s <= state.t) start = s;
    return start !== null && state.t - start <= PARAMS.postFireS;
  }
  scoreSpot(state, angleOffset = 0) {
    const robot = state.robots[state.me];
    const [cx, cz] = this.hub(state);
    const [low, high] = this.bounds();
    const radius = Math.max(low + .3, Math.min(high - .5, low + PARAMS.scoreRadiusFrac * (high - low)));
    const heading = Math.atan2(robot.z - cz, robot.x - cx) + angleOffset + this.scoreOffset;
    const margin = Math.hypot(this.spec.bumper_length, this.spec.bumper_width) / 2 + .1;
    // The robot body must clear the hub box (engine collider, +-0.597 m around
    // the hub centre) inflated by its half-diagonal; close spots stay legal by
    // flanking the hub in z, like the pros' parks.
    const body = margin + .05;
    let candidates = [];
    const deltas = [0]; for (let i = 1; i < 37; i++) for (const s of [-1, 1]) deltas.push(s * i * Math.PI / 36);
    for (const delta of deltas) {
      const x = cx + radius * Math.cos(heading + delta), z = cz + radius * Math.sin(heading + delta);
      const ownSide = robot.alliance === 'blue' ? x < -SCORING_X : x > SCORING_X;
      if (ownSide && Math.abs(x) <= FIELD_X - margin && Math.abs(z) <= FIELD_Z - margin
          && !(Math.abs(x - cx) < .597 + body && Math.abs(z - cz) < .597 + body)) candidates.push([x, z]);
    }
    if (this.profile.score_arc_degrees && this.profile.score_arc_degrees.length) {
      const back = robot.alliance === 'blue' ? Math.PI : 0;
      candidates = this.profile.score_arc_degrees.map(deg => [cx + radius * Math.cos(back + deg * (Math.PI / 180)), cz + radius * Math.sin(back + deg * (Math.PI / 180))]);
    }
    if (this.avoidSlots.length) {
      const free = candidates.filter(p => this.avoidSlots.every(slot => dist(p, slot) >= 1.5));
      if (free.length) candidates = free;
    }
    const preferred = [cx + radius * Math.cos(heading), cz + radius * Math.sin(heading)];
    return minBy(candidates, p => dist(p, preferred));
  }
  fuelPoints(state) {
    const pos = state.fuelPos, where = state.fuelWhere, out = [];
    for (let i = 0; i < where.length; i++) if (where[i] === 0 && pos[3 * i + 1] < F32_02) out.push([pos[3 * i], pos[3 * i + 2]]);
    return out;
  }
  inzoneFuel(state) {
    const robot = state.robots[state.me];
    return this.fuelPoints(state).filter(p => robot.alliance === 'blue' ? p[0] < -SCORING_X : p[0] > SCORING_X);
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
    this.abandon = false; this.shootWhileCollecting = profile.shoot_while_collecting ?? ((spec.max_shot_speed || 0) >= 2); this.reaimUntil = 0;
    this.stallT = null; this.pinT = null; this.unwedgeUntil = 0;
  }

  progress(goal, state) {
    const robot = state.robots[state.me];
    this.abandon = false;
    const scoring = goal.kind === 'Score' && this.planner.hubOpen(state) && robot.hopper > 0;
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
    // Engine geometry (module 2960), rotated (-x,-z) per alliance side:
    // trench posts x in |3.048..4.242|, |z| 2.451-2.756 (tall robots can't pass
    // the 0.565 m bar); tower x |7.12..8.3|, blue z -0.21..+0.79; depot x
    // |7.58..8.3|, blue z -2.46..-1.39. Bump ramps (|z| 0.597-2.451) are drivable.
    for (const s of [-1, 1]) {
      if (!this.spec.trench_cut) {
        const plo = Math.min(s * 3.048, s * 4.242), phi = Math.max(s * 3.048, s * 4.242);
        boxes.push([plo, phi, 2.40, 2.85], [plo, phi, -2.85, -2.40]);
      }
      const tlo = Math.min(s * 7.12, s * 8.35), thi = Math.max(s * 7.12, s * 8.35);
      const dlo = Math.min(s * 7.55, s * 8.35), dhi = Math.max(s * 7.55, s * 8.35);
      if (s === -1) boxes.push([tlo, thi, -.25, .83], [dlo, dhi, -2.5, -1.35]);
      else boxes.push([tlo, thi, -.83, .25], [dlo, dhi, 1.35, 2.5]);
    }
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
        if (j === index) continue;
        // Edges into the target may end inside the box containing it; every
        // other box still blocks.
        const blockers = j !== 1 ? boxes : boxes.filter(b => !(b[0] < target[0] && target[0] < b[1] && b[2] < target[1] && target[1] < b[3]));
        if (blockers.some(box => intersects(nodes[index], point, box))) continue;
        const newCost = cost + dist(nodes[index], point);
        if (newCost < (best.get(j) ?? Infinity)) { best.set(j, newCost); queue.push([newCost, j, [...route, j]]); }
      }
    }
    return start;
  }

  act(goalIn, state) {
    const goal = this.progress(goalIn, state);
    const robot = state.robots[state.me], spec = this.spec, planner = this.planner;
    const active = planner.hubOpen(state);
    const hub = planner.hub(state);
    const [low, high] = planner.bounds();
    const distanceHub = Math.hypot(robot.x - hub[0], robot.z - hub[1]);
    const shooting = goal.kind === 'Score' && active && robot.hopper > 0;
    const sign = robot.alliance === 'blue' ? 1 : -1;
    const ownSide = robot.alliance === 'blue' ? robot.x < -SCORING_X : robot.x > SCORING_X;
    const inZone = -sign * robot.x > ZONE_X - .34;  // bumper-edge legality (~|x| 3.9)
    const atSpot = dist([robot.x, robot.z], goal.point) < .7;
    const target = goal.point;
    let point;
    if (goal.kind === 'Score' && dist([robot.x, robot.z], target) < 2.0) {
      point = target;
    } else {
      point = this.waypoint([robot.x, robot.z], target);
    }
    let dx = point[0] - robot.x, dz = point[1] - robot.z;
    let distance = Math.hypot(dx, dz);
    let direction = Math.atan2(dz, dx);
    const points = planner.fuelPoints(state);
    let pathFuel = false, mowDir = null;
    if (points.length) {
      const c = Math.cos(robot.a), s = Math.sin(robot.a);
      let cone = 0, mx = 0, mz = 0;
      for (const [px, pz] of points) {
        const rx = px - robot.x, rz = pz - robot.z;
        const ahead = rx * c + rz * s, side = rx * -s + rz * c;
        if (ahead > 0 && ahead < 1.2 && Math.abs(side) < spec.bumper_width / 2) pathFuel = true;
        if (ahead > 0 && ahead < 1.6 && Math.abs(side) < spec.bumper_width * .6) { cone++; mx += rx; mz += rz; }
      }
      if (cone >= PARAMS.mowConeMin) mowDir = Math.atan2(mz, mx);
    }
    const collectKind = goal.kind === 'Collect';
    if (collectKind && mowDir !== null) {
      point = [robot.x + 2.5 * Math.cos(mowDir), robot.z + 2.5 * Math.sin(mowDir)];
      dx = point[0] - robot.x; dz = point[1] - robot.z;
      distance = Math.hypot(dx, dz);
      direction = mowDir;
    } else if (collectKind && distance < .6 && pathFuel) {
      point = [robot.x + 1.2 * Math.cos(robot.a), robot.z + 1.2 * Math.sin(robot.a)];
      dx = point[0] - robot.x; dz = point[1] - robot.z;
      distance = Math.hypot(dx, dz);
      direction = Math.atan2(dz, dx);
    }
    const movingShot = this.shootWhileCollecting && goal.kind === 'Collect' && active && ownSide && robot.hopper > 0
      && (spec.max_shot_speed || 0) >= 2 && low + .2 <= distanceHub && distanceHub <= high - .5;
    const arriving = distance < .3;
    const finalWaypoint = dist(point, target) < .01;
    const vmax = spec.max_speed;
    let brakeDist = distance;
    let speed = Math.min(vmax, PARAMS.driveDistGain * brakeDist,
      Math.sqrt(2 * spec.accel * PARAMS.driveBrakeFrac * Math.max(0, brakeDist - .12)));
    if (collectKind) {
      // Human-measured law: pros never brake for collect points and never
      // throttle for heading error — flat cruise pass-through (parity with driver.py).
      speed = vmax * PARAMS.collectCruiseFrac;
    }
    if (arriving && finalWaypoint) {
      speed = 0;
      if (collectKind) speed = vmax * PARAMS.collectCruiseFrac;  // never park at piles (parity)
    } else if (arriving) speed = Math.max(speed, .35);
    const scoreSpeed = spec.max_shot_speed || (spec.brain_equivalents?.scoreSpeedTol ?? .15);
    if (movingShot) speed = Math.min(speed, Math.min(scoreSpeed, spec.max_shot_speed * .6));
    if (shooting && finalWaypoint && low <= distanceHub && distanceHub <= high && distance < .6)
      speed = 0;  // stop dead so the engine's hold/firing/xLock engages
    const v = Math.hypot(robot.vx, robot.vz);
    if (speed > .8) {
      if (v < .18) {
        if (this.stallT === null) this.stallT = state.t;
        else if (state.t - this.stallT > 1.1) {
          this.unwedgeUntil = state.t + .9;
          this.stallT = null;
        }
      } else this.stallT = null;
    }
    // Mirrors driver.py: pinned with a distant target and low/zero commanded
    // speed — the gate above misses it (Score handles wedges via progress()).
    if (goal.kind !== 'Score' && distance > .5 && v < .15) {
      if (this.pinT === null) this.pinT = state.t;
      else if (state.t - this.pinT > 1.5) {
        this.unwedgeUntil = state.t + .9;
        this.pinT = null;
      }
    } else this.pinT = null;
    let vx, vz;
    if (state.t < this.unwedgeUntil) {
      direction = Math.atan2(-dz, -dx);
      speed = .9;
      vx = speed * Math.cos(direction); vz = speed * Math.sin(direction);
    } else {
      vx = speed * Math.cos(direction); vz = speed * Math.sin(direction);
    }
    const margin = Math.hypot(spec.bumper_length, spec.bumper_width) / 2 + .1;
    if (robot.x * vx > 0 && Math.abs(robot.x + vx * .3) > FIELD_X - margin) vx = 0;
    if (robot.z * vz > 0 && Math.abs(robot.z + vz * .3) > FIELD_Z - margin) vz = 0;
    const norm = Math.hypot(vx, vz) / vmax;
    const stick = inverseDeadzone(norm);
    let x = norm ? sign * stick * vz / Math.hypot(vx, vz) : 0;
    let y = norm ? sign * stick * vx / Math.hypot(vx, vz) : 0;
    // Shooter heading: rear shooters aim away (+pi), front shooters (yawDeg 180) at it.
    const shYaw = Math.PI - (spec.shooter_yaw_deg || 0) * Math.PI / 180;
    const aim = goal.kind !== 'Collect' ? Math.atan2(hub[1] - robot.z, hub[0] - robot.x) + shYaw : direction;
    let intake = robot.hopper < planner.capacity(state)
      && (pathFuel || (goal.kind === 'Collect' && !PARAMS.intakeNeedsFuel));
    const outtake = false;
    if (shooting && state.hubs[robot.alliance].active && state.hubs[robot.alliance].secondsToToggle <= planner.scoreTime(state) + 1)
      intake = false;
    if (shooting && this.recovering) intake = false;
    // Engine-enforced speed gate (scoreSpeedTol): pressing score early only
    // buys aim-translation slowdown, so approachFire skips those robots.
    const speedGated = spec.brain_equivalents?.scoreSpeedTol != null;
    let score;
    if (PARAMS.freeFire) {
      score = active && robot.hopper > 0 && ownSide && low <= distanceHub && distanceHub <= high && v <= PARAMS.freeFireSpeed;
      intake = robot.hopper < planner.capacity(state);
    } else {
      score = shooting && inZone && atSpot && low <= distanceHub && distanceHub <= high && v <= scoreSpeed;
      score = score || (movingShot && v <= Math.min(scoreSpeed, spec.max_shot_speed * .6));
      // Press score near the end of approach: spin-up latches on the command
      // without paying aim-translation slowdown for the whole leg.
      if (PARAMS.approachFire && !speedGated && shooting && inZone && low <= distanceHub && distanceHub <= high)
        score = score || distance < 1.6 || v <= scoreSpeed;
    }
    // Inactive-window passing (parity with driver.py): full hopper + hub shut
    // + out of range => brain auto-passes upfield.
    const hubT = state.hubs[robot.alliance]; const toggle = hubT.secondsToToggle ?? hubT.seconds_to_toggle ?? 99;
    if (!active && robot.hopper >= PARAMS.passFrac * planner.capacity(state)
        && toggle > 8 && distanceHub > high + .5 && (goal.kind === 'Collect' || goal.kind === 'Stage' || goal.kind === 'Hold'))
      score = true;
    let turn = inverseDeadzone(Math.max(-1, Math.min(1, -4 * angleError(aim, robot.a) / (.45 * spec.turn_rate))));
    // Mirrors driver.py: while score is pressed the engine brain sets
    // desiredRotation and t.turn is ignored — zeroed for clarity. Turret robots
    // keep chassis steering during Collect (SCORING_ON_THE_MOVE reads omega).
    const steersFree = spec.turret && goal.kind === 'Collect';
    if (score && !this.recovering && !steersFree) turn = 0;
    if (PARAMS.approachFire && !speedGated && shooting && ownSide && low <= distanceHub && distanceHub <= high && v <= scoreSpeed) turn = 0;
    if (shooting && this.recoveryStage === 1 && state.t < this.reaimUntil) { x = y = 0; score = false; }
    if (shooting && state.t < this.wiggleUntil) {
      const pulse = Math.trunc((this.wiggleUntil - state.t) / .3) % 2 ? 1 : -1;
      y = sign * inverseDeadzone(pulse / spec.max_speed);
      x = 0;
      turn = inverseDeadzone(copysign(.25, pulse));
      score = true;
    }
    return { x, y, turn, intake, score, outtake, climb: false, block: false, stow: false, hood: false, fieldRelative: true };
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
