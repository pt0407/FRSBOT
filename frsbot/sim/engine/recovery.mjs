// Offline replay reconstruction against a fresh WASM match for every candidate.
const clamp = (v, lo = -1, hi = 1) => Math.max(lo, Math.min(hi, v));
const angle = (target, actual) => Math.atan2(Math.sin(target - actual), Math.cos(target - actual));
const heading = q => Math.PI - 2 * Math.atan2(q[1], q[3]);
const stick = v => v ? Math.sign(v) * (.12 + .88 * Math.min(Math.abs(v), 1)) : 0;

class ReplayMatch {
  constructor(adapter, segment) {
    this.engine = adapter.engine;
    this.x = this.engine.exports;
    const teams = new Map(segment.robots.map(r => [r.team, adapter.teamByNumber(r.team)]));
    const robots = segment.robots.map((r, i) => ({
      id: r.id, team: r.team, alliance: r.alliance, start: r.start,
      preload: 0, local: i === 0,
    }));
    const config = {
      robots, seed: segment.seed, auto: 'none', event: 'event', solo: robots.length === 1,
      hpThrows: false, hpUnloads: false, freeplay: true, fuelCount: segment.initialFuel.where.length,
      rules: { preload: 0, mode: 'freeplay' },
    };
    this.p = this.engine.withText([JSON.stringify([...teams.values()]), JSON.stringify(config)], (ptr, len) =>
      this.x.fo_new(this.engine.matchData(), ptr[0], len[0], ptr[1], len[1]));
    if (!this.p) throw new Error(this.engine.error());
    this.at = this.x.fx_alloc(112);
    this.robots = robots.length;
    try {
      this.check(this.x.fs_start(this.p, 0));
      this.restore(segment, robots.map(r => teams.get(r.team).intake));
    } catch (error) {
      this.close();
      throw new Error(`reset: ${error.message}`, { cause: error });
    }
  }

  check(status) {
    if (status < 0) throw new Error(this.engine.error());
    return status;
  }

  setBody(i, { pos, quat, vel, omega }) {
    [pos, quat, vel, [0, omega, 0]].forEach((v, k) =>
      this.check(this.x.fo_body_set(this.p, i, k, v[0], v[1], v[2], v[3] ?? 0)));
  }

  setBall(i, mode, a, b, c, wake = true) {
    this.check(this.x.fo_ball_set(this.p, i, mode, a, b, c, +wake));
  }

  inside(i) {
    const n = this.check(this.x.fo_inside(this.p, i));
    return new Set(new Uint16Array(this.x.memory.buffer, this.x.fo_inside_at(this.p, i), n));
  }

  primeCarried(owners, intakes) {
    const queues = intakes.map((_intake, i) => owners.flatMap((owner, id) => owner === i ? [id] : []));
    if (!queues.some(q => q.length)) return new Set();
    const pads = intakes.map((_intake, i) => [-6 + 6 * (i % 3), 0, i < 3 ? -2.6 : 2.6]);
    pads.forEach((pos, i) => this.setBody(i, { pos, quat: [0, 0, 0, 1], vel: [0, 0, 0], omega: 0 }));
    const commands = queues.map(q => [0, 0, 0, Boolean(q.length), false]);
    const budget = Math.max(...queues.map((q, i) => Math.ceil(60 * q.length / intakes[i].rate))) + 45;
    let held;
    for (let tick = 0; tick < budget; tick += 3) {
      held = queues.map((_q, i) => this.inside(i));
      if (queues.every((q, i) => q.every(id => held[i].has(id)))) break;
      queues.forEach((q, i) => {
        const { sides, zone } = intakes[i];
        const [cx, , cz] = pads[i];
        const depth = (zone.x0 + zone.x1) / 2;
        const offered = q.filter(id => !held[i].has(id)).slice(0, 3);
        offered.forEach((id, lane) => {
          this.check(this.x.fo_ball_on(this.p, id, 1));
          const across = (lane - 1) * zone.half * .65;
          this.setBall(id, 0, cx + (sides ? across : depth), .079, cz + (sides ? -depth : across));
          this.setBall(id, 1, 0, 0, 0);
        });
      });
      this.step(commands, 3);
    }
    return new Set(queues.flatMap((_q, i) => [...this.inside(i)]));
  }

  restore(segment, intakes) {
    const { pos, vel, where, owners } = segment.initialFuel;
    if (owners?.some(owner => owner >= 0)) {
      for (let i = 0; i < where.length; i++) this.check(this.x.fo_ball_on(this.p, i, 0));
    }
    const held = owners ? this.primeCarried(owners, intakes) : new Set();
    segment.frames[0].bodies.forEach((body, i) => this.setBody(i, body));
    for (let i = 0; i < where.length; i++) {
      const active = where[i] <= 2;
      if (!held.has(i)) this.check(this.x.fo_ball_on(this.p, i, +active));
      if (!active) continue;
      const j = i * 3;
      this.setBall(i, 0, pos[j], pos[j + 1], pos[j + 2], !held.has(i));
      this.setBall(i, 1, vel[j], vel[j + 1], vel[j + 2], !held.has(i));
    }
  }

  bodies() {
    return Array.from({ length: this.robots }, (_, i) => {
      this.check(this.x.fo_body(this.p, i, this.at));
      const r = new Float64Array(this.x.memory.buffer, this.at, 13);
      return { pos: Array.from(r.slice(0, 3)), quat: Array.from(r.slice(3, 7)),
        vel: Array.from(r.slice(7, 10)), omega: r[11] };
    });
  }

  view(n) {
    const length = this.check(this.x.fo_frame(this.p));
    const info = JSON.parse(this.engine.decode(this.x.fs_out(this.p), length));
    const buffer = this.x.memory.buffer;
    return {
      hoppers: info.robots.map(r => r.hopperFuel),
      where: Uint8Array.from(new Uint8Array(buffer, this.x.fs_where(this.p), n)),
      pos: Float32Array.from(new Float32Array(buffer, this.x.fs_array(this.p, 0, 0), 3 * n)),
    };
  }

  step(commands, ticks) {
    const input = new Float64Array(this.x.memory.buffer, this.x.fs_input(this.p), this.robots * 11);
    commands.forEach(([x, y, turn, intake, score], i) =>
      input.set([x, y, turn, +intake, +score, 0, 0, 0, 0, 1, 0], i * 11));
    for (let k = 0; k < ticks; k++) this.check(this.x.fs_step(this.p));
  }

  close() {
    this.x.fx_free(this.at, 112);
    this.x.fs_drop(this.p);
  }
}

function command(robot, body, from, to, dt, gain, buttons) {
  // Solve around the recorded velocity first, then correct the engine's rolling pose error.
  const vx = (to.pos[0] - from.pos[0]) / dt + gain * (from.pos[0] - body.pos[0]) +
    .22 * (to.vel[0] - body.vel[0]);
  const vz = (to.pos[2] - from.pos[2]) / dt + gain * (from.pos[2] - body.pos[2]) +
    .22 * (to.vel[2] - body.vel[2]);
  const speed = Math.hypot(vx, vz);
  const scale = speed ? (robot.alliance === 'blue' ? 1 : -1) * stick(speed / robot.speed) / speed : 0;
  const turnRate = angle(heading(to.quat), heading(from.quat)) / dt +
    gain * angle(heading(from.quat), heading(body.quat));
  return [vz * scale, vx * scale, stick(-turnRate / robot.turnRate), ...buttons];
}

function fuelResidual(actual, expected) {
  let where = 0, match = 0;
  for (let i = 0; i < expected.where.length; i++) {
    if (actual.where[i] !== expected.where[i]) continue;
    where++;
    const j = i * 3;
    if (Math.hypot(actual.pos[j] - expected.pos[j], actual.pos[j + 1] - expected.pos[j + 1],
                  actual.pos[j + 2] - expected.pos[j + 2]) <= .1) match++;
  }
  return [where / expected.where.length, match / expected.where.length];
}

function rollout(adapter, segment, gain, mode, fixed = null) {
  const match = new ReplayMatch(adapter, segment);
  try {
    const frames = segment.frames;
    const fuelCount = segment.initialFuel.where.length;
    const controls = [], residuals = [];
    let loss = 0;
    let view = match.view(fuelCount);
    const initialHoppers = view.hoppers;
    for (let k = 0; k < frames.length - 1; k++) {
      const a = frames[k], b = frames[k + 1];
      const ticks = b.tick - a.tick;
      const next = fixed?.[k] ?? match.bodies().map((body, i) => {
        const hint = a.buttons[i];
        const buttons = mode === 'track' ? [
          hint[0] && view.hoppers[i] <= b.bodies[i].hopper,
          hint[1] && view.hoppers[i] >= b.bodies[i].hopper,
        ] : [mode === 'intake' || mode !== 'drive' && hint[0],
             mode === 'hint' && hint[1]];
        return command(segment.robots[i], body, a.bodies[i], b.bodies[i], ticks / 60, gain, buttons);
      });
      try {
        match.step(next, ticks);
      } catch (error) {
        throw new Error(`step at tick ${a.tick}: ${error.message}`, { cause: error });
      }
      const bodies = match.bodies();
      view = match.view(fuelCount);
      const [fuelWhere, fuelMatch] = fuelResidual(view, b.fuel);
      const interval = bodies.map(({ pos, quat, vel }, i) => {
        const want = b.bodies[i];
        const pose = Math.hypot(pos[0] - want.pos[0], pos[2] - want.pos[2]);
        const yaw = Math.abs(angle(heading(quat), heading(want.quat)));
        const velocity = Math.hypot(vel[0] - want.vel[0], vel[2] - want.vel[2]);
        const hopper = Math.abs(view.hoppers[i] - want.hopper);
        loss += (pose ** 2 + .035 * yaw ** 2 + .004 * velocity ** 2 + .0004 * hopper ** 2 +
          .03 * (1 - fuelMatch)) * (k === frames.length - 2 ? 3 : 1);
        return [pose, yaw, velocity, hopper, fuelWhere, fuelMatch];
      });
      controls.push(next);
      residuals.push(interval);
    }
    return { loss, controls, residuals, initialHoppers };
  } finally {
    match.close();
  }
}

export function recoverSegment(adapter, segment) {
  let best, rejection;
  const candidates = [[5, 'hint'], [10, 'hint'], [15, 'drive'], [10, 'intake'], [10, 'track']];
  for (const [gain, mode] of candidates) {
    let candidate;
    try {
      candidate = rollout(adapter, segment, gain, mode);
    } catch (error) {
      rejection = error;
      continue;
    }
    if (!best || candidate.loss < best.loss) best = candidate;
  }
  if (!best) throw new Error(`no recoverable candidate: ${rejection.message}`);
  if (segment.contacts?.every(flag => !flag) && best.residuals.at(-1).some(row => row[0] >= .1)) {
    for (const mode of ['hint', 'track']) {
      try {
        const candidate = rollout(adapter, segment, 24, mode);
        if (candidate.loss < best.loss) best = candidate;
      } catch {}
    }
  }
  // Validate with fixed controls from the identical reset, without reading intermediate pose for control.
  return rollout(adapter, segment, 0, 'drive', best.controls);
}
