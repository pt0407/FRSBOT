// Offline replay reconstruction against a fresh WASM match for every candidate.
const clamp = (v, lo = -1, hi = 1) => Math.max(lo, Math.min(hi, v));
const angle = (target, actual) => Math.atan2(Math.sin(target - actual), Math.cos(target - actual));
const heading = q => Math.PI - 2 * Math.atan2(q[1], q[3]);
const stick = v => v ? Math.sign(v) * (.12 + .88 * Math.min(Math.abs(v), 1)) : 0;

class ReplayMatch {
  constructor(adapter, segment, preloads) {
    this.engine = adapter.engine;
    this.x = this.engine.exports;
    const teams = [...new Set(segment.robots.map(r => r.team))].map(adapter.teamByNumber);
    const robots = segment.robots.map((r, i) => ({
      id: r.id, team: r.team, alliance: r.alliance, start: r.start,
      preload: preloads[i], local: i === 0,
    }));
    const config = {
      robots, seed: segment.seed, auto: 'none', event: 'event', solo: robots.length === 1,
      hpThrows: false, hpUnloads: false, freeplay: true, fuelCount: segment.initialFuel.where.length,
      rules: { preload: 'full', mode: 'freeplay' },
    };
    this.p = this.engine.withText([JSON.stringify(teams), JSON.stringify(config)], (ptr, len) =>
      this.x.fo_new(this.engine.matchData(), ptr[0], len[0], ptr[1], len[1]));
    if (!this.p) throw new Error(this.engine.error());
    this.at = this.x.fx_alloc(112);
    this.robots = robots.length;
    try {
      this.check(this.x.fs_start(this.p, 0));
      this.restore(segment);
    } catch (error) {
      this.close();
      throw new Error(`reset: ${error.message}`, { cause: error });
    }
  }

  check(status) {
    if (status < 0) throw new Error(this.engine.error());
    return status;
  }

  restore(segment) {
    const before = this.bodies();
    const { pos, vel, where } = segment.initialFuel;
    const staged = this.view(where.length);
    this.indices = Array.from(where.keys());
    const available = new Set(this.indices.filter(i => where[i] === 2));
    for (let i = 0; i < where.length; i++) {
      if (staged.where[i] !== 2) continue;
      const j = i * 3;
      const robot = before.map(r => Math.hypot(staged.pos[j] - r.pos[0], staged.pos[j + 2] - r.pos[2]))
        .reduce((best, dist, index, all) => dist < all[best] ? index : best, 0);
      const { pos: centre } = segment.frames[0].bodies[robot];
      let nearest = -1, distance = 1.3;
      if (available.has(i) && Math.hypot(pos[j] - centre[0], pos[j + 2] - centre[2]) < distance) {
        nearest = i;
      } else {
        for (const index of available) {
          const d = Math.hypot(pos[index * 3] - centre[0], pos[index * 3 + 2] - centre[2]);
          if (d < distance) [nearest, distance] = [index, d];
        }
      }
      if (nearest >= 0) {
        const current = this.indices.indexOf(nearest);
        [this.indices[i], this.indices[current]] = [nearest, this.indices[i]];
        available.delete(nearest);
      }
    }
    segment.frames[0].bodies.forEach(({ pos, quat, vel, omega }, i) => {
      [pos, quat, vel, [0, omega, 0]].forEach((v, k) =>
        this.check(this.x.fo_body_set(this.p, i, k, v[0], v[1], v[2], v[3] ?? 0)));
    });
    for (let i = 0; i < where.length; i++) {
      // Engine-owned preload balls should stay attached to their robot.
      if (staged.where[i] === 2) continue;
      const index = this.indices[i];
      const active = where[index] <= 1;
      this.check(this.x.fo_ball_on(this.p, i, +active));
      if (!active) continue;
      const j = index * 3;
      this.check(this.x.fo_ball_set(this.p, i, 0, pos[j], pos[j + 1], pos[j + 2], 1));
      this.check(this.x.fo_ball_set(this.p, i, 1, vel[j], vel[j + 1], vel[j + 2], 1));
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

function fuelResidual(actual, expected, indices) {
  let where = 0, match = 0;
  for (let i = 0; i < indices.length; i++) {
    const mapped = indices[i];
    if (actual.where[i] !== expected.where[mapped]) continue;
    where++;
    const j = i * 3, k = mapped * 3;
    if (Math.hypot(actual.pos[j] - expected.pos[k], actual.pos[j + 1] - expected.pos[k + 1],
                  actual.pos[j + 2] - expected.pos[k + 2]) <= .1) match++;
  }
  return [where / expected.where.length, match / expected.where.length];
}

function rollout(adapter, segment, preloads, gain, mode, fixed = null) {
  const match = new ReplayMatch(adapter, segment, preloads);
  try {
    const frames = segment.frames;
    const fuelCount = segment.initialFuel.where.length;
    const controls = [], residuals = [];
    let loss = 0;
    const initialHoppers = match.view(fuelCount).hoppers;
    for (let k = 0; k < frames.length - 1; k++) {
      const a = frames[k], b = frames[k + 1];
      const ticks = b.tick - a.tick;
      const next = fixed?.[k] ?? match.bodies().map((body, i) => {
        const hint = a.buttons[i];
        const buttons = [mode === 'intake' || mode !== 'drive' && hint[0],
                         mode === 'hint' && hint[1]];
        return command(segment.robots[i], body, a.bodies[i], b.bodies[i], ticks / 60, gain, buttons);
      });
      try {
        match.step(next, ticks);
      } catch (error) {
        throw new Error(`step at tick ${a.tick}: ${error.message}`, { cause: error });
      }
      const bodies = match.bodies(), view = match.view(fuelCount);
      const [fuelWhere, fuelMatch] = fuelResidual(view, b.fuel, match.indices);
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
  const wanted = segment.frames[0].bodies.map(r => r.hopper);
  let preloads = wanted;
  for (const [gain, mode] of [[5, 'hint'], [10, 'hint'], [15, 'drive'], [10, 'intake']]) {
    let candidate;
    try {
      candidate = rollout(adapter, segment, preloads, gain, mode);
    } catch (error) {
      rejection = error;
      // A replay may start with a hopper configuration not supported by preload seats.
      preloads = preloads.map(n => Math.min(24, n));
      continue;
    }
    if (!best || candidate.loss < best.loss) best = { ...candidate, preloads };
    if (preloads === wanted) {
      // Feed seats may absorb part of the requested preload before it reaches storage.
      preloads = wanted.map((n, i) => n + clamp(n - candidate.initialHoppers[i], 0, 8));
    }
  }
  if (!best) throw new Error(`no recoverable candidate: ${rejection.message}`);
  // Validate with fixed controls from the identical reset, without reading intermediate pose for control.
  return rollout(adapter, segment, best.preloads, 0, 'drive', best.controls);
}
