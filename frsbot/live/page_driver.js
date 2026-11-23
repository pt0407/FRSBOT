(() => {
  const api = window.__imai;
  const shape = v => Math.abs(v) < .12 ? 0 : Math.sign(v) * (Math.abs(v) - .12) / .88;
  api.predictState = (frame, inputs, L, spec) => {
    const state = api.viewToState(frame.view), old = api.frames.find(f => f.seq === frame.seq - 1);
    if (!L) return state;
    const previous = old ? api.viewToState(old.view) : null;
    const sampleDt = previous ? state.t - previous.t : 0;
    const history = sampleDt > 0 && sampleDt <= .3;
    for (let i = 0; i < state.robots.length; i++) {
      const r = state.robots[i];
      if (i !== state.me) { r.x += r.vx * Math.min(L, .3); r.z += r.vz * Math.min(L, .3); continue; }
      const before = history ? previous.robots.find(p => p.id === r.id) : null;
      let omega = before ? ((r.a - before.a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI : 0;
      omega = before ? omega / sampleDt : 0;
      let remaining = L;
      while (remaining > 1e-9) {
        const dt = Math.min(1 / 60, remaining), phase = api.phaseAt(state.t + L - remaining);
        const enabled = !['countdown', 'gap', 'settle', 'done'].includes(phase);
        const x = enabled ? inputs.x : 0, y = enabled ? inputs.y : 0, turn = enabled ? inputs.turn : 0;
        const norm = Math.hypot(x, y), scale = norm ? shape(Math.min(norm, 1)) / norm : 0, sign = r.alliance === 'blue' ? 1 : -1;
        const vx = sign * y * scale * spec.max_speed, vz = sign * x * scale * spec.max_speed;
        const dx = vx - r.vx, dz = vz - r.vz, change = Math.hypot(dx, dz);
        const factor = change ? Math.min(1, spec.accel * dt / change) : 0;
        r.vx += dx * factor; r.vz += dz * factor;
        omega += Math.max(-10 * dt, Math.min(10 * dt, -shape(turn) * spec.turn_rate - omega));
        r.a += omega * dt; r.x += r.vx * dt; r.z += r.vz * dt;
        remaining -= dt;
      }
    }
    for (let i = 0; i < state.fuelWhere.length; i++) if (state.fuelWhere[i] === 1) {
      const same = history && previous.fuelWhere[i] === 1;
      for (let j = 0; j < 3; j++) {
        const k = 3 * i + j;
        const velocity = same ? (state.fuelPos[k] - previous.fuelPos[k]) / sampleDt - (j === 1 ? 9.81 * sampleDt / 2 : 0) : 0;
        state.fuelPos[k] += velocity * L - (j === 1 ? .5 * 9.81 * L * L : 0);
      }
      state.fuelPos[3 * i + 1] = Math.max(.075, state.fuelPos[3 * i + 1]);
    }
    state.t += L; state.phase = api.phaseAt(state.t);
    for (const a of ['blue', 'red']) state.hubs[a] = {
      active: api.isActive(a, state.t, state.firstInactive),
      secondsToToggle: state.firstInactive == null && state.t >= 23 && state.t < 33 ? 33 - state.t : api.secondsToToggle(a, state.t, state.firstInactive)
    };
    return state;
  };
  api.installDriver = ({spec, profile, horizon}) => {
    const driver = new api.Driver(spec, profile);
    let command = {x: 0, y: 0, turn: 0}, message = null, abandon = false;
    api.setGoal = payload => { message = {...payload, at: performance.now()}; abandon = false; };
    api.driver = () => {
      const frame = api.frames.at(-1);
      if (!frame || !message?.goal || performance.now() - message.at > 1500 || performance.now() - frame.received > 500 ||
          ['countdown', 'settle', 'done'].includes(frame.view.phase)) {
        command = {x:0, y:0, turn:0, fieldRelative:true, climb:false};
        api.driverStatus = {abandon, recovering:driver.recovering, inputs:command, goal:null};
        api.zero(); return;
      }
      const L = message.horizon ?? horizon;
      const state = api.predictState(frame, command, L, spec);
      command = driver.act(message.goal, state);
      if (command.fieldRelative !== true) throw Error('page driver requested robot relative');
      command.climb = false;
      const buttons = Array.from({length: 17}, () => ({pressed: false}));
      for (const [key, i] of Object.entries({intake:6,score:7,outtake:4,block:5,stow:2,hood:1})) buttons[i].pressed = !!command[key];
      api.writePad({axes:[command.x, -command.y, -command.turn, 0], buttons});
      api.driverCount++;
      abandon ||= driver.abandon;
      api.driverStatus = {abandon, recovering:driver.recovering, inputs:command, goal:message.goal, predicted_t:state.t};
    };
  };
})();
