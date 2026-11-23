(() => {
  const plain = value => {
    if (ArrayBuffer.isView(value)) return Array.from(value);
    if (Array.isArray(value)) return value.map(plain);
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, plain(v)]));
    return value;
  };
  const ring = (items, value, limit = 120) => {
    items.push(value);
    if (items.length > limit) items.shift();
  };
  function decode(meta, buf) {
    const n = meta.fuelCount;
    if (!Number.isInteger(n) || n < 0 || !(buf instanceof ArrayBuffer) ||
        buf.byteLength < 77 * n || (buf.byteLength - n) % 4) throw new Error('invalid frame buffer');
    const { fuelCount, prev, ...view } = plain(meta);
    view.fuel = { count: n, pos: Array.from(new Float32Array(buf, 0, 3 * n)),
      where: Array.from(new Uint8Array(buf, buf.byteLength - n, n)) };
    return view;
  }
  const api = window.__imai = {
    pad: { axes: [0, 0, 0, 0], buttons: Array.from({ length: 17 }, () => ({ pressed: false, value: 0, touched: false })), timestamp: 0 },
    frames: [], inputs: [], polls: [], events: [], latencies: [], waitingPoll: [], waitingInput: [], waitingFrame: [],
    seq: 0, writeSeq: 0, frameCount: 0, pollCount: 0, inputCount: 0,
    driver: null, driverCount: 0, driverStatus: null,
    create: null, ready: null, error: null, fieldRelativeError: false, ended: false, blocked: document.hidden,
    decode,
    frameTick(frame) { return Math.round((frame.view.phase === 'countdown' ? -frame.view.clock : frame.view.elapsed) * 60); },
    zero() {
      api.pad.axes.fill(0);
      api.pad.buttons.forEach(b => { b.pressed = false; b.value = 0; b.touched = false; });
      api.pad.timestamp = Math.max(api.pad.timestamp + 1, performance.now());
    },
    writePad({ axes, buttons }) {
      if (!Array.isArray(axes) || axes.length !== 4 || axes.some(x => !Number.isFinite(x)) ||
          !Array.isArray(buttons) || buttons.length !== 17 || buttons.some(b => !b || typeof b !== 'object')) {
        api.zero();
        throw new Error('invalid pad');
      }
      if (api.blocked || document.hidden || api.ended || api.error || api.fieldRelativeError) return api.zero();
      api.pad.axes = axes.map(x => Math.max(-1, Math.min(1, x)));
      api.pad.buttons = buttons.map((b, i) => {
        const pressed = i !== 3 && !!b.pressed;
        return { pressed, value: pressed ? 1 : 0, touched: pressed };
      });
      api.pad.timestamp = Math.max(api.pad.timestamp + 1, performance.now());
      const frame = api.frames.at(-1);
      const write = { id: ++api.writeSeq, at: performance.now(), observedSeq: frame?.seq ?? null,
        observedTick: frame ? api.frameTick(frame) : null };
      ring(api.waitingPoll, write, 1200);
      return write;
    },
    poll(sinceSeq = 0, { latestOnly = true, latencySince = 0 } = {}) {
      const frames = api.frames.filter(f => f.seq > sinceSeq);
      return { frames: latestOnly ? frames.slice(-1) : frames, seq: api.seq,
        received: performance.now(), frameCount: api.frameCount, pollCount: api.pollCount,
        inputCount: api.inputCount, input: api.inputs.at(-1) ?? null, create: api.create, ready: api.ready,
        latencies: api.latencies.filter(t => t.id > latencySince), driverCount: api.driverCount, driverStatus: api.driverStatus,
        error: api.error, fieldRelativeError: api.fieldRelativeError, blocked: api.blocked, ended: api.ended };
    },
  };
  const fail = error => { api.error = String(error.message || error); api.zero(); };
  const NativeWorker = window.Worker;
  window.Worker = class extends NativeWorker {
    constructor(url, options) {
      super(url, options);
      this.addEventListener('error', fail);
      this.addEventListener('messageerror', fail);
      this.addEventListener('message', event => {
        const m = event.data, at = performance.now();
        try {
          if (m.t === 'ready') {
            api.ready = { at, ...plain(m) };
            ring(api.events, api.ready);
          } else if (m.t === 'frame') {
            const view = decode(m.meta, m.buf);
            ring(api.frames, { seq: ++api.seq, received: at, stepTime: m.stepTime, view });
              api.frameCount++;
              for (const trace of api.waitingFrame) {
                trace.frameAt = at; trace.effectTick = api.frameTick(api.frames.at(-1));
                ring(api.latencies, trace, 1200);
              }
              api.waitingFrame = [];
            if (view.phase === 'done') { api.ended = true; api.zero(); }
          } else if (m.t === 'error') {
            api.error = m.message || 'worker error';
            api.zero();
          }
        } catch (error) { api.error = String(error); api.zero(); }
      });
      const post = this.postMessage.bind(this);
      this.postMessage = (m, ...args) => {
        const at = performance.now();
        if (m.t === 'create') {
          api.create = { at, ...plain(m) };
          ring(api.events, api.create);
          api.ended = false;
        } else if (m.t === 'input') {
          ring(api.inputs, { seq: ++api.inputCount, at, input: plain(m.input) });
          if (m.input.fieldRelative !== true) { api.fieldRelativeError = true; api.zero(); }
          for (const trace of api.waitingInput) {
            trace.inputAt = at; ring(api.waitingFrame, trace, 1200);
          }
          api.waitingInput = [];
        }
        return post(m, ...args);
      };
    }
  };
  const pad = { id: 'ImAI virtual standard', index: 0, connected: true, mapping: 'standard', vibrationActuator: null,
    get axes() { return api.pad.axes; }, get buttons() { return api.pad.buttons; }, get timestamp() { return api.pad.timestamp; } };
  Object.defineProperty(navigator, 'getGamepads', { value: () => {
    ring(api.polls, performance.now(), 600);
    api.pollCount++;
    if (api.driver && !api.blocked && !api.ended && !api.error && !api.fieldRelativeError) {
      try { api.driver(); } catch (error) { fail(error); }
    }
    const at = performance.now();
    for (const trace of api.waitingPoll) {
      trace.pollAt = at; trace.sampled = trace === api.waitingPoll.at(-1);
      ring(api.waitingInput, trace, 1200);
    }
    api.waitingPoll = [];
    return [pad, null, null, null];
  } });
  window.addEventListener('blur', () => { api.blocked = true; api.zero(); });
  window.addEventListener('error', fail);
  window.addEventListener('unhandledrejection', event => fail(event.reason || 'unhandled rejection'));
  window.addEventListener('focus', () => { api.blocked = document.hidden; });
  document.addEventListener('visibilitychange', () => { api.blocked = document.hidden; if (api.blocked) api.zero(); });
})();
