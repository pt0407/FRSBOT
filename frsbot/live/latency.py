"""Page-clock transport timing and physical step-response measurements."""
import math

import numpy as np

from frsbot.core.inputs import Inputs
from frsbot.live.bindings import inputs_to_pad


def distribution(values):
    values = list(values)
    if not values:
        return dict(n=0, median_ms=None, p90_ms=None, median_ticks=None, p90_ticks=None)
    median, p90 = (float(x) for x in np.percentile(values, [50, 90]))
    return dict(n=len(values), median_ms=median, p90_ms=p90,
                median_ticks=median * .06, p90_ticks=p90 * .06)


class Latency:
    def __init__(self):
        self.traces = []
        self.steps = []
        self.since = 0

    def ingest(self, packet):
        traces = packet.get("latencies", [])
        self.traces.extend(t for t in traces if t["id"] > self.since)
        if traces:
            self.since = max(self.since, max(t["id"] for t in traces))

    @property
    def estimate(self):
        measured = [s["ms"] for s in self.steps if s.get("ms") is not None]
        if measured:
            return float(np.median(measured)) / 1000
        recent = self.traces[-120:]
        return float(np.median([t["frameAt"] - t["at"] for t in recent])) / 1000 if recent else 0

    def summary(self):
        stages = {name: distribution(t[end] - t[start] for t in self.traces)
                  for name, start, end in [("write_to_poll", "at", "pollAt"),
                                            ("poll_to_input", "pollAt", "inputAt"),
                                            ("input_to_frame", "inputAt", "frameAt"),
                                            ("write_to_frame", "at", "frameAt")]}
        steps = distribution(s["ms"] for s in self.steps if s.get("ms") is not None)
        ticks = [s["tick_delta"] for s in self.steps if s.get("ms") is not None]
        steps.update(effect_tick_median=float(np.median(ticks)) if ticks else None,
                     effect_tick_p90=float(np.percentile(ticks, 90)) if ticks else None,
                     attempts=self.steps)
        return dict(step_response=steps, transport=stages, estimate_ms=self.estimate * 1000,
                    frame_stage="first frame received after input send; not proof of physical effect")


def latency_arg(value):
    if value == "auto":
        return value
    milliseconds = float(value)
    if not math.isfinite(milliseconds) or not 0 <= milliseconds <= 1000:
        raise ValueError("latency must be auto or 0..1000 milliseconds")
    return milliseconds


async def calibrate(page, count, latency):
    """Run wholly on the page clock so Python/CDP scheduling cannot hide responses."""
    axes, buttons = inputs_to_pad(Inputs(y=1, field_relative=True))
    result = await page.evaluate("""async ({count, axes, buttons}) => {
      const api = window.__imai, samples = [], pause = ms => new Promise(r => setTimeout(r, ms));
      const ego = f => f?.view.robots.find(r => r.local) ?? f?.view.robots[0];
      const angle = r => Math.PI - 2 * Math.atan2(r.quat[1], r.quat[3]);
      const speed = r => Math.hypot(r.vel[0], r.vel[2]);
      const deadline = performance.now() + 22000;
      const check = () => { if (api.error || api.blocked || api.fieldRelativeError) throw Error(api.error || 'calibration blocked'); };
      try {
        while (api.frames.at(-1)?.view.phase !== 'auto') {
          check(); if (performance.now() > deadline) throw Error('auto did not start'); await pause(10);
        }
        for (let i = 0; i < count; i++) {
          api.zero();
          let stable = null, previous = null;
          const settleDeadline = performance.now() + 2500;
          while (true) {
            check(); const f = api.frames.at(-1), r = ego(f);
            if (f?.view.phase !== 'auto' || f.view.elapsed > 18) return samples;
            const still = r && speed(r) < .04 && previous && Math.abs(angle(r) - angle(previous)) < .005;
            stable = still ? stable ?? performance.now() : null;
            if (stable !== null && performance.now() - stable > 200) break;
            if (performance.now() > settleDeadline) { samples.push({kind:'settle',error:'did not reach rest'}); return samples; }
            previous = r; await pause(25);
          }
          const before = api.frames.at(-1), r0 = ego(before), turn = i % 2 === 1;
          const command = {axes: turn ? [0,0,-1,0] : axes, buttons};
          const write = api.writePad(command), until = performance.now() + 800;
          let effect = null;
          while (performance.now() < until && !effect) {
            check();
            effect = api.frames.find(f => f.seq > before.seq && f.received >= write.at &&
              (turn ? Math.abs(angle(ego(f)) - angle(r0)) > .008 : speed(ego(f)) > .08));
            if (!effect) await pause(5);
          }
          api.zero();
          samples.push({kind:turn ? 'turn' : 'forward', write_at:write.at, observed_seq:before.seq,
            observed_tick:api.frameTick(before), observed_elapsed:before.view.elapsed,
            effect_seq:effect?.seq ?? null, effect_tick:effect ? api.frameTick(effect) : null,
            ms:effect ? effect.received-write.at : null,
            tick_delta:effect ? api.frameTick(effect)-api.frameTick(before) : null,
            error:effect ? null : 'no response within 800ms'});
        }
        return samples;
      } finally { api.zero(); }
    }""", dict(count=count, axes=axes, buttons=buttons))
    latency.steps.extend(result)
    return result
