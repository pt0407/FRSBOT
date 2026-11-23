"""Run ImAI through a virtual gamepad in solo Singleplayer only."""
import argparse
import asyncio
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from frsbot.bot.imai import ImAI
from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.core.state import Phase
from frsbot.live.bindings import inputs_to_pad
from frsbot.live.latency import Latency, calibrate, latency_arg
from frsbot.live.page_driver import PagePlanner, install
from frsbot.live.predict import Observations, predict

INJECT = Path(__file__).with_name("inject.js")
ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)
POLL_WITH_PAD = ("([seq, latencySince, pad]) => { window.__imai.writePad(pad);"
                 " return JSON.stringify(window.__imai.poll(seq, {latencySince})); }")
GRAPHICS = """() => {
    localStorage.setItem('frc-sim.graphics', 'low');
    localStorage.setItem('rebuilt-sim.settings', JSON.stringify({version: 1, state: {
        showBackground: false, motionBlur: false, sponsorLogos: false, fpsCap: 60, screamTroll: false
    }}));
}"""


def state_summary(state):
    return dict(t=state.t, phase=state.phase.value, me=state.me,
                robots=[asdict(r) for r in state.robots],
                hubs={k: dict(active=v.active, seconds_to_toggle=v.seconds_to_toggle if math.isfinite(v.seconds_to_toggle) else None)
                      for k, v in state.hubs.items()},
                fuel_count=len(state.fuel.where))


async def start_solo(page, team, alliance):
    await page.goto("https://rebuilt.frsim.com", wait_until="networkidle")
    await page.wait_for_timeout(3500)
    await page.locator("button").filter(has_text="Singleplayer").click()
    await page.locator("button").filter(has_text=re.compile(r"^Change$")).click()
    await page.locator("button").filter(has_text=re.compile(rf"^{team}(?=\D)")).first.click()
    await page.locator("button").filter(has_text=re.compile(r"^Drive ")).click()
    await page.locator("button").filter(has_text=re.compile(rf"^{alliance}$", re.I)).click()
    await page.locator("button").filter(has_text="Start match").click()
    await page.wait_for_function("window.__imai.create !== null")


async def write_pad(page, inputs):
    axes, buttons = inputs_to_pad(inputs)
    await page.evaluate("pad => window.__imai.writePad(pad)", dict(axes=axes, buttons=buttons))


async def drive(page, spec, alliance, output, run_id, timeout=300, driver="python", latency_mode="auto", calibration=6):
    bot, seq, acts = ImAI(spec), 0, 0
    timing, observations = Latency(), Observations()
    last_inputs = Inputs(field_relative=True)
    planner = PagePlanner(spec) if driver == "page" else None
    planning, next_plan, plans = None, 0, 0
    started = time.monotonic()
    baseline = await page.evaluate("window.__imai.poll()")
    horizon = lambda: timing.estimate if latency_mode == "auto" else latency_mode / 1000
    last = None
    with (output / f"{run_id}.jsonl").open("w") as stream:
        def record(value):
            stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")

        config = baseline["create"]
        if config["team"] != spec.team or config["alliance"] != alliance or config["auto"] != "none":
            raise RuntimeError("solo setup does not match requested team/alliance/auto")
        record(dict(kind="match", wall_time=time.time(), create=config, ready=baseline["ready"], driver=driver))
        if calibration:
            samples = await calibrate(page, calibration, timing)
            record(dict(kind="calibration", samples=samples))
        if planner:
            await install(page, spec, horizon())
        while time.monotonic() - started < timeout:
            tick = time.monotonic()
            axes, buttons = inputs_to_pad(last_inputs)
            packet = json.loads(await page.evaluate(POLL_WITH_PAD, [seq, timing.since, dict(axes=axes, buttons=buttons)]))
            previous_trace = timing.since
            timing.ingest(packet)
            if timing.since != previous_trace:
                record(dict(kind="latency", traces=packet["latencies"]))
            if packet["error"] or packet["fieldRelativeError"]:
                raise RuntimeError(packet["error"] or "sent input is not fieldRelative; keep the default camera")
            if planning is not None and planning.done():
                goal = planning.result()
                planning = None
                plans += 1
                await page.evaluate("payload => window.__imai.setGoal(payload)", dict(goal=goal, horizon=horizon()))
            if packet["frames"]:
                frame = packet["frames"][-1]
                seq, last = frame["seq"], frame["view"]
                state = observations.state(last)
                L = horizon()
                status = packet["driverStatus"]
                if planner and status:
                    raw = status["inputs"].copy()
                    raw["field_relative"] = raw.pop("fieldRelative")
                    last_inputs = Inputs(**raw)
                predicted = predict(state, last_inputs, L, spec)
                if planner:
                    if planning is None and tick >= next_plan and state.phase != Phase.DONE:
                        planning = asyncio.create_task(asyncio.to_thread(planner.plan, predicted, status))
                        next_plan = tick + .25
                    inputs = last_inputs
                else:
                    inputs = bot.act(predicted)
                    inputs.climb = False
                    if not inputs.field_relative:
                        raise RuntimeError("ImAI requested robot-relative input; no fieldRelative pad binding")
                    last_inputs = inputs
                    acts += 1
                record(dict(kind="step", t=state.t, wall_time=time.time(), seq=seq,
                            received=frame["received"], state=state_summary(state), inputs=asdict(inputs),
                            predicted_t=predicted.t, latency_ms=L * 1000,
                            blocked=packet["blocked"], sent_input=packet["input"], driver_status=status))
                if state.phase == Phase.DONE:
                    break
            await asyncio.sleep(0)
        else:
            raise TimeoutError(f"solo match did not end within {timeout} seconds")
        if planning is not None:
            await planning
        await page.evaluate("() => { window.__imai.driver = null; window.__imai.zero(); }")
        await page.locator("button").filter(has_text="Play again").wait_for(timeout=30000)
        await page.wait_for_function("!Array.from(document.querySelectorAll('button')).some(b => b.textContent.includes('Counting'))", timeout=30000)
        await page.set_viewport_size(dict(width=1280, height=720))
        text = await page.locator("body").inner_text()
        screenshot = output / f"{run_id}-end.png"
        await page.screenshot(path=str(screenshot), full_page=True)
        seconds = (packet["received"] - baseline["received"]) / 1000
        result = dict(team=spec.team, alliance=alliance, seed=packet["create"]["seed"],
                      create=packet["create"], ready=packet["ready"],
                      driver=driver, latency_mode=latency_mode, latency=timing.summary(),
                      start=packet["create"]["start"], final_score=last["score"], in_game_score_text=text,
                      frames_per_second=(packet["frameCount"] - baseline["frameCount"]) / seconds,
                      pad_polls_per_second=(packet["pollCount"] - baseline["pollCount"]) / seconds,
                      input_sends_per_second=(packet["inputCount"] - baseline["inputCount"]) / seconds,
                      bot_acts_per_second=(packet["driverCount"] if planner else acts) / seconds,
                      planner_per_second=plans / seconds if planner else None, screenshot=str(screenshot))
        record(dict(kind="summary", wall_time=time.time(), **result))
        (output / f"{run_id}-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


async def run(args, spec):
    from playwright.async_api import async_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime(f"imai-{spec.team}-{args.alliance}-%Y%m%dT%H%M%S%fZ")
    async with async_playwright() as playwright:
        if args.cdp:
            browser = await playwright.chromium.connect_over_cdp(args.cdp)
        else:
            browser = await playwright.chromium.launch(headless=not args.headed,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0"),
                     "XAUTHORITY": os.environ.get("XAUTHORITY", str(Path.home() / ".Xauthority"))},
                args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--no-sandbox"])
        page = None
        try:
            page = await (browser.contexts[0].new_page() if args.cdp
                          else browser.new_page(viewport=dict(width=640, height=360), device_scale_factor=1))
            await page.add_init_script(path=str(INJECT))
            await page.add_init_script(script=f"({GRAPHICS})()")
            await start_solo(page, spec.team, args.alliance)
            await page.set_viewport_size(dict(width=320, height=180))
            return await drive(page, spec, args.alliance, args.out, run_id,
                               driver=args.driver, latency_mode=args.latency, calibration=args.calibrate)
        finally:
            if page is not None and not page.is_closed():
                try:
                    await page.evaluate("() => { if (window.__imai) { window.__imai.driver = null; window.__imai.zero(); } }")
                except Exception:
                    LOG.exception("could not zero pad before closing browser")
            if args.cdp:
                if page is not None and not page.is_closed():
                    await page.close()
            else:
                await browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", type=int, required=True)
    parser.add_argument("--alliance", choices=("blue", "red"), default="blue")
    parser.add_argument("--mode", choices=("solo",), default="solo")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--cdp", metavar="URL", help="attach to an already-running Chrome, e.g. http://127.0.0.1:9222")
    parser.add_argument("--driver", choices=("python", "page"), default="python")
    parser.add_argument("--calibrate", type=int, default=6, metavar="N")
    parser.add_argument("--latency", type=latency_arg, default="auto", metavar="auto|MS")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "live")
    args = parser.parse_args()
    if not 0 <= args.calibrate <= 12:
        parser.error("--calibrate must be 0..12")
    specs = {s.team: s for s in load_specs()}
    if args.team not in specs:
        parser.error(f"unknown team {args.team}")
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(asyncio.run(run(args, specs[args.team])), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
