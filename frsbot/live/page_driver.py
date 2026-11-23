"""Load the unmodified Node driver port and delegate goal selection to ImAI."""
import re
from dataclasses import asdict
from pathlib import Path

from frsbot.bot.calibration import profile
from frsbot.bot.imai import ImAI
from frsbot.core.inputs import Inputs

SOURCE = Path(__file__).resolve().parents[1] / "sim" / "engine" / "driver.mjs"


def browser_source():
    source = SOURCE.read_text().split("// CLI used by the parity test:")[0]
    source = re.sub(r"^import .*;\n", "", source, flags=re.M).replace("export ", "")
    return "(() => {\n" + source + "\nObject.assign(window.__imai, {Driver, viewToState, phaseAt, isActive, secondsToToggle}); })();"


class GoalSink:
    def __init__(self):
        self.abandon = False
        self.goal = None

    def act(self, goal, state):
        self.goal = goal
        return Inputs(field_relative=True)


class PagePlanner:
    def __init__(self, spec):
        self.bot = ImAI(spec)
        self.sink = GoalSink()
        self.bot.driver = self.sink
        # The page driver only understands goals — the open-loop auto route emits
        # raw Inputs and would starve it for the first 20 s.
        self.bot.auto = None

    def plan(self, state, status):
        self.sink.goal = None
        self.sink.abandon = bool(status and status.get("abandon"))
        self.bot.act(state)
        return asdict(self.sink.goal) if self.sink.goal else None


async def install(page, spec, horizon):
    await page.evaluate(browser_source())
    await page.add_script_tag(path=str(Path(__file__).with_name("page_driver.js")))
    await page.evaluate("config => window.__imai.installDriver(config)",
                        dict(spec=asdict(spec), profile=profile(spec.team), horizon=horizon))
