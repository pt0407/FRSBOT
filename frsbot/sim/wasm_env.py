"""Synchronous, 60 Hz offline backend using the game's real WASM physics."""
import json
import math
import os
import select
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from frsbot.core.field import ALLIANCES, seconds_to_toggle
from frsbot.core.inputs import Inputs
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState
from frsbot.sim.engine.fetch import ENGINE_DIR

SERVER = Path(__file__).resolve().parent / "engine" / "server.mjs"


def find_node() -> str:
    node = shutil.which("node")
    if node:
        return node
    versions = Path.home() / ".nvm" / "versions" / "node"
    candidates = sorted(versions.glob("v*/bin/node"),
                        key=lambda p: tuple(int(n) for n in p.parents[1].name.lstrip("v").split(".")), reverse=True)
    for candidate in candidates:
        if os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError("Node.js is required; install it on PATH or under ~/.nvm/versions/node")


def view_to_state(view: dict, freeplay: bool = False) -> GameState:
    """Views already use replay world coordinates and SI units, not field fx/fy."""
    t = -float(view["clock"]) if view["phase"] == "countdown" else float(view["elapsed"])
    robots = []
    for raw in view["robots"]:
        quat = raw["quat"]
        robots.append(RobotState(
            id=raw["id"], team=raw["team"], alliance=raw["alliance"],
            x=raw["pos"][0], z=raw["pos"][2],
            a=math.pi - 2 * math.atan2(quat[1], quat[3]),
            vx=raw["vel"][0], vz=raw["vel"][2],
            hopper=raw["hopperFuel"], capacity=raw["hopperCapacity"],
            gate=raw.get("gate"), mode=raw.get("state"),
        ))
    fuel = view["fuel"]
    pos = np.asarray(fuel["pos"], dtype=np.float32).reshape(fuel["count"], 3)
    where = np.asarray(fuel["where"], dtype=np.uint8)
    if where.shape != (fuel["count"],):
        raise ValueError("engine FUEL position/where counts disagree")
    return GameState(
        t=t, phase=Phase(view["phase"]),
        hubs={alliance: HubState(view["hubs"][alliance]["active"],
                                 math.inf if freeplay else seconds_to_toggle(alliance, t, view["firstInactive"]))
              for alliance in ALLIANCES},
        robots=robots, fuel=FuelState(pos, where),
        me=next((i for i, robot in enumerate(view["robots"]) if robot.get("local")), 0),
    )


class JsonWasmEnv:
    """Original one-robot JSON-lines backend (server.mjs); kept as the benchmark baseline and protocol reference."""

    def __init__(self, engine_dir: Path = ENGINE_DIR, node: str | None = None, timeout: float = 120):
        self.engine_dir = Path(engine_dir)
        self.node = node
        self.timeout = timeout
        self.process = None
        self.view = None
        self.engine = None
        self.build = None
        self._buffer = b""
        self._freeplay = False
        self.driver = {}

    def _spawn(self):
        if not (self.engine_dir / "manifest.json").is_file():
            raise FileNotFoundError("engine assets missing; run python3 -m frsbot.sim.engine.fetch")
        self.process = subprocess.Popen([self.node or find_node(), str(SERVER), str(self.engine_dir)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self._buffer = b""

    def _request(self, cmd: str, **payload):
        if self.process is None:
            raise RuntimeError("call reset before using WasmEnv")
        try:
            packet = (json.dumps({"cmd": cmd, **payload}, allow_nan=False) + "\n").encode()
            self.process.stdin.write(packet)
            deadline = time.monotonic() + self.timeout
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise TimeoutError(f"engine {cmd} timed out after {self.timeout} seconds")
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError(f"engine exited during {cmd} (status {self.process.poll()}); see stderr")
                self._buffer += data
            line, self._buffer = self._buffer.split(b"\n", 1)
            response = json.loads(line)
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError):
            self._terminate()
            raise
        if not response["ok"]:
            raise RuntimeError(response["error"])
        self.engine, self.build = response.get("engine"), response.get("build")
        if "view" in response:
            self.view = response["view"]
        self.driver = response.get("driver", {})
        return response.get("view")

    def reset(self, team: int, alliance: str = "blue", start: dict | None = None, seed: int = 1,
              auto: str = "none", event: str = "event", hp: bool = True, unload: bool = True,
              freeplay: bool = False, countdown: bool = False, driver: dict | None = None) -> GameState:
        """`driver` (frsbot.sim.ipc.driver_config(spec)) lets step() take a planner Goal for the JS driver."""
        if self.process is None:
            self._spawn()
        self._request("new", config=dict(team=team, alliance=alliance, start=start if start is not None else dict(fx=0, fy=0, headingDeg=0),
                                         seed=seed, auto=auto, event=event, hp=hp, unload=unload, freeplay=freeplay),
                      drivers={"0": driver} if driver else {})
        self._freeplay = freeplay
        return view_to_state(self._request("start", countdown=countdown), freeplay)

    def step(self, inputs: Inputs, ticks: int = 1) -> GameState:
        if self.view is None:
            raise RuntimeError("call reset before stepping")
        if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 60000:
            raise ValueError("ticks must be an integer in [1, 60000]")
        robot = next((r for r in self.view["robots"] if r.get("local")), self.view["robots"][0])
        if hasattr(inputs, "kind"):  # planner Goal: the JS driver acts every tick
            return view_to_state(self._request("step", goals={str(robot["id"]): dict(kind=inputs.kind, point=list(inputs.point))},
                                               n=ticks), self._freeplay)
        raw = asdict(inputs)
        raw["fieldRelative"] = raw.pop("field_relative")
        raw.pop("seq")
        return view_to_state(self._request("step", inputs={str(robot["id"]): raw}, n=ticks), self._freeplay)

    def state(self) -> GameState:
        return view_to_state(self._request("state"), self._freeplay)

    def _terminate(self):
        if self.process is not None:
            process, self.process = self.process, None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdin.close()
            process.stdout.close()
        self.view = None

    def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                self._request("close")
                self.process.wait(timeout=5)
        finally:
            self._terminate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class WasmEnv:
    """Real-engine matches over the binary protocol (vec_server.mjs). One match; one to six robots.

    `reset(team, ...)` keeps the original one-robot call; `reset(robots=[RobotSlot(...)], ego=i, seed=...)` makes a
    multi-robot match and returns robot i's GameState. `step(Inputs | Goal, ticks)` drives the ego robot and returns
    its GameState; `step({i: Inputs | Goal | None}, ticks)` returns a StepResult with every robot's view, alliance
    scores and breakdowns. Robots without an action use their RobotSlot.policy (idle when None).
    `latency=(lo, hi)` delays observations by U(lo, hi) seconds and holds inputs for 1-3 ticks, drawn per episode.
    """

    def __init__(self, engine_dir: Path = ENGINE_DIR, node: str | None = None, timeout: float = 120,
                 latency=None):
        self.engine_dir, self.node, self.timeout, self.latency = Path(engine_dir), node, timeout, latency
        self.vec = None
        self.engine = self.build = None
        self.observation = None

    @property
    def process(self):
        return self.vec.workers[0].worker.engine.process if self.vec and self.vec.workers else None

    @property
    def view(self) -> dict | None:
        return self.observation.view() if self.observation else None

    @property
    def delay_ticks(self) -> int:
        return self.vec.latencies[0][0] if self.vec else 0

    @property
    def hold_ticks(self) -> int:
        return self.vec.latencies[0][1] if self.vec else 1

    def reset(self, team: int | None = None, alliance: str = "blue", start: dict | None = None, seed: int = 1,
              auto: str = "none", event: str = "event", hp: bool = True, unload: bool = True,
              freeplay: bool = False, countdown: bool = False, robots: list | None = None, ego: int = 0) -> GameState:
        from frsbot.sim.vec_env import RobotSlot, VecWasmEnv
        if robots is None:
            if team is None:
                raise ValueError("reset needs a team or robots=[RobotSlot(...)]")
            robots = [RobotSlot(team, alliance, start if start is not None else dict(fx=0, fy=0, headingDeg=0))]
        if self.vec is None:
            if not (self.engine_dir / "manifest.json").is_file():
                raise FileNotFoundError("engine assets missing; run python3 -m frsbot.sim.engine.fetch")
            self.vec = VecWasmEnv(1, 1, self.engine_dir, self.node, self.timeout, self.latency, processes=False)
            self.engine, self.build = self.vec.engine, self.vec.build
        state = self.vec.reset([dict(robots=robots, ego=ego, seed=seed, auto=auto, event=event, hp=hp, unload=unload,
                                     freeplay=freeplay, countdown=countdown)])[0]
        self.observation = self.vec.observations[0]
        return state

    def step(self, inputs, ticks: int = 1):
        if self.observation is None:
            raise RuntimeError("call reset before stepping")
        if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 60000:
            raise ValueError("ticks must be an integer in [1, 60000]")
        ego = self.vec.egos[0]
        actions = inputs if isinstance(inputs, dict) else {ego: inputs}
        result = self.vec.step([actions], ticks)[0]
        self.observation = result.observation
        return result if isinstance(inputs, dict) else result.views[ego]

    def state(self) -> GameState:
        if self.observation is None:
            raise RuntimeError("call reset before using WasmEnv")
        self.observation = self.vec.state(0)
        return self.observation.state(self.vec.egos[0])

    def close(self):
        if self.vec is not None:
            self.vec.close()
        self.vec, self.observation = None, None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
