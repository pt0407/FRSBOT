"""Python end of the binary engine protocol served by frsbot/sim/engine/vec_server.mjs."""
import json
import math
import os
import select
import struct
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from frsbot.bot.planner import Goal
from frsbot.core.field import ALLIANCES, seconds_to_toggle
from frsbot.core.inputs import Inputs
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState
from frsbot.sim.engine.fetch import ENGINE_DIR

VEC_SERVER = Path(__file__).resolve().parent / "engine" / "vec_server.mjs"
OP_NEW, OP_STEP, OP_STATE, OP_CLOSE, OP_EXIT = 1, 2, 3, 4, 5
CMD_INPUTS, CMD_GOAL, CMD_IDLE = 0, 1, 2
HEADER, ROBOT, MAX_ROBOTS = 24, 25, 6
PHASES = list(Phase)
SCORE_KEYS = ("autoFuel", "teleFuel", "inactiveFuel", "towerAuto", "towerTele", "foulPoints", "total")
GATE_KEYS = ("open", "hood", "flywheel", "range", "heading", "speed", "tag")
ENGINE_BUTTONS = ("intake", "score", "outtake", "climb", "block", "stow", "field_relative", "hood")
GOAL_KINDS = ("Hold", "Collect", "Score", "Stage", "Feed", "ZoneMow", "Seed")


@dataclass
class Observation:
    """One decoded match block. `state(i)` is robot i's view: the full field, as the live client shows it."""
    slot: int
    t: float
    phase: Phase
    first_inactive: str | None
    hubs: dict[str, HubState]
    robots: list[RobotState]
    fuel: FuelState
    ego: int
    scores: dict[str, int]
    breakdowns: dict[str, dict[str, float]]
    driver: list[dict] = field(default_factory=list)
    rows: np.ndarray | None = field(default=None, repr=False)  # raw f64 robot rows (see vec_server.mjs encode)

    def view(self) -> dict:
        """The subset of the engine's JSON view (server.mjs) that tools read: phase, clock, robots, score."""
        robots = []
        for robot, r in zip(self.robots, self.rows):
            robots.append(dict(id=robot.id, team=robot.team, alliance=robot.alliance, local=bool(r[3]),
                               pos=r[4:7].tolist(), quat=r[7:11].tolist(), vel=r[11:14].tolist(), hopperFuel=robot.hopper,
                               hopperCapacity=robot.capacity, gate=robot.gate, state=robot.mode, drumRpm=float(r[23])))
        return dict(phase=self.phase.value, elapsed=max(self.t, 0.0), clock=-self.t if self.phase == Phase.COUNTDOWN else None,
                    firstInactive=self.first_inactive, robots=robots, score=self.breakdowns,
                    hubs={a: dict(active=h.active) for a, h in self.hubs.items()},
                    fuel=dict(count=len(self.fuel.where), pos=self.fuel.pos.ravel().tolist(), where=self.fuel.where.tolist()))

    def state(self, robot: int | None = None) -> GameState:
        return GameState(t=self.t, phase=self.phase, hubs=self.hubs, robots=self.robots, fuel=self.fuel,
                         me=self.ego if robot is None else robot)


def encode_command(robot: int, action) -> bytes:
    if action is None:
        return struct.pack("<BB", robot, CMD_IDLE)
    if isinstance(action, Goal):
        if not all(math.isfinite(v) for v in action.point):
            raise ValueError("goal point must be finite")
        return struct.pack("<BBBff", robot, CMD_GOAL, GOAL_KINDS.index(action.kind), *action.point)
    if isinstance(action, Inputs):
        bits = sum(1 << i for i, name in enumerate(ENGINE_BUTTONS) if getattr(action, name))
        return struct.pack("<BBfffH", robot, CMD_INPUTS, action.x, action.y, action.turn, bits)
    raise TypeError(f"robot {robot} action must be Inputs, Goal or None, not {type(action).__name__}")


def decode_block(buf: memoryview, at: int, modes: dict[int, str], freeplay: bool) -> tuple[Observation, int]:
    head = np.frombuffer(buf, np.float64, HEADER + ROBOT * MAX_ROBOTS, at)
    h, rows = head[:HEADER], head[HEADER:].reshape(MAX_ROBOTS, ROBOT)
    count = int(h[23])
    t = -float(h[2]) if int(h[3]) == 0 else float(h[1])
    first = None if h[4] < 0 else ALLIANCES[int(h[4])]
    hubs = {a: HubState(bool(h[5 + i]), math.inf if freeplay else seconds_to_toggle(a, t, first)) for i, a in enumerate(ALLIANCES)}
    robots, driver = [], []
    for r in rows[:int(h[7])]:
        qy, qw = r[8], r[10]
        gate = None if r[16] < 0 else {k: bool(int(r[16]) >> b & 1) for b, k in enumerate(GATE_KEYS)}
        robots.append(RobotState(int(r[0]), int(r[1]), ALLIANCES[int(r[2])], float(r[4]), float(r[6]),
                                 math.pi - 2 * math.atan2(qy, qw), float(r[11]), float(r[13]), int(r[14]), int(r[15]),
                                 gate=gate, mode=modes.get(int(r[17]))))
        driver.append(dict(abandoned=bool(r[18]), recovering=bool(r[19]), stage=int(r[20]), scored=int(r[21]), command=int(r[22]),
                           fouls=int(r[24])))
    base = at + (HEADER + ROBOT * MAX_ROBOTS) * 8
    pos = np.frombuffer(buf, np.float32, count * 3, base).reshape(count, 3).copy()
    where = np.frombuffer(buf, np.uint8, count, base + count * 12).copy()
    breakdowns = {a: dict(zip(SCORE_KEYS, h[8 + 7 * i:15 + 7 * i].tolist())) for i, a in enumerate(ALLIANCES)}
    size = base + count * 13 + (-count * 13) % 8 - at
    return Observation(int(h[0]), t, PHASES[int(h[3])], first, hubs, robots, FuelState(pos, where), int(h[22]),
                       {a: int(b["total"]) for a, b in breakdowns.items()}, breakdowns, driver, rows[:int(h[7])].copy()), size


class EngineProcess:
    """One Node process holding many matches (slots). `send` and `receive` are split so callers can pipeline
    several processes: send to all, then receive from all."""

    def __init__(self, engine_dir: Path = ENGINE_DIR, node: str | None = None, timeout: float = 120):
        from frsbot.sim.wasm_env import find_node
        engine_dir = Path(engine_dir)
        if not (engine_dir / "manifest.json").is_file():
            raise FileNotFoundError("engine assets missing; run python3 -m frsbot.sim.engine.fetch")
        self.timeout = timeout
        self.modes: dict[int, str] = {}
        self.freeplay: dict[int, bool] = {}
        self.process = subprocess.Popen([node or find_node(), str(VEC_SERVER), str(engine_dir)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self._buffer = bytearray()
        hello = json.loads(bytes(self.receive("start")))
        self.engine, self.build = hello["engine"], hello["build"]

    def send(self, payload: bytes):
        if self.process is None:
            raise RuntimeError("engine process is closed")
        try:
            self.process.stdin.write(struct.pack("<I", len(payload)) + payload)
        except OSError:
            self.terminate()
            raise

    def _read_exact(self, n: int, what: str, deadline: float):
        fd = self.process.stdout.fileno()
        while len(self._buffer) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError(f"engine {what} timed out after {self.timeout} seconds")
            data = os.read(fd, max(1 << 20, n - len(self._buffer)))
            if not data:
                raise RuntimeError(f"engine exited during {what} (status {self.process.poll()}); see stderr")
            self._buffer += data

    def receive(self, what: str = "request") -> memoryview:
        if self.process is None:
            raise RuntimeError("engine process is closed")
        deadline = time.monotonic() + self.timeout
        try:
            self._read_exact(4, what, deadline)
            size = struct.unpack_from("<I", self._buffer)[0]
            self._read_exact(4 + size, what, deadline)
        except (OSError, TimeoutError, RuntimeError):
            self.terminate()
            raise
        frame = bytes(self._buffer[4:4 + size])
        del self._buffer[:4 + size]
        status, extra = struct.unpack_from("<BI", frame)
        if extra:
            self.modes.update({code: s for code, s in json.loads(frame[5:5 + extra])})
        body = memoryview(frame)[5 + extra:]
        if status:
            raise RuntimeError(bytes(body).decode())
        return body

    def observations(self, body: memoryview) -> list[Observation]:
        count, at, out = struct.unpack_from("<H", body)[0], 8, []
        for _ in range(count):
            slot = int(np.frombuffer(body, np.float64, 1, at)[0])
            obs, size = decode_block(body, at, self.modes, self.freeplay.get(slot, False))
            out.append(obs)
            at += size
        return out

    def send_new(self, slot: int, config: dict, drivers: dict | None = None, delay: int = 0, hold: int = 1,
                 countdown: bool = False):
        self.freeplay[slot] = bool(config.get("freeplay", False))
        self.send(bytes([OP_NEW]) + json.dumps(dict(slot=slot, config=config, drivers=drivers or {}, delay=delay,
                                                    hold=hold, countdown=countdown), allow_nan=False).encode())

    def send_step(self, commands: dict[int, dict[int, object]], ticks: int = 1):
        if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 0xffff:
            raise ValueError("ticks must be an integer in [1, 65535]")
        parts = [struct.pack("<BHH", OP_STEP, ticks, len(commands))]
        for slot, actions in commands.items():
            parts.append(struct.pack("<HB", slot, len(actions)))
            parts.extend(encode_command(robot, action) for robot, action in actions.items())
        self.send(b"".join(parts))

    def send_state(self, slots: list[int]):
        self.send(struct.pack(f"<BH{len(slots)}H", OP_STATE, len(slots), *slots))

    def request(self, send, *args, **kwargs) -> list[Observation]:
        send(*args, **kwargs)
        return self.observations(self.receive())

    def close_slot(self, slot: int):
        self.send(struct.pack("<BH", OP_CLOSE, slot))
        self.receive("close")

    def terminate(self):
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

    def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                self.send(bytes([OP_EXIT]))
                self.receive("exit")
                self.process.wait(timeout=5)
        except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
            pass
        finally:
            self.terminate()


def driver_config(spec) -> dict:
    from frsbot.bot.calibration import profile
    return dict(spec=asdict(spec), profile=profile(spec.team))
