"""Vectorised real-engine matches: one Node worker per core, several matches per worker, up to 6 robots per match."""
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from frsbot.core.inputs import Inputs
from frsbot.core.specs import load_specs
from frsbot.core.state import GameState, Phase
from frsbot.sim.engine.fetch import ENGINE_DIR
from frsbot.sim.ipc import EngineProcess, Observation, driver_config
from frsbot.sim.policies import IdlePolicy

TICK = 1 / 60


@dataclass
class RobotSlot:
    team: int
    alliance: str = "blue"
    start: dict | None = None
    policy: object | None = None  # used whenever this robot gets no action in step(); None means idle
    preload: int = 8


@dataclass
class StepResult:
    t: float
    views: dict[int, GameState]  # every robot's own view of the same field
    scores: dict[str, int]
    breakdowns: dict[str, dict[str, float]]
    done: bool
    ego: int
    observation: Observation = field(repr=False)


def default_starts(robots: list[RobotSlot]) -> list[dict]:
    """Spread robots along their alliance's starting line when no start pose is given."""
    lanes = {"blue": 0, "red": 0}
    starts = []
    for robot in robots:
        if robot.start is not None:
            starts.append(robot.start)
            continue
        lane = lanes[robot.alliance]
        lanes[robot.alliance] += 1
        fy = (2.0, 6.07, 4.035)[lane % 3]
        starts.append(dict(fx=3.6, fy=fy, headingDeg=0) if robot.alliance == "blue" else dict(fx=12.94, fy=fy, headingDeg=180))
    return starts


@dataclass
class Latency:
    """docs/LIVE.md §2: observation delay (seconds) and input hold (ticks), each drawn per episode."""
    delay: tuple[float, float] = (0.0, 0.15)
    hold: tuple[int, int] = (1, 3)

    def draw(self, rng) -> tuple[int, int]:
        lo, hi = self.delay
        if not 0 <= lo <= hi <= 10:
            raise ValueError("latency must satisfy 0 <= lo <= hi <= 10 seconds")
        if not 1 <= self.hold[0] <= self.hold[1] <= 60:
            raise ValueError("hold must satisfy 1 <= lo <= hi <= 60 ticks")
        return round(rng.uniform(lo, hi) * 60), int(rng.integers(self.hold[0], self.hold[1] + 1))


class _Worker:
    """One Node engine process and the policies of its matches. Runs in-process or inside a _RemoteWorker."""

    def __init__(self, engine_dir, node, timeout, latency, n_matches):
        self.engine = EngineProcess(engine_dir, node, timeout)
        self.latency = latency
        self.specs = {s.team: s for s in load_specs()}
        self.drivers = {}
        self.policies = [[] for _ in range(n_matches)]
        self.ego = [0] * n_matches
        self.latencies = [(0, 1)] * n_matches
        self.observations: list[Observation | None] = [None] * n_matches

    def info(self):
        return self.engine.engine, self.engine.build

    def _driver(self, team):
        if team not in self.drivers:
            self.drivers[team] = driver_config(self.specs[team])
        return self.drivers[team]

    def reset(self, items):
        """items: [(slot, config)]. Returns [(slot, observation, (delay, hold))]; replies stay in sync on errors."""
        sent = []
        for slot, config in items:
            config = dict(config)
            robots = [r if isinstance(r, RobotSlot) else RobotSlot(**r) for r in config.pop("robots")]
            for r in robots:
                if r.team not in self.specs:
                    raise RuntimeError(f"unknown team {r.team}")
            countdown = config.pop("countdown", False)
            match = dict(dict(ego=0, seed=1, auto="none", event="event", hp=True, unload=True, freeplay=False), **config)
            match["robots"] = [dict(team=r.team, alliance=r.alliance, start=s, preload=r.preload)
                               for r, s in zip(robots, default_starts(robots))]
            latency = self.latency.draw(np.random.default_rng(match["seed"])) if self.latency else (0, 1)
            self.engine.send_new(slot, match, {i: self._driver(r.team) for i, r in enumerate(robots)}, *latency, countdown)
            sent.append((slot, [r.policy or IdlePolicy() for r in robots], match["ego"], latency))
        out, error = [], None
        for slot, policies, ego, latency in sent:
            try:
                obs = self.engine.observations(self.engine.receive("new"))[0]
            except RuntimeError as e:
                error = error or e
                continue
            self.policies[slot], self.ego[slot], self.latencies[slot], self.observations[slot] = policies, ego, latency, obs
            out.append((slot, obs, latency))
        if error:
            raise error
        return out

    def step(self, actions: dict[int, dict], ticks: int):
        for slot in actions:
            if self.observations[slot] is None:
                raise RuntimeError("call reset before stepping")
        commands, batches = {}, defaultdict(list)
        for slot, given in actions.items():
            obs, commands[slot] = self.observations[slot], dict(given)
            for i, policy in enumerate(self.policies[slot]):
                if i in given:
                    continue
                if hasattr(policy, "act_batch"):
                    batches[id(policy)].append((policy, slot, i))
                else:
                    commands[slot][i] = policy.act(obs.state(i), obs.driver[i])
        for items in batches.values():
            states = [self.observations[slot].state(i) for _, slot, i in items]
            infos = [self.observations[slot].driver[i] for _, slot, i in items]
            for (_, slot, i), action in zip(items, items[0][0].act_batch(states, infos)):
                commands[slot][i] = action
        observed = self.engine.request(self.engine.send_step, commands, ticks)
        for obs in observed:
            self.observations[obs.slot] = obs
        return observed

    def state(self, slots):
        observed = self.engine.request(self.engine.send_state, slots)
        for obs in observed:
            self.observations[obs.slot] = obs
        return observed

    def close(self):
        self.engine.close()


def _serve_worker(connection, args):
    try:
        worker = _Worker(*args)
    except BaseException as error:  # report startup failures to the parent
        connection.send(("error", error))
        return
    connection.send(("ok", worker.info()))
    try:
        while True:
            method, payload = connection.recv()
            if method == "close":
                break
            try:
                connection.send(("ok", getattr(worker, method)(*payload)))
            except Exception as error:  # noqa: BLE001 - forwarded to the caller
                connection.send(("error", error))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        worker.close()
        connection.close()


class _RemoteWorker:
    """A _Worker in its own Python process, so rule/replay policies of different workers run on different cores."""

    def __init__(self, *args):
        import multiprocessing
        context = multiprocessing.get_context("forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve_worker, args=(child, args), daemon=True)
        self.process.start()
        child.close()
        self._info = self._result()

    def info(self):
        return self._info

    def _result(self):
        try:
            status, value = self.connection.recv()
        except EOFError:
            raise RuntimeError(f"engine worker exited (status {self.process.exitcode})") from None
        if status == "error":
            raise value
        return value

    def submit(self, method, *payload):
        self.connection.send((method, payload))

    def call(self, method, *payload):
        self.submit(method, *payload)
        return self._result()

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(("close", ()))
            except OSError:
                pass
            self.process.join(10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
        self.connection.close()


class _LocalWorker:
    def __init__(self, *args):
        self.worker = _Worker(*args)
        self._pending = None

    def info(self):
        return self.worker.info()

    def submit(self, method, *payload):
        try:
            self._pending = ("ok", getattr(self.worker, method)(*payload))
        except Exception as error:  # noqa: BLE001 - re-raised by _result, like a remote worker
            self._pending = ("error", error)

    def _result(self):
        (status, value), self._pending = self._pending, None
        if status == "error":
            raise value
        return value

    def call(self, method, *payload):
        self.submit(method, *payload)
        return self._result()

    def close(self):
        self.worker.close()


class VecWasmEnv:
    """`VecWasmEnv(n_workers=nproc, matches_per_worker=k)`: n Node engine workers, each running k matches.

    Workers run in their own Python processes (`processes=True`, default when n_workers > 1), so robots driven by
    their RobotSlot.policy (rule ImAI, replay, idle) are computed in parallel next to their engine. Actions passed to
    `step` come from the caller, which can batch them (e.g. one network forward pass for every match)."""

    def __init__(self, n_workers: int | None = None, matches_per_worker: int = 1, engine_dir=ENGINE_DIR,
                 node: str | None = None, timeout: float = 120, latency: tuple[float, float] | Latency | None = None,
                 processes: bool | None = None):
        n_workers = n_workers or os.cpu_count() or 1
        if n_workers < 1 or matches_per_worker < 1:
            raise ValueError("n_workers and matches_per_worker must be positive")
        self.latency = Latency(tuple(latency)) if latency is not None and not isinstance(latency, Latency) else latency
        self.per_worker = matches_per_worker
        cls = _RemoteWorker if (n_workers > 1 if processes is None else processes) else _LocalWorker
        self.workers = []
        try:
            for _ in range(n_workers):
                self.workers.append(cls(engine_dir, node, timeout, self.latency, matches_per_worker))
        except BaseException:
            self.close()
            raise
        self.engine, self.build = self.workers[0].info()
        n = n_workers * matches_per_worker
        self.observations: list[Observation | None] = [None] * n
        self.egos = [0] * n
        self.latencies = [(0, 1)] * n

    def __len__(self):
        return len(self.observations)

    def _gather(self, method, by_worker: dict[int, list]):
        for w, payload in by_worker.items():
            self.workers[w].submit(method, *payload)
        results, error = {}, None
        for w in by_worker:
            try:
                results[w] = self.workers[w]._result()
            except Exception as e:  # noqa: BLE001 - every worker must be read before re-raising
                error = error or e
        if error:
            raise error
        return results

    def reset(self, configs: list[dict], matches: list[int] | None = None) -> list[GameState]:
        """configs[j] = dict(robots=[RobotSlot...], ego=0, seed=1, ...) for match matches[j] (default 0..len-1)."""
        indices = list(range(len(configs))) if matches is None else list(matches)
        if len(indices) != len(configs) or any(not 0 <= j < len(self) for j in indices):
            raise ValueError(f"configs must map onto the {len(self)} matches")
        by_worker = defaultdict(lambda: [[]])
        for j, config in zip(indices, configs):
            by_worker[j // self.per_worker][0].append((j % self.per_worker, config))
            self.egos[j] = config.get("ego", 0)
        for w, replies in self._gather("reset", by_worker).items():
            for slot, obs, latency in replies:
                j = w * self.per_worker + slot
                self.observations[j], self.latencies[j] = obs, latency
        return [self.observations[j].state(self.egos[j]) for j in indices]

    def step(self, actions: list[dict] | None = None, ticks: int = 1, matches: list[int] | None = None) -> list[StepResult]:
        """actions[j] maps robot index -> Inputs | Goal | None for match matches[j] (default: all matches)."""
        indices = list(range(len(self))) if matches is None else list(matches)
        actions = actions if actions is not None else [{} for _ in indices]
        if len(actions) != len(indices):
            raise ValueError("one action dict per stepped match")
        if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 0xffff:
            raise ValueError("ticks must be an integer in [1, 65535]")
        by_worker = defaultdict(lambda: [{}, ticks])
        for j, given in zip(indices, actions):
            if self.observations[j] is None:
                raise RuntimeError("call reset before stepping")
            by_worker[j // self.per_worker][0][j % self.per_worker] = given
        for w, observed in self._gather("step", by_worker).items():
            for obs in observed:
                self.observations[w * self.per_worker + obs.slot] = obs
        results = []
        for j in indices:
            obs = self.observations[j]
            results.append(StepResult(obs.t, {i: obs.state(i) for i in range(len(obs.robots))}, obs.scores,
                                      obs.breakdowns, obs.phase == Phase.DONE, self.egos[j], obs))
        return results

    def state(self, j: int = 0) -> Observation:
        w, slot = divmod(j, self.per_worker)
        self.observations[j] = self.workers[w].call("state", [slot])[0]
        return self.observations[j]

    def close(self):
        for worker in getattr(self, "workers", []):
            worker.close()
        self.workers = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
