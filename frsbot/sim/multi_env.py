"""Multi-robot matches using the original JSON engine transport.

Stand-in for the throughput branch's WasmEnv(robots=...); replace it on merge.
"""
from dataclasses import asdict, dataclass, field, replace

from frsbot.core.inputs import Inputs
from frsbot.core.state import GameState, Phase
from frsbot.sim.wasm_env import WasmEnv, view_to_state


@dataclass
class RobotSlot:
    team: int
    alliance: str = "blue"
    start: dict | None = None
    policy: object | None = None
    preload: int = 8


@dataclass
class StepResult:
    t: float
    views: dict[int, GameState]
    scores: dict[str, int]
    breakdowns: dict[str, dict[str, float]]
    done: bool
    ego: int
    observation: object = field(repr=False)
    events: list[dict] = field(default_factory=list)


def default_starts(robots: list[RobotSlot]) -> list[dict]:
    lanes = {"blue": 0, "red": 0}
    starts = []
    for robot in robots:
        if robot.start is not None:
            starts.append(robot.start)
            continue
        lane = lanes[robot.alliance]
        lanes[robot.alliance] += 1
        fy = (2.0, 6.07, 4.035)[lane % 3]
        starts.append(dict(fx=3.6, fy=fy, headingDeg=0) if robot.alliance == "blue" else
                      dict(fx=12.94, fy=fy, headingDeg=180))
    return starts


class _Observation:
    def __init__(self, state, raw, ego):
        self.t, self.phase, self.hubs = state.t, state.phase, state.hubs
        self.robots, self.fuel, self.ego = state.robots, state.fuel, ego
        self.breakdowns = raw["score"]
        self.scores = {a: int(s["total"]) for a, s in self.breakdowns.items()}
        self._state, self._raw = state, raw
        scored = {d["robot"]: d["fuel"] for d in raw.get("drivers", [])}
        self.driver = [dict(abandoned=False, recovering=False, stage=0, scored=scored.get(r.id, 0), command=0)
                       for r in self.robots]

    def state(self, robot=None):
        return replace(self._state, me=self.ego if robot is None else robot)

    def view(self):
        return self._raw


class MultiWasmEnv(WasmEnv):
    def reset(self, robots: list[RobotSlot], ego: int = 0, seed: int = 1, auto: str = "none",
              event: str = "event", hp: bool = True, unload: bool = True,
              freeplay: bool = False, countdown: bool = False) -> GameState:
        if not isinstance(robots, list) or not 1 <= len(robots) <= 6:
            raise ValueError("robots requires 1 to 6 slots")
        self.slots = [r if isinstance(r, RobotSlot) else RobotSlot(**r) for r in robots]
        if isinstance(ego, bool) or not isinstance(ego, int) or not 0 <= ego < len(robots):
            raise ValueError("unknown ego index")
        if any(r.alliance not in ("blue", "red") for r in self.slots):
            raise ValueError("alliance must be blue or red")
        slots = [dict(team=r.team, alliance=r.alliance, start=s, preload=r.preload)
                 for r, s in zip(self.slots, default_starts(self.slots))]
        if self.process is None:
            self._spawn()
        self._request("new", config=dict(robots=slots, seed=seed, auto=auto, event=event,
                                         hp=hp, unload=unload, freeplay=freeplay))
        self._freeplay, self.ego = freeplay, ego
        self._request("start", countdown=countdown)
        return self._result().views[ego]

    def _result(self):
        self.observation = _Observation(view_to_state(self.view, self._freeplay), self.view, self.ego)
        obs = self.observation
        return StepResult(obs.t, {i: obs.state(i) for i in range(len(obs.robots))}, obs.scores,
                          obs.breakdowns, obs.phase == Phase.DONE, self.ego, obs, list(self.view.get("events", [])))

    def step(self, inputs: dict[int, Inputs | None], ticks: int = 1) -> StepResult:
        if self.view is None:
            raise RuntimeError("call reset before stepping")
        if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 60000:
            raise ValueError("ticks must be an integer in [1, 60000]")
        if not isinstance(inputs, dict):
            raise ValueError("inputs must map robot indices to Inputs or None")
        count = len(self.view["robots"])
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in inputs):
            raise ValueError("unknown robot index")
        actions, batches = dict(inputs), {}
        for i, slot in enumerate(self.slots):
            if i in inputs or slot.policy is None:
                continue
            if hasattr(slot.policy, "act_batch"):
                batches.setdefault(id(slot.policy), (slot.policy, []))[1].append(i)
            else:
                actions[i] = slot.policy.act(self.observation.state(i), self.observation.driver[i])
        for policy, indices in batches.values():
            values = policy.act_batch([self.observation.state(i) for i in indices],
                                      [self.observation.driver[i] for i in indices])
            actions.update(zip(indices, values))
        packet = {}
        for i, robot in enumerate(self.view["robots"]):
            action = actions.get(i)
            if action is not None and not isinstance(action, Inputs):
                raise TypeError("action must be Inputs or None")
            raw = asdict(action if action is not None else Inputs())
            raw["fieldRelative"] = raw.pop("field_relative")
            raw.pop("seq")
            packet[str(robot["id"])] = raw
        self._request("step", inputs=packet, n=ticks)
        return self._result()

    def state(self) -> GameState:
        if self.view is None:
            raise RuntimeError("call reset before using MultiWasmEnv")
        self._request("state")
        return self._result().views[self.ego]
