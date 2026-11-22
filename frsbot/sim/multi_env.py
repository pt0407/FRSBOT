"""Multi-robot real-engine matches, with one ordered ego observation per slot."""
from dataclasses import asdict, replace

from frsbot.core.inputs import Inputs
from frsbot.core.state import GameState
from frsbot.sim.wasm_env import WasmEnv, view_to_state


class MultiWasmEnv(WasmEnv):
    def reset(self, robots: list[dict], seed: int = 1, auto: str = "none",
              event: str = "event", hp: bool = True, unload: bool = True,
              freeplay: bool = False, countdown: bool = False) -> list[GameState]:
        if not isinstance(robots, list) or not 1 <= len(robots) <= 6:
            raise ValueError("robots requires 1 to 6 slots")
        slots = [dict(team=r["team"], alliance=r["alliance"],
                      start=r.get("start", dict(fx=0, fy=0, headingDeg=0)),
                      preload=r.get("preload", 8)) for r in robots]
        if self.process is None:
            self._spawn()
        self._request("new", config=dict(robots=slots, seed=seed, auto=auto, event=event,
                                         hp=hp, unload=unload, freeplay=freeplay))
        self._freeplay = freeplay
        self._request("start", countdown=countdown)
        return self._states()

    def _states(self) -> list[GameState]:
        state = view_to_state(self.view, self._freeplay)
        return [replace(state, me=i) for i in range(len(state.robots))]

    def step(self, inputs: dict[int, Inputs], ticks: int = 1) -> list[GameState]:
        if self.view is None:
            raise RuntimeError("call reset before stepping")
        if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 60000:
            raise ValueError("ticks must be an integer in [1, 60000]")
        if not isinstance(inputs, dict):
            raise ValueError("inputs must map robot indices to Inputs")
        count = len(self.view["robots"])
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in inputs):
            raise ValueError("unknown robot index")
        packet = {}
        for i, robot in enumerate(self.view["robots"]):
            raw = asdict(inputs.get(i, Inputs()))
            raw["fieldRelative"] = raw.pop("field_relative")
            raw.pop("seq")
            packet[str(robot["id"])] = raw
        self._request("step", inputs=packet, n=ticks)
        return self._states()

    def state(self) -> list[GameState]:
        self._request("state")
        return self._states()
