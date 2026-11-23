"""Read the numeric training archive as shared GameState, Inputs and Goal types."""
import numpy as np

from frsbot.bot.planner import Goal
from frsbot.core.field import ALLIANCES
from frsbot.core.inputs import Inputs
from frsbot.core.state import FuelState, GameState, HubState, Phase, RobotState


def examples(path):
    with np.load(path, allow_pickle=False) as run:
        if int(run["version"]) != 1:
            raise ValueError("unknown imitation dataset version")
        data = {key: run[key] for key in run.files}
    for k, t in enumerate(data["t"]):
        robots = [RobotState(
            int(id_), int(team), str(alliance), *map(float, data["pose"][k, i]),
            int(data["hopper"][k, i]), int(data["capacity"][i]),
        ) for i, (id_, team, alliance) in enumerate(zip(data["robot_id"], data["robot_team"], data["robot_alliance"]))]
        hubs = {a: HubState(bool(data["hub_active"][k, j]), float(data["hub_toggle"][k, j]))
                for j, a in enumerate(ALLIANCES)}
        fuel = FuelState(data["fuel_pos"][k], data["fuel_where"][k])
        for i in range(len(robots)):
            state = GameState(float(t), Phase(data["phase"][k]), hubs, robots, fuel, i)
            inputs = Inputs.decode(data["inputs"][k, i].tobytes())
            inputs.x, inputs.y, inputs.turn = map(float, data["axes"][k, i])
            residual = dict(zip(map(str, data["residual_names"]), map(float, data["residual"][k, i])))
            goal = Goal(str(data["goal_kind"][k, i]), tuple(map(float, data["goal_point"][k, i])))
            yield state, inputs, residual, goal
