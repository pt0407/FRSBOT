"""Print five-second straight-drive traces and batched WASM/JSON throughput."""
import argparse
import math
import time

from frsbot.core.inputs import Inputs
from frsbot.sim.wasm_env import WasmEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", type=int, nargs="+", default=[2910, 254])
    parser.add_argument("--ticks", type=int, default=3000)
    args = parser.parse_args()
    with WasmEnv() as env:
        for team in args.teams:
            state = env.reset(team, start=dict(fx=0, fy=0, headingDeg=0), seed=1)
            print(f"team={team} engine={env.engine} start={{fx:0,fy:0,headingDeg:0}} seed=1")
            print("t_s x_m z_m speed_m_s hopper")
            for second in range(6):
                if second:
                    state = env.step(Inputs(y=1, field_relative=True), ticks=60)
                robot = state.robots[state.me]
                print(f"{state.t:.1f} {robot.x:.4f} {robot.z:.4f} {math.hypot(robot.vx, robot.vz):.4f} {robot.hopper}")
            env.reset(team, seed=1)
            before = time.perf_counter()
            env.step(Inputs(), ticks=args.ticks)
            elapsed = time.perf_counter() - before
            print(f"throughput team={team} ticks={args.ticks} wall_s={elapsed:.4f} ticks_per_wall_s={args.ticks / elapsed:.1f} (one batched step; includes JSON/IPC; excludes reset)")


if __name__ == "__main__":
    main()
