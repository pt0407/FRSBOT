"""PPO fine-tune of the goal-conditioned drive net in the real engine.

Initialised from the behaviour-cloned drive_policy.pt. Reward is the ego
alliance's score delta per control step; the rule driver keeps terminals,
buttons and recovery, so the net learns speeds/arcs where BC could not servo.
One full match per env per iteration; weights are re-exported for live use.
Usage: python -m frsbot.imitation.train_rl [--iterations 400] [--parallel 8]
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from frsbot.bot.imai import ImAI
from frsbot.bot.learned_drive import MODEL
from frsbot.bot.planner import Goal
from frsbot.bot.tuning import PARAMS
from frsbot.core.specs import load_specs
from frsbot.imitation.train_drive import features
from frsbot.sim.vec_env import RobotSlot, VecWasmEnv

ROOT = Path(__file__).resolve().parents[2]
SPECS = {s.team: s for s in load_specs()}
OUT = ROOT / "data" / "models"


def policy_net():
    return nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))


def value_net():
    return nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--teams", nargs="+", type=int, default=[254, 4414, 1678, 2481])
    ap.add_argument("--blend", type=float, default=.6)
    ap.add_argument("--std", type=float, default=.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    PARAMS["learned_drive"] = 1
    PARAMS["learned_blend"] = args.blend
    PARAMS["learned_turn_blend"] = 0

    policy = policy_net()
    if MODEL.exists():
        policy.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
        print("initialised from BC drive_policy.pt", flush=True)
    value = value_net()
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(value.parameters(), lr=1e-3)

    workers = max(1, int(math.sqrt(args.parallel)))
    per = max(1, args.parallel // workers)
    env = VecWasmEnv(n_workers=workers, matches_per_worker=per)
    n = workers * per
    print(f"{n} parallel matches ({workers} workers x {per})", flush=True)

    bots = {}
    seeds = np.arange(args.seed, args.seed + n * 4)
    seed_at = 0

    def reset_all():
        nonlocal seed_at
        configs = []
        for j in range(n):
            team = args.teams[j % len(args.teams)]
            bots[j] = ImAI(SPECS[team])
            configs.append(dict(robots=[RobotSlot(team=team)], seed=int(seeds[seed_at % len(seeds)])))
            seed_at += 1
        return env.reset(configs)

    states = reset_all()
    best_mean, t_start = 0., time.time()
    for iteration in range(1, args.iterations + 1):
        traj = [[] for _ in range(n)]  # per match: (feat, action, rule, learned, reward, done)
        prev_score = [0.] * n
        alliance = [None] * n
        done = [False] * n
        t0 = time.time()
        while not all(done):
            actions, batch_feat = [], []
            live = [j for j in range(n) if not done[j]]
            for j in live:
                inputs = bots[j].act(states[j])
                steer = bots[j].driver.last_steer
                goal = Goal(steer["kind"], tuple(map(float, steer["point"])))
                noise = rng.normal(0, args.std, 2).astype(np.float32) if steer["learned"] else np.zeros(2, np.float32)
                action = np.clip(np.array([inputs.x, inputs.y], np.float32) + noise, -1, 1)
                alliance[j] = alliance[j] or states[j].robots[states[j].me].alliance
                traj[j].append((features(states[j], SPECS[states[j].robots[states[j].me].team], goal),
                                action.copy(), tuple(map(float, steer["rule"])), steer["learned"]))
                inputs.x, inputs.y = map(float, action)
                actions.append({states[j].me: inputs})
            padded = actions + [{}] * (n - len(actions))
            results = env.step(padded, 3)
            by_match = {r.ego: r for r in results}
            for k, j in enumerate(live):
                r = results[j]
                states[j] = r.views[r.ego]
                # Reward must be robot-attributed: the alliance total includes
                # HP human-player scoring, action-independent noise that swamps
                # the policy gradient (measured: score drifted down over 18
                # iterations while value loss rose).
                total = float(r.observation.driver[r.ego]["scored"]) if r.observation and r.observation.driver else r.breakdowns[alliance[j]]["total"]
                reward = total - prev_score[j]
                prev_score[j] = total
                end = r.done or r.t >= 163
                traj[j][-1] += (reward, end)
                if end:
                    done[j] = True
        # flatten batch
        feats = torch.tensor(np.array([s[0] for t in traj for s in t], np.float32))
        acts = torch.tensor(np.array([s[1] for t in traj for s in t], np.float32))
        rules = torch.tensor(np.array([s[2] for t in traj for s in t], np.float32))
        mask = torch.tensor(np.array([s[3] for t in traj for s in t], np.float32))
        rews, dones, lens = [], [], [len(t) for t in traj]
        for t in traj:
            rews += [s[4] for s in t]
            dones += [s[5] for s in t]
        rews_t = torch.tensor(rews, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        # GAE per episode
        with torch.no_grad():
            vals = value(feats).squeeze(-1)
        adv = torch.zeros_like(rews_t)
        lastgae, pos = 0., 0
        gamma, lam = .997, .95
        for length in lens:
            for i in range(pos + length - 1, pos - 1, -1):
                terminal = dones_t[i].item()
                nextv = 0. if terminal or i == pos + length - 1 else vals[i + 1].item()
                delta = rews_t[i].item() + gamma * nextv * (1 - terminal) - vals[i].item()
                lastgae = delta + gamma * lam * (1 - terminal) * (lastgae if i < pos + length - 1 else 0.)
                adv[i] = lastgae
            lastgae, pos = 0., pos + length
        ret = adv + vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # PPO update; reference log-probs come from the rollout-time policy.
        # The executed x/y mean is blend*net + (1-blend)*rule (turn stays rule).
        var = args.std ** 2

        def means(net_out, rule_xy):
            # No clamp here: clamping the mean zeroes the gradient exactly where
            # sticks saturate, and this domain lives at +-1. Actions are clipped
            # only when sent to the engine.
            return args.blend * net_out[:, :2] + (1 - args.blend) * rule_xy
        with torch.no_grad():
            logp_old = -((acts - means(policy(feats), rules)) ** 2).sum(-1) / (2 * var)
        pl_losses, vl_losses = [], []
        for _ in range(args.epochs):
            order = torch.randperm(len(feats))
            for s in range(0, len(feats), 4096):
                idx = order[s:s + 4096]
                mean = means(policy(feats[idx]), rules[idx])
                logp = -((acts[idx] - mean) ** 2).sum(-1) / (2 * var)
                ratio = torch.exp(logp - logp_old[idx])
                a = adv[idx] * mask[idx]
                surr = torch.min(ratio * a, ratio.clamp(1 - .2, 1 + .2) * a)
                pl = -surr.mean()
                opt_p.zero_grad(); pl.backward(); opt_p.step()
                vl = ((value(feats[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
                opt_v.zero_grad(); vl.backward(); opt_v.step()
                pl_losses.append(pl.item()); vl_losses.append(vl.item())
        total_score = sum(prev_score) / n
        best_mean = max(best_mean, total_score)
        if total_score >= best_mean:
            # Only overwrite the production weights when the noisy rollout mean
            # improves; a drifting run must never destroy the BC init.
            torch.save(policy.state_dict(), MODEL)
        print(f"iter {iteration}: mean score {total_score:.0f} best {best_mean:.0f} "
              f"rollout {time.time() - t0:.0f}s pl {np.mean(pl_losses):.4f} vl {np.mean(vl_losses):.2f} "
              f"elapsed {(time.time() - t_start) / 60:.0f}m", flush=True)
        states = reset_all()
    env.close()


if __name__ == "__main__":
    main()
