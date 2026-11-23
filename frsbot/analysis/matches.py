"""Observational FUEL, role and defense analysis of decoded multiplayer matches."""
import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from frsbot.core.field import HUB_CENTRES
from frsbot.core.specs import load_specs
from frsbot.replay.convert import convert
from frsbot.replay.rbrp import parse


ALLIANCE_ZONE_EDGE = 4.2418


def in_zone(robot, spec, alliance):
    reach = (abs(math.cos(robot.a)) * spec.bumper_length + abs(math.sin(robot.a)) * spec.bumper_width) / 2
    return robot.x - reach <= -ALLIANCE_ZONE_EDGE if alliance == "blue" else robot.x + reach >= ALLIANCE_ZONE_EDGE


def contact(a, b, sa, sb):
    """2D oriented bumper overlap with 4 cm tolerance, not a physics contact claim."""
    delta = np.array([b.x - a.x, b.z - a.z])
    axes = [np.array([math.cos(r.a), math.sin(r.a)]) for r in (a, b)]
    bases = [(v, np.array([-v[1], v[0]])) for v in axes]
    halves = [(s.bumper_length / 2, s.bumper_width / 2) for s in (sa, sb)]
    for axis in (*bases[0], *bases[1]):
        radius = sum(h[0] * abs(axis @ v[0]) + h[1] * abs(axis @ v[1]) for h, v in zip(halves, bases))
        if abs(delta @ axis) > radius + .04:
            return False
    return True


def segment_distance(point, a, b):
    ab = b - a
    u = np.clip((point - a) @ ab / max(float(ab @ ab), 1e-9), 0, 1)
    return float(np.linalg.norm(point - a - u * ab))


def episodes(mask, times, dt, minimum=.5):
    result = []
    start = None
    for i, value in enumerate([*mask, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if sum(dt[start:i]) >= minimum:
                result.append((start, i))
            start = None
    return result


def analyze(states, *, events=(), metas=(), specs=None, partial=True):
    if not states:
        return {"robots": [], "duration_seconds": 0, "partial": True, "warnings": ["empty match"]}
    times = np.array([s.t for s in states])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("match times must be finite and strictly increasing")
    roster = states[0].robots
    if not roster:
        raise ValueError("match requires at least one robot")
    ids = [r.id for r in roster]
    identity = [(r.id, r.team, r.alliance) for r in roster]
    if len(set(ids)) != len(ids) or any([(r.id, r.team, r.alliance) for r in s.robots] != identity for s in states):
        raise ValueError("robot roster must be unique and stable")
    n, nr = len(states), len(ids)
    by_id = {r.id: i for i, r in enumerate(roster)}
    specs = {s.team: s for s in (load_specs() if specs is None else specs)}
    robot_specs = [specs[r.team] for r in roster]
    nf = len(states[0].fuel.where)
    if any(s.fuel.pos.shape != (nf, 3) or s.fuel.where.shape != (nf,) for s in states):
        raise ValueError("FUEL identities and shapes must remain stable")
    dt = np.maximum(0, np.minimum(times[1:], 163) - np.maximum(times[:-1], 0))
    dt = np.r_[dt, 0.]
    by_frame = defaultdict(list)
    score_events = []
    tally = None
    for index, event in events:
        if not 0 <= index < n:
            continue
        by_frame[index].append(event)
        if event["t"] == "score":
            score_events.append((index, event))
        elif event["t"] == "tally":
            tally = event
    timeline = [{"t": float(times[min(index, n - 1)]), "score": m["score"]}
                for index, m in metas if "score" in m and 0 <= index < n]
    score_times = [m["t"] for m in timeline]

    def score_at(t, alliance):
        at = bisect.bisect_right(score_times, t) - 1
        if at < 0:
            return 0
        score = timeline[at]["score"]
        other = "red" if alliance == "blue" else "blue"
        return score[alliance]["total"] - score[other]["total"]

    positions = np.array([[[r.x, r.z] for r in s.robots] for s in states])
    near = np.zeros((n, nr), bool)
    touching = np.zeros((n, nr), bool)
    lane = np.zeros((n, nr), bool)
    pair_seconds, pair_contacts = Counter(), Counter()
    opportunity = np.zeros((n, nr), bool)
    for k, state in enumerate(states):
        for i, r in enumerate(state.robots):
            hub = np.array(HUB_CENTRES[r.alliance][:2])
            distance = np.linalg.norm(positions[k, i] - hub)
            spec = robot_specs[i]
            own_zone = in_zone(r, spec, r.alliance)
            opportunity[k, i] = (r.hopper > 0 and state.hubs[r.alliance].active and own_zone
                                 and distance >= (spec.min_range or 0)
                                 and distance <= (spec.max_range or math.inf))
            for j, other in enumerate(state.robots):
                if other.alliance == r.alliance:
                    continue
                close = np.linalg.norm(positions[k, i] - positions[k, j]) <= 1.5
                hit = close and contact(r, other, spec, robot_specs[j])
                near[k, i] |= close
                touching[k, i] |= hit
                pair_seconds[(r.id, other.id)] += dt[k] * close
                pair_contacts[(r.id, other.id)] += dt[k] * hit
                other_hub = np.array(HUB_CENTRES[other.alliance][:2])
                if other.hopper and state.hubs[other.alliance].active:
                    lane[k, i] |= segment_distance(positions[k, i], positions[k, j], other_hub) <= .7

    holders = np.full(nf, -1, int)
    flights = {}
    hints = {}
    launches, entries, transfers = [], [], []
    collected = np.zeros((n, nr), int)
    initial = np.zeros(nr, int)
    ambiguous = 0
    previous = None
    for k, state in enumerate(states):
        for e in by_frame[k]:
            if e["t"] == "intake" and e.get("robot") in by_id:
                hints[e["fuel"]] = (by_id[e["robot"]], k)
        where = state.fuel.where
        current = np.full(nf, -1, int)
        carried = np.flatnonzero(where == 2)
        if len(carried):
            distances = np.linalg.norm(state.fuel.pos[carried][:, [0, 2]][:, None, :] - positions[k][None, :, :], axis=2)
            for row, fuel in enumerate(carried):
                order = np.argsort(distances[row])
                i = int(order[0])
                spec = robot_specs[i]
                radius = math.hypot(spec.bumper_length, spec.bumper_width) / 2 + .35
                hint = hints.get(int(fuel))
                if hint and k - hint[1] <= 6:
                    current[fuel] = hint[0]
                elif holders[fuel] >= 0:
                    current[fuel] = holders[fuel]
                elif distances[row, i] <= radius and (nr == 1 or distances[row, order[1]] - distances[row, i] >= .15):
                    current[fuel] = i
                else:
                    ambiguous += 1
        if previous is None:
            for i in current[current >= 0]:
                initial[i] += 1
        else:
            for fuel in np.flatnonzero((previous == 2) & (where == 1)):
                i = int(holders[fuel])
                r = state.robots[i] if i >= 0 else None
                hub_shot = bool(r and in_zone(r, robot_specs[i], r.alliance))
                launch = {"fuel": int(fuel), "robot_index": i, "frame": k, "t": float(times[k]),
                          "hub_shot_proxy": hub_shot, "position": positions[k, i].tolist() if i >= 0 else None}
                launches.append(launch)
                flights[int(fuel)] = launch
            for fuel in np.flatnonzero((previous != 2) & (where == 2)):
                i = int(current[fuel])
                if i >= 0:
                    collected[k, i] += 1
                    launch = flights.get(int(fuel))
                    if (launch and not launch.get("entered_hub")
                            and launch["robot_index"] >= 0 and launch["robot_index"] != i
                            and 0 <= times[k] - launch["t"] <= 10
                            and roster[launch["robot_index"]].alliance == roster[i].alliance):
                        transfers.append({"from": ids[launch["robot_index"]], "to": ids[i], "t": float(times[k]),
                                          "fuel": int(fuel), "launch_t": launch["t"]})
                flights.pop(int(fuel), None)
            for fuel in np.flatnonzero((previous != 3) & (where == 3)):
                launch = flights.get(int(fuel))
                if launch and times[k] - launch["t"] <= 8:
                    launch["entered_hub"] = True
                    alliance = min(HUB_CENTRES, key=lambda a: np.linalg.norm(
                        state.fuel.pos[fuel, [0, 2]] - np.array(HUB_CENTRES[a][:2])))
                    entries.append({**launch, "entry_frame": k, "entry_t": float(times[k]), "alliance": alliance,
                                    "active": state.hubs[alliance].active})
            for fuel in np.flatnonzero(np.isin(where, [4, 5, 6])):
                flights.pop(int(fuel), None)
        holders, previous = current, where

    matched = set()
    consumed_launches = set()
    credited = []
    entries_by_fuel = defaultdict(list)
    for i, entry in enumerate(entries):
        entries_by_fuel[entry["fuel"]].append((i, entry))
    unmatched = Counter()
    for frame, event in score_events:
        candidates = [(i, e) for i, e in entries_by_fuel[event["fuel"]]
                      if i not in matched and e["alliance"] == event["alliance"]
                      and (e["fuel"], e["frame"]) not in consumed_launches
                      and -.05 <= times[frame] - e["entry_t"] <= 5]
        if candidates:
            i, entry = min(candidates, key=lambda v: abs(v[1]["entry_t"] - times[frame]))
            matched.add(i)
            consumed_launches.add((entry["fuel"], entry["frame"]))
            if entry["robot_index"] >= 0:
                credited.append({**entry, "points": event.get("points", 0)})
                continue
        unmatched[event["alliance"]] += event.get("points", 0)
    if not score_events:
        for entry in entries:
            key = (entry["fuel"], entry["frame"])
            if entry["robot_index"] >= 0 and key not in consumed_launches:
                credited.append({**entry, "points": int(entry["active"])})
                consumed_launches.add(key)

    shooting = np.zeros((n, nr), bool)
    passing = np.zeros((n, nr), bool)
    collecting = np.zeros((n, nr), bool)
    for launch in launches:
        i, k = launch["robot_index"], launch["frame"]
        if i >= 0:
            end = np.searchsorted(times, times[k] + 1)
            (shooting if launch["hub_shot_proxy"] else passing)[k:end, i] = True
    for k, i in zip(*np.nonzero(collected)):
        collecting[k:np.searchsorted(times, times[k] + 1), i] = True
    roles = np.full((n, nr), "transit", dtype=object)
    for k, state in enumerate(states):
        for i, r in enumerate(state.robots):
            opponent_zone = in_zone(r, robot_specs[i], "red" if r.alliance == "blue" else "blue")
            if near[k, i] or lane[k, i] or opponent_zone:
                roles[k, i] = "defender"
    roles[collecting], roles[passing], roles[shooting] = "collector", "feeder", "scorer"
    official = {r["robot"]: r for r in tally["robots"]} if tally else {}
    strength = {r.id: official.get(r.id, {}).get("autoFuel", 0) + official.get(r.id, {}).get("teleFuel", 0)
                if r.id in official else sum(e["points"] for e in credited
                                             if e["robot_index"] == i and e["alliance"] == r.alliance)
                for i, r in enumerate(roster)}
    summaries = []
    for i, r in enumerate(roster):
        all_shots = [e for e in credited if e["robot_index"] == i]
        shots = [e for e in all_shots if e["alliance"] == r.alliance]
        def rate(mask):
            seconds = float(dt[opportunity[:, i] & mask].sum())
            points = sum(e["points"] for e in shots if opportunity[max(0, e["frame"] - 1), i]
                         and mask[max(0, e["frame"] - 1)])
            return {"seconds": seconds, "fuel": points, "fuel_per_second": points / seconds if seconds else None}
        defended, clear = rate(near[:, i]), rate(~near[:, i])
        loss = (clear["fuel_per_second"] - defended["fuel_per_second"]
                if clear["fuel_per_second"] is not None and defended["fuel_per_second"] is not None else None)
        defense_starts = []
        for start, end in episodes((roles[:, i] == "defender") & (dt > 0), times, dt, 1):
            defense_starts.append({"t": float(times[start]), "seconds": float(dt[start:end].sum()),
                                   "time_left": max(0, 163 - float(times[start])),
                                   "score_difference": score_at(times[start], r.alliance),
                                   "own_hub_active": states[start].hubs[r.alliance].active,
                                   "position": positions[start, i].tolist()})
        responses = []
        for start, end in episodes(near[:, i] & (dt > 0), times, dt):
            before = max(0, np.searchsorted(times, times[start] - 2))
            after = min(end - 1, np.searchsorted(times, times[start] + 2))
            v1 = positions[start, i] - positions[before, i]
            v2 = positions[after, i] - positions[start, i]
            turn = (math.degrees(math.acos(float(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
                    if np.linalg.norm(v1) > .4 and np.linalg.norm(v2) > .4 else None)
            old_spots = [s["position"] for s in shots if times[start] - 5 <= s["t"] < times[start]]
            new_spots = [s["position"] for s in shots if times[start] <= s["t"] < times[min(end, n - 1)]]
            shift = float(np.linalg.norm(np.mean(new_spots, axis=0) - np.mean(old_spots, axis=0))) if old_spots and new_spots else None
            responses.append({"t": float(times[start]), "seconds": float(dt[start:end].sum()),
                              "route_turn_degrees": turn, "shot_spot_shift_m": shift,
                              "pass_launches": sum(not l["hub_shot_proxy"] and l["robot_index"] == i
                                                   and start <= l["frame"] < end for l in launches),
                              "contact_proxy_seconds": float(dt[start:end][touching[start:end, i]].sum())})
        summaries.append({"id": r.id, "team": r.team, "alliance": r.alliance,
                          "spec": {"fire_rate": robot_specs[i].fire_rate, "intake_rate": robot_specs[i].intake_rate,
                                   "max_speed": robot_specs[i].max_speed, "capacity": robot_specs[i].total_capacity},
                          "scoring_rank": 1 + sum(strength[q.id] > strength[r.id] for q in roster if q.alliance == r.alliance),
                          "tracked_active_fuel": sum(e["points"] for e in shots), "tracked_hub_entries": len(shots),
                          "opponent_hub_scores": [e for e in all_shots if e["alliance"] != r.alliance and e["points"]],
                          "engine_tally": official.get(r.id), "initial_carried": int(initial[i]),
                          "collected": int(collected[:, i].sum()),
                          "pass_launch_proxy": sum(l["robot_index"] == i and not l["hub_shot_proxy"] for l in launches),
                          "completed_ally_transfers": sum(t["from"] == r.id for t in transfers),
                          "role_seconds": {role: float(dt[roles[:, i] == role].sum())
                                           for role in ("scorer", "collector", "feeder", "defender", "transit")},
                          "near_opponent_seconds": float(dt[near[:, i]].sum()),
                          "contact_proxy_seconds": float(dt[touching[:, i]].sum()),
                          "defended_rate": defended, "undefended_rate": clear,
                          "rate_loss_fuel_per_second": loss,
                          "estimated_loss_fuel": loss * defended["seconds"] if loss is not None else None,
                          "defense_starts": defense_starts, "defended_responses": responses})
    return {"partial": partial, "duration_seconds": float(dt.sum()), "robots": summaries,
            "score_timeline": timeline, "final_score": timeline[-1]["score"] if timeline else None,
            "winner": (max(timeline[-1]["score"], key=lambda a: timeline[-1]["score"][a]["total"])
                       if timeline and len({s["total"] for s in timeline[-1]["score"].values()}) > 1 else None),
            "unattributed_active_fuel": dict(unmatched), "ambiguous_holder_frames": ambiguous,
            "credited_score_source": "score events matched to tracked hub entries" if score_events else "hub-entry active flags",
            "score_event_count": len(score_events), "tracked_entry_count": len(entries),
            "transfers": transfers,
            "pairs": [{"robot": a, "opponent": b, "near_seconds": seconds, "contact_proxy_seconds": pair_contacts[a, b]}
                      for (a, b), seconds in pair_seconds.items()],
            "hub_windows": {a: [{"t": float(times[k]), "active": s.hubs[a].active} for k, s in enumerate(states)
                                if k == 0 or s.hubs[a].active != states[k - 1].hubs[a].active] for a in HUB_CENTRES},
            "fouls": [{"time": float(times[k]), **e} for k, e in events if 0 <= k < n and e["t"] == "foul"],
            "blockades": [{"time": float(times[k]), **e} for k, e in events if 0 <= k < n and e["t"] == "blockade"]}


def aggregate(matches):
    groups = defaultdict(lambda: {"robots": 0, "role_seconds": Counter(), "defense_starts": []})
    for match in matches:
        for r in match["robots"]:
            group = groups[r["scoring_rank"]]
            group["robots"] += 1
            group["role_seconds"].update(r["role_seconds"])
            group["defense_starts"].extend(r["defense_starts"])
    for group in groups.values():
        total = sum(group["role_seconds"].values())
        group["role_fraction"] = {role: seconds / total if total else 0 for role, seconds in group["role_seconds"].items()}
        starts = group["defense_starts"]
        group["switch_context"] = {
            "leading": sum(s["score_difference"] > 0 for s in starts),
            "trailing": sum(s["score_difference"] < 0 for s in starts),
            "tied": sum(s["score_difference"] == 0 for s in starts),
            "own_hub_inactive": sum(not s["own_hub_active"] for s in starts),
            "mean_time_left": sum(s["time_left"] for s in starts) / len(starts) if starts else None}
    return {"matches": len(matches), "by_scoring_rank": dict(groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in args.replays:
        rep = parse(path.read_bytes(), fuel=True)
        result = analyze(convert(rep), events=rep.events, metas=rep.metas, partial=rep.meta.get("partial", True))
        result["source"] = str(path)
        results.append(result)
        for r in result["robots"]:
            print(f"{path.stem} id={r['id']} {r['alliance']} team={r['team']} "
                  f"tracked_fuel={r['tracked_active_fuel']} collected={r['collected']} "
                  f"defender_s={r['role_seconds']['defender']:.2f} nearby_s={r['near_opponent_seconds']:.2f}")
    args.output.write_text(json.dumps({"matches": results, "aggregate": aggregate(results)}, indent=2, allow_nan=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
