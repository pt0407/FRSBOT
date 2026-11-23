"""Re-run tapped solo match configurations offline, with and without delay/hold."""
import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from frsbot.bot.imai import ImAI
from frsbot.core.specs import load_specs
from frsbot.core.state import Phase
from frsbot.sim.vec_env import Latency
from frsbot.sim.wasm_env import WasmEnv

RESET_KEYS = ('team', 'alliance', 'seed', 'start', 'auto', 'event', 'hp', 'unload', 'freeplay', 'countdown')


def load_run(path):
    path = Path(path)
    summary = json.loads(path.read_text())
    create, ready = summary.get('create'), summary.get('ready')
    if create is None:
        if not path.name.endswith('-summary.json'):
            raise ValueError('legacy summary must have a sibling <run>.jsonl and end in -summary.json')
        log = path.with_name(path.name.removesuffix('-summary.json') + '.jsonl')
        with log.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record.get('kind') == 'match':
                    create, ready = record.get('create'), record.get('ready')
                    break
    if not isinstance(create, dict) or any(k not in create for k in RESET_KEYS):
        raise ValueError('tapped create message is incomplete; refusing to substitute match defaults')
    if create['auto'] != 'none' or create['freeplay'] or create.get('practice') or create.get('builderAuto') or create.get('troll'):
        raise ValueError('only standard solo matches with auto none can be compared')
    if create.get('engine', 'wasm') != 'wasm':
        raise ValueError('live match did not use the WASM engine')
    if create['alliance'] not in ('blue', 'red'):
        raise ValueError('invalid alliance')
    for key in ('team', 'alliance', 'seed', 'start'):
        if summary[key] != create[key]:
            raise ValueError(f'summary and tapped create disagree on {key}')
    if not isinstance(create['start'], dict) or any(not math.isfinite(create['start'][k]) for k in ('fx', 'fy', 'headingDeg')):
        raise ValueError('start requires finite fx, fy, headingDeg')
    if any(not isinstance(create[k], bool) for k in ('hp', 'unload', 'freeplay', 'countdown')):
        raise ValueError('match switches must be booleans')
    return summary, create, ready


def timing_model(summary, delay_ms=None):
    latency = summary['latency']
    measured = latency.get('step_response', {}).get('median_ms')
    source = 'step_response_median'
    if measured is None:
        measured, source = latency['estimate_ms'], 'running_estimate'
    if delay_ms is not None:
        measured, source = delay_ms, 'override'
    rate = summary['pad_polls_per_second']
    if not math.isfinite(measured) or not 0 <= measured <= 10000 or not math.isfinite(rate) or rate <= 0:
        raise ValueError('latency must be finite 0..10000 ms and pad poll rate must be positive')
    hold = max(1, round(60 / rate))
    if hold > 60:
        raise ValueError('live poll interval exceeds backend maximum of 60 held ticks')
    return dict(requested_delay_ms=measured, delay_source=source, delay_ticks=round(measured * .06),
                hold_ticks=hold, pad_polls_per_second=rate)


def rollout(spec, create, model=None):
    delay_ticks, hold_ticks = (model['delay_ticks'], model['hold_ticks']) if model else (0, 1)
    delay = delay_ticks / 60
    latency = Latency((delay, delay), (hold_ticks, hold_ticks)) if model else None
    bot, goals, blocked, phase_scores = ImAI(spec), Counter(), Counter(), {}
    started, steps, previous_phase = time.monotonic(), 0, None
    with WasmEnv(latency=latency) as env:
        state = env.reset(**{k: create[k] for k in RESET_KEYS})
        initial = asdict(state.robots[state.me])
        limit = round((166 + max(0, -state.t)) * 60) + delay_ticks + 2
        while True:
            score = env.view['score']
            if state.phase != previous_phase:
                phase_scores[state.phase.value] = {a: s['total'] for a, s in score.items()}
                previous_phase = state.phase
            if state.phase == Phase.DONE:
                break
            if steps >= limit:
                raise TimeoutError('offline match did not reach done within the scheduled ticks')
            inputs = bot.act(state)
            inputs.climb = False
            if bot.goal is not None:
                goals[bot.goal.kind] += 1 / 60
            if inputs.score:
                for reason, value in (state.robots[state.me].gate or {}).items():
                    if reason != 'open' and value:
                        blocked[reason] += 1 / 60
            state = env.step(inputs, ticks=1)
            steps += 1
        result = dict(score=score[create['alliance']]['total'], final_score=score, phase_scores=phase_scores,
                      delay_ticks=env.delay_ticks, hold_ticks=env.hold_ticks, ticks=steps,
                      engine=env.engine, build=env.build, initial_robot=initial,
                      goal_seconds=dict(goals), blocked_gate_seconds=dict(blocked))
    return dict(**result, wall_seconds=time.monotonic() - started)


def compare(path, delay_ms=None):
    summary, create, ready = load_run(path)
    specs = {s.team: s for s in load_specs()}
    if create['team'] not in specs:
        raise ValueError(f"unknown team {create['team']}")
    model = timing_model(summary, delay_ms)
    ideal = rollout(specs[create['team']], create)
    degraded = rollout(specs[create['team']], create, model)
    build = ready.get('build') if ready else None
    return dict(summary=str(path), team=create['team'], alliance=create['alliance'], seed=create['seed'],
                driver=summary['driver'], create=create, live_ready=ready,
                engine_build_matches_live=ideal['build'] == build if build is not None else None,
                live_score=summary['final_score'][create['alliance']]['total'],
                live_latency=summary['latency']['step_response'], model=model,
                ideal=ideal, delayed=degraded,
                caveat='Same create configuration, not identical controls: no calibration or compensation offline. '
                       'Fixed observation delay and latest-input hold approximate, not reproduce, browser timing.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', type=Path, nargs='+')
    parser.add_argument('--delay-ms', type=float, help='override the measured observation delay')
    args = parser.parse_args()
    for path in args.summaries:
        try:
            result = compare(path, args.delay_ms)
        except (ValueError, KeyError, OSError) as error:
            parser.error(f'{path}: {error}')
        output = path.with_name(path.stem.removesuffix('-summary') + '-compare.json')
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
