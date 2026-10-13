#!/usr/bin/env python3
"""Measure how well the local Laya server makes this robot's decisions.

Two labelled suites, built from the same state and question code the robot uses:
  driving  synthetic sensor/waypoint situations -> acceptable motion primitives
  heldout  generated states (data/driving_eval.jsonl) -> the code driver's exact choice
  tasks    spoken instructions -> expected skill, place and direction; plus replace/queue
  tasks-holdout  20 phrasings kept out of all tuning

  python tools/eval_laya.py                     # both suites, default checkpoint routing
  python tools/eval_laya.py --model typed-decisions --suite driving
  python tools/eval_laya.py --mock              # sanity-check the harness offline

Reports accuracy, mean top-confidence, accuracy above the pilot's confidence bar, and latency.
"""

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'create3_laya'))

from create3_laya import create3  # noqa: E402
from create3_laya.laya_client import LayaClient, top_confidence  # noqa: E402
from create3_laya.mock_laya import MockLaya  # noqa: E402
from create3_laya.pilot import LayaPilot, Snapshot  # noqa: E402
from create3_laya.tasks import (DOCK_DESCRIPTION, START_DESCRIPTION,  # noqa: E402
                                TaskInterpreter, load_places)

DIRECTIONS = {   # waypoint bearing in degrees (left positive) -> label used in the state
    'straight ahead': 0, 'ahead, slightly left': 25, 'ahead, slightly right': -25,
    'to the left': 70, 'to the right': -70, 'behind, on the left': 130,
    'behind, on the right': -130, 'directly behind': 180,
}
OBSTACLES = {    # name -> {sensor: range in m}; unlisted sensors are clear
    'clear': {},
    'front far': {'front_center_left': 0.15},
    'front near': {'front_center_left': 0.07, 'front_left': 0.12, 'front_center_right': 0.12},
    'front very close': {'front_center_left': 0.03, 'front_left': 0.06,
                         'front_center_right': 0.06},
    'left side near': {'side_left': 0.07, 'left': 0.08},
    'right side near': {'right': 0.07, 'front_right': 0.08},
    'front and left blocked': {'front_center_left': 0.06, 'front_left': 0.05, 'left': 0.06},
    'front and right blocked': {'front_center_left': 0.06, 'front_center_right': 0.05,
                                'front_right': 0.06},
}
# instruction -> (skill, place or None, direction or None)
TASKS = [
    ('go to the east hall', 'go_to', 'east_hall', None),
    ('drive over to the north west room', 'go_to', 'north_west_room', None),
    ('head to the room on the west side', 'go_to', 'west_room', None),
    ('please visit the south east room', 'go_to', 'south_east_room', None),
    ('navigate to the top right corner', 'go_to', 'north_east_corner', None),
    ('take me to the bottom left corner', 'go_to', 'south_west_corner', None),
    ('go to the south room', 'go_to', 'south_room', None),
    ('move to the east room', 'go_to', 'east_room', None),
    ('can you go to the big hall on the right', 'go_to', 'east_hall', None),
    ('go to the north-east corner slowly', 'go_to', 'north_east_corner', None),
    ('return to the dock', 'return_to_dock', None, None),
    ('go home and charge', 'return_to_dock', None, None),
    ('come back to your charger', 'return_to_dock', None, None),
    ('your battery is low, go dock', 'return_to_dock', None, None),
    ('turn left', 'rotate', None, 'left'),
    ('spin right 45 degrees', 'rotate', None, 'right'),
    ('rotate 180 degrees to the left', 'rotate', None, 'left'),
    ('turn around', 'rotate', None, None),
    ('do a full spin to the right', 'rotate', None, 'right'),
    ('drive forward one meter', 'drive', None, 'forward'),
    ('move ahead 50 cm', 'drive', None, 'forward'),
    ('back up a little', 'drive', None, 'backward'),
    ('reverse 30 centimeters', 'drive', None, 'backward'),
    ('explore the maze', 'explore', None, None),
    ('wander around for a bit', 'explore', None, None),
    ('look around the place', 'explore', None, None),
    ('wait 5 seconds', 'wait', None, None),
    ('pause for a moment', 'wait', None, None),
    ('stop', 'stop', None, None),
    ('halt right now', 'stop', None, None),
    ('cancel that', 'stop', None, None),
    ('pick up the red cup', 'unsupported', None, None),
    ('clean the kitchen floor', 'unsupported', None, None),
    ('sing me a song', 'unsupported', None, None),
    ('open the door', 'unsupported', None, None),
]
# Phrasings never used while tuning the interpreter's questions and rules. Scored 18/20 on
# typed-decisions when first run (2026-10-10); both misses fell below the confidence bar, so
# the robot asked for clarification instead of acting. Keep it untouched as a holdout. 19/20
# after the unsure-skill/sure-place rule (tuned on the depot phrasings, not on these).
TASKS_HOLDOUT = [
    ('go over to the dock', 'return_to_dock', None, None),
    ('please travel to the east hall', 'go_to', 'east_hall', None),
    ('swing by the south room', 'go_to', 'south_room', None),
    ('drive into the north-west room', 'go_to', 'north_west_room', None),
    ('can you reach the top-right corner?', 'go_to', 'north_east_corner', None),
    ('find your way to the south west corner', 'go_to', 'south_west_corner', None),
    ('go dock yourself', 'return_to_dock', None, None),
    ('rotate left by 60 degrees', 'rotate', None, 'left'),
    ('make a quarter turn to the right', 'rotate', None, 'right'),
    ('spin around', 'rotate', None, None),
    ('inch forward a little', 'drive', None, 'forward'),
    ('back away 40 cm', 'drive', None, 'backward'),
    ('go straight ahead 2 meters', 'drive', None, 'forward'),
    ('take a tour of the maze', 'explore', None, None),
    ('wander about', 'explore', None, None),
    ('wait here for 20 seconds', 'wait', None, None),
    ('stop right there', 'stop', None, None),
    ('please halt', 'stop', None, None),
    ('water the plants', 'unsupported', None, None),
    ('bring me the remote', 'unsupported', None, None),
]
# A second set, written after the 'find', 'handles' and odometry changes and scored once before
# any further tuning (2026-10-10): 15/20, one wrong action ("spin to face backwards" drove
# backward; "face backwards" now reads as 180 degrees). Don't tune against it. Rerun for
# regressions only after the start/dock place changes: 16/20, no wrong actions. No longer a
# clean holdout: the travel/handling-word rules (19/20) came after seeing its misses.
TASKS_HOLDOUT_2 = [
    ('head to the north east corner', 'go_to', 'north_east_corner', None),
    ('please go over to the west room', 'go_to', 'west_room', None),
    ('return home', 'return_to_dock', None, None),
    ('go find the forklift', 'find', None, None),
    ('drive to the shelves', 'find', None, None),
    ('take me to where the barrels are', 'find', None, None),
    ('look for a person', 'find', None, None),
    ('move forward 5 meters', 'drive', None, 'forward'),
    ('go 3 meters to the right', 'drive', None, 'right'),
    ('back up 10 centimeters', 'drive', None, 'backward'),
    ('turn right 90 degrees', 'rotate', None, 'right'),
    ('spin to face backwards', 'rotate', None, None),
    ('wander around a little', 'explore', None, None),
    ('hang on for 15 seconds', 'wait', None, None),
    ('stop now', 'stop', None, None),
    ('pick that box up', 'unsupported', None, None),
    ('carry this to the kitchen', 'unsupported', None, None),
    ('open the window', 'unsupported', None, None),
    ('mop the floor', 'unsupported', None, None),
    ('tell me the time', 'unsupported', None, None),
]
# The TurtleBot 4 depot after a pallet search: no place catalog, so 'start' and 'dock' (as the
# robot node sets them up) plus the remembered sighting. Written from the depot runs, where
# "come back to where you started" asked for clarification; tuned against, so not a holdout.
TASKS_DEPOT = [
    ('come back to where you started', 'go_to', 'start', None),
    ('go back to your starting point', 'go_to', 'start', None),
    ('return to where you began', 'go_to', 'start', None),
    ('go back to the start', 'go_to', 'start', None),
    ('go back to the pallets', 'go_to', 'seen_pallet', None),
    ('return to the dock', 'return_to_dock', None, None),
    ('go home', 'return_to_dock', None, None),
    ('go to where the pallets are stored', 'find', None, None),
    ('find the forklift', 'find', None, None),
    ('drive to the pallet before you', 'find', None, None),      # was "find the forklift"
    ('go to the shelves behind you', 'find', None, None),        # was refused
    ('find the forklift on your left', 'find', None, 'left'),
    ('move the box', 'unsupported', None, None),                 # was "drive 0.5 m"
    ('drive to the pallet near the shelves with red barrels', 'find', None, None),
    ('go by the shelves', 'find', None, None),                   # was refused
    ('go 2 meters forward', 'drive', None, 'forward'),
    ('turn around', 'rotate', None, None),
]
DEPOT_PLACES = {'start': {'xy': (0.0, 0.0), 'description': START_DESCRIPTION},
                'dock': {'xy': (-0.6, 0.0), 'face': (0.2, 0.0), 'description': DOCK_DESCRIPTION},
                'seen_pallet': {'xy': (3.6, -2.9), 'description': 'where the pallet was seen'}}
# instruction given while busy -> expected timing
TIMING_CASES = [
    ('when you are done, turn around', 'after'),
    ('after that go to the south room', 'after'),
    ('once you finish, go to the dock', 'after'),
    ('afterwards, explore the maze', 'after'),
    ('go to the east room instead', 'now'),
    ('turn around', 'now'),
    ('go to the east hall right now', 'now'),
    ('forget that, go to the west room', 'now'),
]


def acceptable(direction, obstacle):
    """(preferred, also acceptable) actions for a situation, following the pilot's rules."""
    if obstacle in ('front near', 'front very close'):
        side = {'turn_left', 'turn_right'}
        if 'left' in direction:
            pref = {'turn_left'}
        elif 'right' in direction:
            pref = {'turn_right'}
        else:
            pref = side
        return pref, side | ({'back_up'} if obstacle == 'front very close' else set())
    if obstacle == 'front and left blocked':
        return {'turn_right'}, {'turn_right', 'back_up'}
    if obstacle == 'front and right blocked':
        return {'turn_left'}, {'turn_left', 'back_up'}
    deg = DIRECTIONS[direction]
    if abs(deg) > 60:
        turn = 'turn_left' if deg > 0 else 'turn_right'
        if deg == 180:
            return {'turn_left', 'turn_right'}, {'turn_left', 'turn_right'}
        veer = 'veer_left' if deg > 0 else 'veer_right'
        return {turn}, {turn, veer} if abs(deg) < 100 else {turn}
    if abs(deg) > 15:
        veer = 'veer_left' if deg > 0 else 'veer_right'
        blocked_side = (deg > 0 and obstacle == 'left side near') or \
                       (deg < 0 and obstacle == 'right side near')
        if blocked_side:
            return {'forward', 'creep'}, {'forward', 'creep'}
        return {veer}, {veer, 'forward', 'creep'}
    if obstacle == 'front far':
        return {'creep'}, {'creep', 'veer_left', 'veer_right'}
    if obstacle == 'left side near':
        return {'forward', 'creep'}, {'forward', 'creep', 'veer_right'}
    if obstacle == 'right side near':
        return {'forward', 'creep'}, {'forward', 'creep', 'veer_left'}
    return {'forward'}, {'forward', 'creep'}


def driving_cases():
    for direction, obstacle in itertools.product(DIRECTIONS, OBSTACLES):
        yield direction, obstacle, False
    for direction in ('straight ahead', 'to the left', 'ahead, slightly right'):
        yield direction, 'front very close', True


def driving_state(direction, obstacle, bumped):
    import math
    pilot = LayaPilot((math.cos(math.radians(DIRECTIONS[direction])) * 3.0,
                       math.sin(math.radians(DIRECTIONS[direction])) * 3.0))
    ranges = {s.name: create3.IR_MAX_RANGE for s in create3.IR_SENSORS}
    ranges.update(OBSTACLES[obstacle])
    ir = {name: create3.range_to_intensity(r) for name, r in ranges.items()}
    hazards = frozenset(['bump_front_center']) if bumped else frozenset()
    snap = Snapshot(0.0, 0.0, 0.0, 0.0, ir, hazards)
    query = pilot.prepare(snap)
    return query


def run_driving(client, min_confidence, show):
    rows = []
    for direction, obstacle, bumped in driving_cases():
        query = driving_state(direction, obstacle, bumped)
        pref, ok = ({'back_up'}, {'back_up'}) if bumped else acceptable(direction, obstacle)
        start = time.perf_counter()
        response = client.decide(query.state, query.questions)
        latency = time.perf_counter() - start
        answer = response['answers']['action']
        conf = top_confidence(answer['probabilities'])
        rows.append({'case': f"{direction} / {obstacle}{' / bumped' if bumped else ''}",
                     'choice': answer['choice'], 'preferred': answer['choice'] in pref,
                     'acceptable': answer['choice'] in ok, 'confidence': conf,
                     'latency': latency})
        if show:
            mark = 'OK ' if rows[-1]['acceptable'] else 'BAD'
            print(f"  {mark} {rows[-1]['case']:52s} -> {answer['choice']:10s} "
                  f"conf {conf:.2f}   want {sorted(pref)}")
    return report('driving', rows, min_confidence)


def run_heldout(client, min_confidence, path, show):
    """Exact agreement with the code driver on generated states the fine-tune never saw."""
    rows, detour_ok = [], []
    with open(path, encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    for row in data:
        start = time.perf_counter()
        answers = client.decide(row['state'], row['questions'])['answers']
        latency = time.perf_counter() - start
        action = answers['action']
        good = action['choice'] == row['expected']['action']
        rows.append({'case': row['expected']['action'], 'choice': action['choice'],
                     'preferred': good, 'acceptable': good, 'latency': latency,
                     'confidence': top_confidence(action['probabilities'])})
        detour_ok.append(answers['detour_side']['choice'] == row['expected']['detour_side'])
        if show and not good:
            print(f"  BAD want {row['expected']['action']:10s} got {action['choice']:10s} "
                  f"{json.dumps(row['state']['obstacle_sensors_left_to_right'])} "
                  f"{row['state']['next_waypoint']['direction']}")
    result = report('driving-heldout', rows, min_confidence)
    result['detour_accuracy'] = round(statistics.fmean(detour_ok), 3)
    return result


def run_tasks(client, min_confidence, show, cases=None, name='tasks', places=None):
    if places is None:
        # As the robot node sets them up: the maze catalog plus the starting point.
        places = load_places(ROOT / 'create3_laya' / 'config' / 'maze_places.json')
        places.setdefault('start', {'xy': (0.0, 0.0), 'description': START_DESCRIPTION})
    interpreter = TaskInterpreter(places, client)
    cases = cases or TASKS
    rows = []
    for text, skill, place, direction in cases:
        start = time.perf_counter()
        clause = interpreter.classify(text)
        latency = time.perf_counter() - start
        good = clause.skill == skill
        if place:
            good = good and clause.place == place
        if direction:
            good = good and clause.direction == direction
        rows.append({'case': text, 'choice': f'{clause.skill}/{clause.place}/{clause.direction}',
                     'preferred': good, 'acceptable': good,
                     'confidence': clause.skill_confidence, 'latency': latency,
                     'skill_ok': clause.skill == skill,
                     'place_ok': place is None or clause.place == place,
                     'acted_wrongly': not good and not interpreter._check(clause)})
        if show:
            print(f"  {'OK ' if good else 'BAD'} {text:42s} -> {rows[-1]['choice']:40s} "
                  f"skill conf {clause.skill_confidence:.2f}   want {skill}/{place}/{direction}")
    result = report(name, rows, min_confidence)
    result['wrong_actions'] = sum(r['acted_wrongly'] for r in rows)
    result['skill_accuracy'] = round(statistics.fmean(r['skill_ok'] for r in rows), 3)
    result['place_accuracy_when_named'] = round(statistics.fmean(
        r['place_ok'] for r, t in zip(rows, cases) if t[2]), 3)
    if cases is not TASKS:
        return result
    timing_ok = []
    for text, expected in TIMING_CASES:
        plan = interpreter.interpret(text, busy=True)
        timing_ok.append(plan.timing == expected)
        if show:
            print(f"  {'OK ' if timing_ok[-1] else 'BAD'} busy: {text:38s} -> {plan.timing:5s} "
                  f"conf {plan.timing_confidence:.2f}   want {expected}")
    result['timing_accuracy'] = round(statistics.fmean(timing_ok), 3)
    return result


def report(name, rows, min_confidence):
    confident = [r for r in rows if r['confidence'] >= min_confidence]
    latencies = sorted(r['latency'] for r in rows)
    return {
        'suite': name,
        'cases': len(rows),
        'preferred_accuracy': round(statistics.fmean(r['preferred'] for r in rows), 3),
        'acceptable_accuracy': round(statistics.fmean(r['acceptable'] for r in rows), 3),
        'mean_confidence': round(statistics.fmean(r['confidence'] for r in rows), 3),
        f'share_above_{min_confidence}': round(len(confident) / len(rows), 3),
        f'acceptable_accuracy_above_{min_confidence}': (
            round(statistics.fmean(r['acceptable'] for r in confident), 3) if confident else None),
        'latency_p50_ms': round(latencies[len(latencies) // 2] * 1000, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mock', action='store_true')
    p.add_argument('--url')
    p.add_argument('--model', help="Laya checkpoint: english, typed-decisions, multilingual")
    p.add_argument('--suite', choices=['driving', 'heldout', 'tasks', 'tasks-holdout',
                                       'tasks-holdout2', 'tasks-depot', 'all'],
                   default='all')
    p.add_argument('--heldout', default=str(ROOT / 'data' / 'driving_eval.jsonl'),
                   help='generated states for the held-out driving suite')
    p.add_argument('--min-confidence', type=float, default=0.3)
    p.add_argument('-v', '--verbose', action='store_true', help='print every case')
    args = p.parse_args()
    client = MockLaya(seed=0) if args.mock else LayaClient(args.url, args.model, timeout=30)
    results = []
    if args.suite in ('driving', 'all'):
        results.append(run_driving(client, args.min_confidence, args.verbose))
    if args.suite in ('heldout', 'all') and Path(args.heldout).exists():
        results.append(run_heldout(client, args.min_confidence, args.heldout, args.verbose))
    if args.suite in ('tasks', 'all'):
        results.append(run_tasks(client, args.min_confidence, args.verbose))
    if args.suite in ('tasks-holdout', 'all'):
        results.append(run_tasks(client, args.min_confidence, args.verbose, TASKS_HOLDOUT,
                                 'tasks-holdout'))
    if args.suite in ('tasks-holdout2', 'all'):
        results.append(run_tasks(client, args.min_confidence, args.verbose, TASKS_HOLDOUT_2,
                                 'tasks-holdout2'))
    if args.suite in ('tasks-depot', 'all'):
        results.append(run_tasks(client, args.min_confidence, args.verbose, TASKS_DEPOT,
                                 'tasks-depot', places=dict(DEPOT_PLACES)))
    print(json.dumps({'model': 'mock' if args.mock else (args.model or 'auto'),
                      'results': results}, indent=2))


if __name__ == '__main__':
    main()
