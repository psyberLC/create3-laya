#!/usr/bin/env python3
"""Generate laya-train data that teaches a Laya checkpoint this robot's driving decisions.

Each row is a labelled state in exactly the format the pilot sends at run time (built by
`make_state`), with both driving questions and the answers the rule policy gives:

  {"state": {...}, "questions": {"action": ..., "detour_side": ...},
   "expected": {"action": "turn_left", "detour_side": "left"}}

  python tools/make_driving_data.py --rows 3000 --out data/driving_train.jsonl --seed 1
  python tools/make_driving_data.py --rows 400 --out data/driving_eval.jsonl --seed 2
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'create3_laya'))

from create3_laya import create3  # noqa: E402
from create3_laya.pilot import (ACTIONS, DIRECTION_LABELS, DISTANCE_LABELS,  # noqa: E402
                                PROGRESS_LABELS, SENSOR_LABELS, LayaPilot, make_state,
                                rule_action, rule_detour)

LEVELS = ['clear', 'far', 'near', 'very close']


def sample_sensors(rng):
    """Obstacles come in spatially coherent patches, the way walls and boxes look to the IR."""
    levels = [0] * len(SENSOR_LABELS)
    kind = rng.random()
    if kind < 0.3:
        pass                                              # open floor
    elif kind < 0.75:                                     # one obstacle patch
        center = rng.randrange(len(levels))
        peak = rng.choice([1, 2, 2, 3, 3])
        width = rng.choice([0, 1, 1, 2])
        for i in range(len(levels)):
            d = abs(i - center)
            if d <= width:
                levels[i] = max(levels[i], max(1, peak - d))
    elif kind < 0.9:                                      # wall along one side
        side = rng.choice([range(0, 3), range(4, 7)])
        level = rng.choice([1, 1, 2, 3])
        for i in side:
            levels[i] = max(levels[i], level - (1 if i in (2, 4) and rng.random() < 0.5 else 0))
    else:                                                 # clutter
        levels = [rng.choice([0, 0, 1, 2, 3]) for _ in levels]
    labels = [LEVELS[max(0, min(3, v))] for v in levels]
    if rng.random() < 0.03:
        labels[rng.randrange(len(labels))] = 'unknown'
    return dict(zip(SENSOR_LABELS, labels))


def sample_state(rng):
    bumps = ()
    if rng.random() < 0.07:
        zones = list(create3.BUMP_LABELS.values())
        bumps = tuple(sorted(rng.sample(zones, rng.choice([1, 1, 2]))))
    return make_state(
        direction=rng.choices(DIRECTION_LABELS, weights=[30, 15, 15, 10, 10, 7, 7, 6])[0],
        distance=rng.choices(DISTANCE_LABELS, weights=[1, 3, 3, 2])[0],
        sensors=sample_sensors(rng),
        bumps=bumps,
        progress=rng.choice(PROGRESS_LABELS),
        detour=rng.choices(['none', 'left', 'right'], weights=[6, 2, 2])[0],
        recent=[rng.choice(list(ACTIONS)) for _ in range(rng.randrange(5))],
        cliff=rng.random() < 0.02,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--rows', type=int, default=3000)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()

    rng = random.Random(args.seed)
    questions = LayaPilot(seed=args.seed).build_questions()
    # Canonical option order; laya-train --shuffle-options augments it, as run time does.
    questions['action']['criteria'] = {a: ACTIONS[a][2] for a in ACTIONS}
    questions['detour_side']['criteria'] = dict(sorted(
        questions['detour_side']['criteria'].items(), key=lambda kv: ['none', 'left', 'right']
        .index(kv[0])))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    actions, detours = Counter(), Counter()
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        for _ in range(args.rows):
            state = sample_state(rng)
            expected = {'action': rule_action(state), 'detour_side': rule_detour(state)}
            actions[expected['action']] += 1
            detours[expected['detour_side']] += 1
            f.write(json.dumps({'state': state, 'questions': questions,
                                'expected': expected}) + '\n')
    print(f'wrote {args.rows} rows to {out}')
    print('action labels:', dict(actions.most_common()))
    print('detour labels:', dict(detours.most_common()))


if __name__ == '__main__':
    main()
