"""Offline stand-in for the Laya server. This is NOT a model.

A hand-written heuristic that reads the same `state` the pilot and task interpreter send to
Laya and answers their questions with probability distributions in the shape of Laya's
response, so the whole pipeline (state building, confidence routing, fallbacks, missions,
logging) can be exercised without the model server. Unknown questions get a uniform answer.
"""

import json
import math
import random
import re

from .laya_client import top_confidence

LEVEL = {'clear': 0, 'far': 1, 'near': 2, 'very close': 3, 'unknown': 2}
WAYPOINT_PREFERENCE = {
    'straight ahead': 'forward',
    'ahead, slightly left': 'veer_left',
    'ahead, slightly right': 'veer_right',
    'to the left': 'turn_left',
    'to the right': 'turn_right',
    'behind, on the left': 'turn_left',
    'behind, on the right': 'turn_right',
}
SKILL_WORDS = {
    'return_to_dock': r'\b(dock|charg\w*|home|back to (the )?start|return)\b',
    'rotate': r'\b(turn|spin|rotate)\b(?!.*\b(go|drive) to\b)',
    'drive': r'\b(forward|backward|back up|reverse|ahead)\b',
    'explore': r'\b(explore|wander|look around|roam)\b',
    'wait': r'\b(wait|pause|hold on)\b',
    'stop': r'\b(stop|halt|cancel|freeze)\b',
    'unsupported': r'\b(pick|grab|carry|clean|vacuum|open|fly|sing|say|speak|bring)\b',
    'go_to': r'\b(go|drive|head|move|navigate|visit|get)\b.*\b(to|into)\b',
}


class MockLaya:
    model = 'mock (heuristic stand-in, not a model)'

    def __init__(self, seed=None, temperature=0.45, noise=0.25, unsure_rate=0.05):
        self.rng = random.Random(seed)
        self.temperature = temperature
        self.noise = noise
        self.unsure_rate = unsure_rate

    def decide(self, state, questions):
        answers = {}
        for key, question in questions.items():
            kind = question['type']
            if kind == 'noul':
                answers[key] = {'type': 'noul', 'noul': 0.5}
                continue
            criteria = question['criteria']
            options = (list(criteria) if kind == 'choice'
                       else [str(i) for i in range(len(criteria))])
            scorer = getattr(self, f'_{key}_utility', None) if kind == 'choice' else None
            utility = scorer(state, criteria) if scorer else {o: 0.0 for o in options}
            noisy = key in ('action', 'detour_side')
            answers[key] = self._answer(kind, options, utility, criteria, noisy)
        size = len(json.dumps([state, questions]))
        return {'model': self.model, 'answers': answers,
                'usage': {'input_tokens': size // 4, 'output_tokens': 8 * len(questions)}}

    def _answer(self, kind, options, utility, criteria, noisy=True):
        temperature = self.temperature
        if noisy and self.rng.random() < self.unsure_rate:
            temperature *= 8          # occasionally unsure, to exercise confidence routing
        scores = {o: utility.get(o, 0.0) / temperature + self.rng.gauss(0, self.noise)
                  for o in options}
        top = max(scores.values())
        exp = {o: math.exp(s - top) for o, s in scores.items()}
        total = sum(exp.values())
        probs = {o: round(e / total, 4) for o, e in exp.items()}
        if kind == 'choice':
            return {'type': 'choice', 'choice': max(probs, key=probs.get),
                    'probabilities': probs, 'confidence': round(top_confidence(probs), 4)}
        expected = sum(int(o) * p for o, p in probs.items())
        return {'type': 'score', 'score': round(expected, 4), 'probabilities': probs,
                'legend': {str(i): c for i, c in enumerate(criteria)},
                'confidence': round(top_confidence(probs), 4)}

    # -- driving questions -----------------------------------------------------------

    @staticmethod
    def _read(state):
        sensors = {k: LEVEL.get(v, 2) for k, v in state['obstacle_sensors_left_to_right'].items()}
        front = max(sensors['front left'], sensors['front'], sensors['front right'])
        left = max(sensors['far left'], sensors['left'], sensors['front left'])
        right = max(sensors['far right'], sensors['right'], sensors['front right'])
        plan = state['detour_plan']
        plan = ('left' if plan.endswith('its left') else
                'right' if plan.endswith('its right') else None)
        return front, left, right, plan

    def _action_utility(self, state, criteria):
        front, left, right, plan = self._read(state)
        direction = state['next_waypoint']['direction']
        u = {o: 0.0 for o in criteria}

        def add(name, value):
            if name in u:
                u[name] += value

        add('back_up', 3.0 if state['bumper'] != 'not pressed' else -1.5)
        preferred = WAYPOINT_PREFERENCE.get(direction) or f"turn_{plan or 'left'}"
        weight = 0.8 if plan else 2.0
        add(preferred, weight)
        if preferred == 'forward':
            add('creep', weight * 0.4)

        add('forward', -1.6 * front)
        add('creep', -0.3 - 1.2 * max(front - 1, 0))
        add('veer_left', -1.4 * max(front - 1, 0) - 1.4 * max(left - 1, 0))
        add('veer_right', -1.4 * max(front - 1, 0) - 1.4 * max(right - 1, 0))
        if front >= 2:
            open_side = plan or ('left' if left < right else 'right' if right < left else
                                 ('left' if 'left' in direction else 'right'))
            add(f'turn_{open_side}', 2.5)
        elif plan:
            wall = right if plan == 'left' else left
            add('forward', 1.2 if wall >= 1 else 0.0)
            add(f"veer_{'right' if plan == 'left' else 'left'}", 1.2 if wall == 0 else -0.5)
        recent = state['recent_actions_oldest_first']
        if 'turn_left' in recent and 'turn_right' in recent:
            add(f"turn_{plan or 'left'}", 0.8)       # break left/right dithering
        return u

    def _detour_side_utility(self, state, criteria):
        front, left, right, plan = self._read(state)
        direction = state['next_waypoint']['direction']
        u = {o: 0.0 for o in criteria}
        if front >= 1 and 'ahead' in direction:
            better = ('left' if left < right else 'right' if right < left else
                      'left' if 'left' in direction else 'right')
            u[better] += 2.0
            u['none'] -= 0.5
        elif plan and state['progress_last_5_seconds'] != 'getting closer':
            u[plan] += 1.5
        else:
            u['none'] += 1.5
        return u

    # -- task questions --------------------------------------------------------------

    def _skill_utility(self, state, criteria):
        text = state['instruction'].lower()
        u = {o: 0.0 for o in criteria}
        for skill, pattern in SKILL_WORDS.items():
            if skill in u and re.search(pattern, text):
                u[skill] += 2.0
                break
        return u

    def _place_utility(self, state, criteria):
        text = set(re.findall(r'[a-z]+', state['instruction'].lower()))
        u = {}
        for name, description in criteria.items():
            words = set(re.findall(r'[a-z]+', f'{name} {description or ""}'.lower()))
            u[name] = 1.2 * len(text & set(name.split('_'))) + 0.2 * len(text & words)
        if 'no_place' in u:
            u['no_place'] = 0.9
        return u

    def _direction_utility(self, state, criteria):
        text = state['instruction'].lower()
        u = {o: 0.0 for o in criteria}
        for option, pattern in (('left', r'\bleft\b'), ('right', r'\bright\b'),
                                ('backward', r'\b(back|backward|reverse)\b'),
                                ('forward', r'\b(forward|ahead)\b')):
            if option in u and re.search(pattern, text):
                u[option] += 2.0
                return u
        if 'unspecified' in u:
            u['unspecified'] += 1.0
        return u

    def _pace_utility(self, state, criteria):
        text = state['instruction'].lower()
        slow = re.search(r'\b(slow\w*|careful\w*|gentl\w*)\b', text)
        return {'slow': 2.0 if slow else 0.0, 'normal': 0.0 if slow else 1.5}

    def _timing_utility(self, state, criteria):
        text = state['instruction'].lower()
        after = re.search(r"\b(after|afterwards|later|when (you are|you're) (done|finished))\b",
                          text)
        return {'after': 2.0 if after else 0.0, 'now': 0.0 if after else 1.5}

    def _destination_utility(self, state, criteria):
        text = state['instruction'].lower()
        travel = re.search(SKILL_WORDS['go_to'], text) or re.search(
            SKILL_WORDS['return_to_dock'], text)
        return {'yes': 2.0 if travel else 0.0, 'no': 0.0 if travel else 1.5}

    def _motion_utility(self, state, criteria):
        text = state['instruction'].lower()
        turn = re.search(SKILL_WORDS['rotate'], text) and not re.search(
            SKILL_WORDS['drive'], text)
        return {'turn': 1.5 if turn else 0.0, 'move': 0.0 if turn else 1.0}

    def _object_utility(self, state, criteria):
        text = state['instruction'].lower()
        u = {o: 0.0 for o in criteria}
        for name in criteria:
            if name != 'other' and re.search(rf'\b{name}', text):
                u[name] += 2.0
                return u
        u['other'] = 1.0
        return u

    def _handles_utility(self, state, criteria):
        handle = re.search(SKILL_WORDS['unsupported'], state['instruction'].lower())
        return {'yes': 2.0 if handle else 0.0, 'no': 0.0 if handle else 1.5}
