"""Natural-language tasks -> robot steps, with Laya doing the language understanding.

Laya does not generate text, so it never writes a plan. Instead:
  * code splits the instruction into clauses ("go to the east hall, then come back"),
  * code pulls out numbers with units ("90 degrees", "1.5 m", "10 seconds"),
  * Laya classifies each clause in one request: which skill, which named place, which
    direction, what pace,
  * answers below the confidence bar become a clarifying question instead of an action,
    and requests outside the skill set are refused.
"""

import json
import math
import random
import re
from dataclasses import dataclass, field

from .laya_client import top_confidence
from .mission import Step
from .vision import OBJECTS

SKILLS = {
    'go_to': 'Drive to a named place or room.',
    'return_to_dock': 'Go back home to the charging dock and dock.',
    'rotate': 'Turn or spin in place by some angle.',
    'drive': 'Move a distance forward, backward, left or right, like "go 10 meters forward".',
    'find': 'Go to an object you can see, like pallets, shelves, or a forklift.',
    'explore': 'Wander and explore with no particular destination.',
    'wait': 'Pause or wait for a while.',
    'stop': 'Stop, halt, or cancel.',
    'unsupported': 'Something a floor robot cannot do, like picking up objects, cleaning, '
                   'or talking.',
}
SKILL_SUMMARY = ('go to a named place or to an object I can see, return to the dock, rotate, '
                 'drive a distance, explore, wait, or stop')
# "go to where the pallets are stored" -> "pallets"; used when Laya's object answer is 'other'.
OBJECT_PHRASE = re.compile(
    r'\b(?:to|find|near|towards?|by|reach)\s+(?:where\s+)?(?:the\s+|a\s+|an\s+|some\s+)?'
    r'([a-z][a-z -]{1,40}?)'
    r'(?:\s+(?:is|are|was|were|stand|stands|sit|sits|lie|lies|stored|kept)\b|[,.!?]|$)',
    re.IGNORECASE)
# The objects in the text, read by code: a literal name settles Laya's object answer, which
# took "drive to the pallet before you" for a forklift.
OBJECT_NAMES = {
    'pallet': r'pallets?', 'shelf': r'shel(?:f|ves)|(?:storage )?racks?',
    'box': r'box(?:es)?|crates?|cartons?', 'forklift': r'fork-?lifts?',
    'stairs': r'stairs?|staircases?', 'door': r'doors?|doorways?',
    'pillar': r'pillars?|columns?', 'barrel': r'barrels?|drums?',
    'person': r'person|people|persons',
}
OBJECT_WORDS = {label: re.compile(rf'\b(?:{words})\b', re.IGNORECASE)
                for label, words in OBJECT_NAMES.items()}
NEVER = re.compile(r'(?!)')
# Words before an object's name that say which one, not what it looks like: the choice of
# sighting (nearest, in the asked-for direction) covers them; the detector shouldn't see them.
NOT_VISUAL = {'nearest', 'closest', 'next', 'first', 'second', 'last', 'other', 'same',
              'this', 'that', 'these', 'those', 'near', 'nearby', 'far', 'farthest', 'any'}
# Which side a search must look, read from the text: Laya's direction answer (meant for
# turns and moves) said "right" for "drive to the pallet near the shelves with red barrels".
# "Behind the shelves" and "in front of the shelves" name a reference (NEAR), not a side,
# and "right in front of you" is ahead.
SIDES = ((re.compile(r'\bbehind\b(?!\s+(?:the|a|an|some)\b)', re.IGNORECASE), math.pi),
         (re.compile(r'\bleft\b', re.IGNORECASE), math.pi / 2),
         (re.compile(r'\bright\b(?!\s+(?:in front|ahead|before|behind|here|there)\b)',
                     re.IGNORECASE), -math.pi / 2),
         (re.compile(r'\b(?:ahead|before you|in front(?!\s+of\s+(?!you\b|me\b)))',
                     re.IGNORECASE), 0.0))
# "the pallet near the shelves with red barrels": the reference after the relation word.
NEAR = re.compile(r'\b(?:near|next to|beside|by|close to|alongside|behind|in front of)\s+'
                  r'(?:the\s+|a\s+|an\s+|some\s+)?(?!(?:you|me|yourself)\b)'
                  r'([a-z][a-z -]*?)\s*(?:[,.!?]|$)', re.IGNORECASE)
# Travelling to something, as opposed to doing something to it ("move to the box", not
# "move the box"). Laya's handling answer can't separate them: driving to shelves or a door
# scored 0.51-0.57, real handling requests 0.61-0.86, "move the box" 0.47.
GO_TOWARD = re.compile(r'\b(?:(?:go|drive|head|move|navigate|come|return|get)'
                       r'(?:\s+(?:back|over|up))?\s+(?:to|towards?|near|by|next to|close to)'
                       r'|visit|approach)\b', re.IGNORECASE)
# Only verbs that never describe a place ("open area", "go close to" stay travel); Laya's
# handling answer covers the rest ("open the door": 0.61).
HANDLING = re.compile(
    r'\b(?:pick|grab|bring|fetch|carry|lift|push|pull|wipe|mop|deliver|put|throw|stack|unload)'
    r'\b|\bmove\s+(?:the|a|an|this|that|these|those)\s+(?:\w+\s+)?(?:'
    + '|'.join(OBJECT_NAMES.values()) + r')\b',
    re.IGNORECASE)
DIRECTIONS = {
    'left': 'Turn or move to the left.',
    'right': 'Turn or move to the right.',
    'forward': 'Move forward or ahead.',
    'backward': 'Move backward or reverse.',
    'unspecified': 'No direction is given.',
}
PACES = {'normal': 'Normal speed.', 'slow': 'Slowly, carefully, or gently.'}
DESTINATION = {
    'yes': 'Yes: it names a room, corner, hall, the dock, or the starting point to drive to.',
    'no': 'No: it asks to turn, spin, back up, move a short distance, wait, stop, or something '
          'else.',
}
HANDLES = {
    'yes': 'Yes: it asks to pick up, carry, bring, clean, water, open, or otherwise handle '
           'something.',
    'no': 'No: it only asks the robot to move, turn, look for, wait, or stop.',
}
MOTION = {
    'turn': 'Turn or face a new direction while staying in the same spot.',
    'move': 'Move along the floor, forward or backward.',
}
TIMING = {
    'now': 'Do it right away, replacing what the robot is doing.',
    'after': 'Do it after the current task is finished, like "afterwards" or "when done".',
}
NO_PLACE = 'no_place'
# Places every world gets. The robot starts on its dock, but the two descriptions must not
# share words like "started": "come back to where you started" split Laya between them.
START_DESCRIPTION = 'the starting point: the spot where the robot was when it started'
DOCK_DESCRIPTION = "the charging dock, the robot's home base"
# With "no place", 10 options: Laya ships no valid calibration for Choice questions with more.
MAX_PLACE_OPTIONS = 9

ACTION_VERBS = (r'(?:go|drive|head|move|navigate|visit|return|come|turn|spin|rotate|back|'
                r'reverse|explore|wander|wait|pause|stop|halt|dock|charge)')
CLAUSE_SPLIT = re.compile(     # sentence ends (not decimal points), "then"-style joiners, and
    r'\s*(?:[;!?]+|\.(?!\d)'   # ", <verb>" / "and <verb>" where a new command starts
    r'|,?\s*\b(?:and then|then|after that|afterwards|and finally|finally)\b,?'
    r'|,\s*(?:and\s+)?(?=' + ACTION_VERBS + r'\b)|\s+and\s+(?=' + ACTION_VERBS + r'\b))\s*',
    re.IGNORECASE)
WORD_NUMBERS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'fifteen': 15, 'twenty': 20,
                'thirty': 30, 'ninety': 90, 'half a': 0.5, 'half an': 0.5}
NUMBER = r'\b(\d+(?:\.\d+)?|half an?|an?|' + '|'.join(
    w for w in WORD_NUMBERS if ' ' not in w and len(w) > 2) + r')'
ANGLE = re.compile(NUMBER + r'\s*(?:°|deg(?:ree)?s?\b)', re.IGNORECASE)
DISTANCE = re.compile(NUMBER + r'\s*(cm|centimet(?:er|re)s?|m|met(?:er|re)s?|ft|feet|foot)\b',
                      re.IGNORECASE)
DURATION = re.compile(NUMBER + r'\s*(s|secs?|seconds?|mins?|minutes?)\b', re.IGNORECASE)
COUNT = re.compile(NUMBER + r'\s+(?:places|rooms|spots|stops|locations)\b', re.IGNORECASE)
TURN_WORDS = ((re.compile(r'\b(full|whole|complete)\s+(turn|circle|spin|rotation)\b|\b360\b',
                          re.IGNORECASE), 360.0),
              (re.compile(r'\bturn(ing)?\s+(around|back)\b|\babout[- ]face\b'
                          r'|\bhalf\s+(turn|circle)\b'
                          r'|\bface\s+(backwards?|the\s+other\s+way|the\s+opposite\s+way|back)\b',
                          re.IGNORECASE), 180.0),
              (re.compile(r'\bquarter\s+turn\b', re.IGNORECASE), 90.0))
UNIT_TO_M = {'cm': 0.01, 'centimeter': 0.01, 'centimetre': 0.01, 'ft': 0.3048, 'feet': 0.3048,
             'foot': 0.3048}

DEFAULT_ANGLE = 90.0
DEFAULT_DISTANCE = 0.5
DEFAULT_WAIT = 3.0
DEFAULT_EXPLORE_STOPS = 3
MAX_DISTANCE = 20.0
BACKUP_LIMIT = 0.15         # the Create 3 refuses to reverse further (no cliff sensors behind)
EXPLORE_HOP = 3.0           # metres per wander leg when there are no named places


@dataclass
class Clause:
    text: str
    skill: str = None
    skill_confidence: float = 0.0
    place: str = None
    place_confidence: float = 0.0
    place_probability: float = 0.0     # Laya's p for its place answer, not rescaled by option count
    direction: str = 'unspecified'
    pace: str = 'normal'
    pace_confidence: float = 0.0
    target_object: str = None      # Laya's object answer (a key of vision.OBJECTS)
    object_confidence: float = 0.0
    object_phrase: str = None      # code's reading of the object words, for 'other'
    near: str = None               # a reference for find: "near the shelves with red barrels"
    handles_probability: float = 0.0   # Laya's p('yes'): asks to handle an object?
    destination: str = None        # Laya's literal yes/no: does it name a place to drive to?
    motion: str = None             # Laya's literal turn/move answer
    motion_confidence: float = 0.0
    numbers: dict = field(default_factory=dict)
    answers: dict = field(default_factory=dict)
    latency: float = None
    resolved: str = None           # which code rule adjusted Laya's skill answer, if any


@dataclass
class TaskPlan:
    text: str
    clauses: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    clarification: str = None      # set when Laya was unsure; nothing should run
    refusal: str = None            # set when a clause asks for something unsupported
    timing: str = 'now'            # for a task given while busy: 'now' (replace) or 'after'
    timing_confidence: float = None

    @property
    def stops(self):
        return bool(self.steps) and self.steps[0].kind == 'stop'

    @property
    def runnable(self):
        return bool(self.steps) and not self.clarification and not self.refusal

    def summary(self):
        if self.refusal:
            return self.refusal
        if self.clarification:
            return self.clarification
        return '; '.join(step.describe() for step in self.steps)


def load_places(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    places = {}
    for name, p in data['places'].items():
        places[name] = {'xy': tuple(p['xy']), 'description': p['description']}
        if 'face' in p:             # point to turn toward on arrival (the dock's beacons)
            places[name]['face'] = tuple(p['face'])
    return places


HAS_ACTION_VERB = re.compile(r'\b' + ACTION_VERBS + r'\b', re.IGNORECASE)


def split_clauses(text):
    """Split into commands; lead-ins without a verb ("actually,", "when you're done,") stay
    attached to the command they introduce."""
    parts = [c.strip(' ,') for c in CLAUSE_SPLIT.split(text) if c and c.strip(' ,')]
    clauses, carry = [], ''
    for part in parts:
        if not HAS_ACTION_VERB.search(part):
            carry = f'{carry}, {part}' if carry else part
            continue
        clauses.append(f'{carry}, {part}' if carry else part)
        carry = ''
    if carry:
        if clauses:
            clauses[-1] = f'{clauses[-1]}, {carry}'
        else:
            clauses.append(carry)
    return clauses


def _named_objects(text):
    """The object labels the text names, in the order it names them."""
    found = sorted((m.start(), label) for label, pattern in OBJECT_WORDS.items()
                   if (m := pattern.search(text)))
    return [label for _, label in found]


def _number(token):
    token = ' '.join(token.lower().split())
    return float(WORD_NUMBERS[token]) if token in WORD_NUMBERS else float(token)


def extract_numbers(text):
    """Numbers with units, in robot units: degrees, metres, seconds, stop count."""
    found = {}
    if m := ANGLE.search(text):
        found['degrees'] = _number(m.group(1))
    else:
        for pattern, degrees in TURN_WORDS:
            if pattern.search(text):
                found['degrees'] = degrees
                break
    if m := DISTANCE.search(text):
        unit = m.group(2).lower().rstrip('s')
        found['metres'] = _number(m.group(1)) * UNIT_TO_M.get(unit, 1.0)
    if m := DURATION.search(text):
        scale = 60.0 if m.group(2).lower().startswith('m') else 1.0
        found['seconds'] = _number(m.group(1)) * scale
    if m := COUNT.search(text):
        found['count'] = int(_number(m.group(1)))
    return found


class TaskInterpreter:
    def __init__(self, places, client, *, min_skill=0.3, min_place=0.25, min_timing=0.2,
                 min_place_alone=0.5, seed=None):
        self.places = places
        self.client = client
        self.min_skill = min_skill
        self.min_place = min_place
        # A place Laya rates likelier than every alternative together (p >= 0.5) settles an
        # unsure skill answer ("go back to your starting point": skill stop at 0.18, place
        # start at p 0.82). A raw probability, unlike top_confidence, doesn't drop as
        # remembered sightings add place options.
        self.min_place_alone = min_place_alone
        # Per-question bar: on Laya's typed-decisions checkpoint, "after" cues ("when you are
        # done", "once you finish") score p >= 0.62 while cue-less commands stay below 0.6.
        self.min_timing = min_timing
        self.rng = random.Random(seed)

    def offered_places(self):
        """Place name -> description for Laya, at most MAX_PLACE_OPTIONS of them."""
        names = list(self.places)
        options = {}
        if len(names) > MAX_PLACE_OPTIONS and 'start' in names and 'dock' in names:
            # The robot starts on its dock, so the dock stands in for the starting point.
            names.remove('start')
            options['dock'] = self.places['dock']['description'] + ', where the robot started'
        seen = [n for n in names if n.startswith('seen_')]
        while len(names) > MAX_PLACE_OPTIONS and seen:
            names.remove(seen.pop(0))       # oldest sighting first
        return {n: options.get(n, self.places[n]['description'])
                for n in names[:MAX_PLACE_OPTIONS]}

    def questions(self):
        place_options = self.offered_places()
        place_options[NO_PLACE] = 'The instruction names no place.'
        return {
            'object': {'type': 'choice',
                       'instructions': 'Which kind of object does the instruction ask the robot '
                                       'to go to or find?',
                       'criteria': {k: v[0] for k, v in OBJECTS.items()}},
            'skill': {'type': 'choice',
                      'instructions': 'Which single action does this instruction ask the '
                                      'robot to do?',
                      'criteria': SKILLS},
            'place': {'type': 'choice',
                      'instructions': 'Which place does the instruction name as the '
                                      'destination?',
                      'criteria': place_options},
            'direction': {'type': 'choice',
                          'instructions': 'Which direction does the instruction say to turn '
                                          'or move?',
                          'criteria': DIRECTIONS},
            'pace': {'type': 'choice',
                     'instructions': 'How fast does the instruction ask the robot to move?',
                     'criteria': PACES},
            # A plainer, literal question: the skill answer alone lets direction words inside
            # place names ("top right corner") pull it toward 'rotate'.
            'destination': {'type': 'choice',
                            'instructions': 'Does the instruction ask the robot to travel to '
                                            'a destination place?',
                            'criteria': DESTINATION},
            'motion': {'type': 'choice',
                       'instructions': 'Does the instruction ask the robot to turn in place, or '
                                       'to move along the floor?',
                       'criteria': MOTION},
            # Object words pull the skill answer toward 'find' ("pick up the red cup").
            'handles': {'type': 'choice',
                        'instructions': 'Does the instruction ask the robot to handle or change '
                                        'an object, such as picking it up, carrying, bringing, '
                                        'cleaning, watering, or opening it?',
                        'criteria': HANDLES},
        }

    def classify(self, text, clock=None):
        clause = Clause(text, numbers=extract_numbers(text))
        start = clock() if clock else None
        response = self.client.decide({'instruction': text}, self.questions())
        clause.latency = clock() - start if clock else None
        answers = response.get('answers', {})
        clause.answers = answers
        skill = answers.get('skill', {})
        clause.skill = skill.get('choice')
        clause.skill_confidence = top_confidence(skill.get('probabilities') or {'?': 0.0})
        place = answers.get('place', {})
        clause.place = place.get('choice')
        clause.place_confidence = top_confidence(place.get('probabilities') or {'?': 0.0})
        clause.place_probability = max((place.get('probabilities') or {'?': 0.0}).values())
        clause.direction = answers.get('direction', {}).get('choice') or 'unspecified'
        pace = answers.get('pace', {})
        clause.pace = pace.get('choice') or 'normal'
        clause.pace_confidence = top_confidence(pace.get('probabilities') or {'?': 0.0})
        target = answers.get('object', {})
        clause.target_object = target.get('choice')
        clause.object_confidence = top_confidence(target.get('probabilities') or {'?': 0.0})
        # The object asked for comes before a relation ("the pallet near the shelves"); what
        # follows it is the reference, whichever of the two Laya's object answer picked.
        # With nothing named before it ("go behind the shelves"), the object after it is the
        # one to go to.
        relation = NEAR.search(text)
        named = _named_objects(text[:relation.start()]) if relation else []
        if named:
            clause.near = relation.group(1).strip()
            clause.target_object = named[0]
        else:
            named = _named_objects(text)
        if named:
            if clause.target_object not in named:
                clause.target_object = named[0]              # the first one the text names
            clause.object_confidence = 1.0
        if m := OBJECT_PHRASE.search(text):
            clause.object_phrase = m.group(1).strip()
        clause.handles_probability = (answers.get('handles', {}).get('probabilities') or
                                      {}).get('yes', 0.0)
        clause.destination = answers.get('destination', {}).get('choice')
        motion = answers.get('motion', {})
        clause.motion = motion.get('choice')
        clause.motion_confidence = top_confidence(motion.get('probabilities') or {'?': 0.0})
        self._resolve(clause)
        return clause

    def _resolve(self, clause):
        """Combine Laya's separate answers in code where one literal answer settles another."""
        named_place = (clause.place in self.places
                       and clause.place_confidence >= self.min_place)
        named_object = (clause.target_object in OBJECTS and clause.target_object != 'other'
                        and clause.object_confidence >= self.min_place)
        travel = GO_TOWARD.search(clause.text) and not HANDLING.search(clause.text)
        if HANDLING.search(clause.text) or (clause.handles_probability > 0.5 and not travel):
            # Errs toward refusing: on Laya's typed-decisions checkpoint, handling requests
            # scored 0.54-0.86 and plain moves at most 0.43; a handling word in the text
            # settles it, and so does a plain "go to ..." the other way.
            clause.skill = 'unsupported'
            clause.resolved = 'asks to handle an object, which this robot cannot do'
        elif (clause.skill == 'unsupported' and named_object and travel
              and OBJECT_WORDS.get(clause.target_object, NEVER).search(clause.text)):
            # "go to the shelves behind you" came back unsupported (0.59): it names an object
            # to drive to, and nothing to do to it.
            clause.skill = 'find'
            clause.skill_confidence = max(clause.skill_confidence, self.min_skill)
            clause.resolved = 'names an object to go to, and nothing to handle'
        elif clause.skill == 'find' and named_place and not named_object:
            clause.skill, clause.skill_confidence = 'go_to', clause.place_confidence
            clause.resolved = 'names a known place, so go there'
        elif clause.skill in ('rotate', 'drive') and clause.destination == 'yes' and named_place:
            clause.skill, clause.skill_confidence = 'go_to', clause.place_confidence
            clause.resolved = 'names a destination place, so go there'
        elif clause.skill == 'go_to' and not named_place and named_object:
            clause.skill = 'find'
            clause.resolved = 'no known place, but an object to look for'
        elif (clause.skill_confidence < self.min_skill and clause.skill != 'unsupported'
              and clause.place in self.places
              and clause.place_probability >= self.min_place_alone):
            clause.skill = 'return_to_dock' if clause.place == 'dock' else 'go_to'
            clause.skill_confidence = max(clause.place_confidence, self.min_skill)
            clause.resolved = 'unsure of the action, but sure of the place, so go there'
        elif (clause.skill in ('drive', 'rotate') and clause.skill_confidence < self.min_skill
              and ('metres' in clause.numbers) != ('degrees' in clause.numbers)):
            # "back up 10 centimeters": skill drive at 0.2, rotate close behind. A distance
            # (or an angle, never both) in the text says which.
            clause.skill = 'drive' if 'metres' in clause.numbers else 'rotate'
            clause.skill_confidence = self.min_skill
            clause.resolved = 'a distance or an angle in the text says drive or turn'
        # An angle ("turn around" = 180) or a distance in the text keeps the original skill.
        # A weak turn/move answer (p < 0.6) defers to a forward/backward direction.
        elif (clause.skill == 'rotate' and 'degrees' not in clause.numbers
              and (clause.motion == 'move'
                   or (clause.motion_confidence < 0.2
                       and clause.direction in ('forward', 'backward')))):
            clause.skill = 'drive'
            clause.resolved = 'moves along the floor, so drive'
        elif (clause.skill == 'drive' and clause.motion == 'turn'
              and 'metres' not in clause.numbers):
            clause.skill = 'rotate'
            clause.resolved = 'turns in place, so rotate'
        elif clause.skill == 'stop' and 'seconds' in clause.numbers:
            clause.skill = 'wait'
            clause.resolved = 'a stop with a duration is a wait'

    def interpret(self, text, clock=None, busy=False):
        """Plan for `text`. With `busy`, also decide whether it replaces or follows the
        current task (a stop request always applies immediately)."""
        plan = TaskPlan(text)
        if busy:
            self._timing(plan)
        for part in split_clauses(text):
            clause = self.classify(part, clock)
            plan.clauses.append(clause)
            problem = self._check(clause)
            if problem:
                kind, message = problem
                setattr(plan, kind, message)
                plan.steps = []
                return plan
            # "go back to the charging station and dock" splits into two dock requests.
            if (clause.skill == 'return_to_dock' and len(plan.clauses) > 1
                    and plan.clauses[-2].skill == 'return_to_dock'):
                continue
            plan.steps.extend(self._steps(clause))
        return plan

    def _timing(self, plan):
        question = {'timing': {'type': 'choice',
                               'instructions': 'The robot is busy with another task. When '
                                               'should it do this new instruction?',
                               'criteria': TIMING}}
        answer = self.client.decide({'instruction': plan.text}, question)['answers']['timing']
        probs = answer.get('probabilities') or {}
        plan.timing_confidence = top_confidence(probs) if probs else 0.0
        # Replacing the current task is the safer reading when Laya is unsure.
        if answer.get('choice') == 'after' and plan.timing_confidence >= self.min_timing:
            plan.timing = 'after'

    def _check(self, clause):
        """None if the clause can run, else ('refusal' | 'clarification', message)."""
        # Refusing is safe, so an 'unsupported' top answer needs no confidence bar when the
        # clause uses none of the robot's own verbs ("pick up the cup").
        if clause.skill == 'unsupported' and (clause.skill_confidence >= self.min_skill
                                              or clause.handles_probability > 0.5
                                              or not HAS_ACTION_VERB.search(clause.text)):
            return 'refusal', f"I can't do \"{clause.text}\". I can {SKILL_SUMMARY}."
        if clause.skill not in SKILLS or clause.skill_confidence < self.min_skill:
            top = _top_two(clause.answers.get('skill', {}).get('probabilities', {}))
            options = ' or '.join(f'"{SKILLS[s].rstrip(".").lower()}"' for s in top)
            return ('clarification',
                    f'I\'m not sure what "{clause.text}" asks for. Did you mean {options}?')
        if clause.skill == 'go_to' and (clause.place not in self.places
                                        or clause.place_confidence < self.min_place):
            names = ', '.join(n.replace('_', ' ') for n in self.places)
            return ('clarification',
                    f'Where should I go for "{clause.text}"? Places I know: {names}.')
        if clause.skill == 'find' and not self._find_queries(clause):
            return ('clarification',
                    f'What should I look for in "{clause.text}"? For example: '
                    + ', '.join(k for k in OBJECTS if k != 'other') + '.')
        return None

    def _find_queries(self, clause):
        """(label, detector queries) for a find clause, or None if nothing to look for."""
        label = clause.target_object
        if label in OBJECTS and label != 'other' and clause.object_confidence >= self.min_place:
            # Words before the object's name narrow the search: "the red barrel" asks the
            # detector for "a red barrel". Words after it ("before you") are not the object.
            phrase = clause.object_phrase or ''
            m = OBJECT_WORDS[label].search(phrase) if label in OBJECT_WORDS else None
            words = [w for w in phrase[:m.start()].split() if w.lower() not in NOT_VISUAL] \
                if m else []
            if words:
                qualified = ' '.join(words + [m.group(0)])
                return qualified, [f'a {qualified}']
            return label, OBJECTS[label][1]
        if clause.object_phrase:
            return clause.object_phrase, [f'a {clause.object_phrase}']
        return None

    def _steps(self, clause):
        slow = clause.pace == 'slow' and clause.pace_confidence >= self.min_skill
        pace = 0.6 if slow else 1.0
        n = clause.numbers
        if clause.skill == 'go_to':
            return [self._go(clause.place, pace, clause.text)]
        if clause.skill == 'return_to_dock':
            steps = [self._go('dock', pace, clause.text)]
            if 'face' in self.places['dock']:
                # The Create 3 only accepts a dock goal while it can see the dock.
                steps.append(Step('face', target=self.places['dock']['face'],
                                  place='the dock', clause=clause.text))
            return steps + [Step('dock', clause=clause.text)]
        if clause.skill == 'rotate':
            degrees = n.get('degrees', DEFAULT_ANGLE)
            sign = -1.0 if clause.direction == 'right' else 1.0
            return [Step('rotate', amount=sign * math.radians(degrees), pace=pace,
                         clause=clause.text)]
        if clause.skill == 'drive':
            # Turn in place, then a straight drive measured by odometry. Route goals (the
            # pilot, or Nav2 on the TurtleBot 4) curved and stopped ~0.2 m short in Gazebo.
            # "back up a little" reverses as far as the Create 3 allows; no turning around.
            default = BACKUP_LIMIT if clause.direction == 'backward' else DEFAULT_DISTANCE
            metres = min(n.get('metres', default), MAX_DISTANCE)
            drive = Step('drive', amount=metres, pace=pace, clause=clause.text)
            if clause.direction in ('left', 'right'):
                side = 1.0 if clause.direction == 'left' else -1.0
                return [Step('rotate', amount=side * math.pi / 2, pace=pace,
                             clause=clause.text), drive]
            if clause.direction == 'backward':
                if metres <= BACKUP_LIMIT:
                    return [Step('drive', amount=-metres, pace=pace, clause=clause.text)]
                return [Step('rotate', amount=math.pi, pace=pace, clause=clause.text), drive]
            return [drive]
        if clause.skill == 'find':
            label, queries = self._find_queries(clause)
            # "the pallet on your left": look that way, and choose among what is there.
            bearing = next((angle for pattern, angle in SIDES if pattern.search(clause.text)),
                           None)
            near_queries = ()
            if clause.near:
                # The whole phrase, plus what it is "with": "red barrels" stand out on their own.
                near_queries = (clause.near,) + tuple(
                    part.strip() for part in clause.near.split(' with ')[1:2] if part.strip())
            return [Step('find', place=label, queries=tuple(queries), pace=pace,
                         clause=clause.text, bearing=bearing, near=clause.near,
                         near_queries=near_queries)]
        if clause.skill == 'explore':
            count = max(1, n.get('count', DEFAULT_EXPLORE_STOPS))
            names = [p for p in self.places
                     if p not in ('dock', 'start') and not p.startswith('seen_')]
            if len(names) >= 2:
                self.rng.shuffle(names)
                return [self._go(name, pace, clause.text) for name in names[:count]]
            # No map of places (e.g. the depot): wander by odometry in varied directions.
            steps = []
            for _ in range(count):
                angle = self.rng.uniform(-math.pi, math.pi)
                steps.append(Step('go_relative', pace=pace, clause=clause.text,
                                  offset=(EXPLORE_HOP * math.cos(angle),
                                          EXPLORE_HOP * math.sin(angle))))
            return steps
        if clause.skill == 'wait':
            return [Step('wait', amount=n.get('seconds', DEFAULT_WAIT), clause=clause.text)]
        return [Step('stop', clause=clause.text)]

    def _go(self, place, pace, text):
        return Step('go_to', target=self.places[place]['xy'], place=place, pace=pace, clause=text)


def _top_two(probabilities):
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    return [r for r in ranked if r in SKILLS and r != 'unsupported'][:2] or ['go_to', 'explore']
