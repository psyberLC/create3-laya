"""The pilot: turns sensors and a goal into wheel commands, with Laya or code choosing motions.

Code plans the route over the known map and turns IR intensities, odometry and the next
waypoint into named buckets ("near", "ahead, slightly left"). From that labelled state, a
driver picks one motion primitive every half second:

  driver='code'  the rule policy below (`rule_action`), deterministic and instant
  driver='laya'  the Laya server answers two Choice questions in one request:
                   action       which motion primitive to run next
                   detour_side  which side to pass an unexpected obstacle on; held for a few
                                seconds and fed back as state, giving the model memory
                 Low-confidence or failed answers fall back to the rule policy. Zero-shot
                 Laya checkpoints drive poorly; use a checkpoint fine-tuned on this task
                 (tools/make_driving_data.py + laya-train).

Either way a safety layer running at control rate can slow, redirect or override the motion,
and bumps or cliffs trigger a back-up reflex.
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field

from . import create3
from .laya_client import LayaError, top_confidence
from .planner import Route

# name: (linear m/s, angular rad/s, criteria shown to the model). Kept short: Laya shares a
# token budget across a question's options (192 tokens on the English checkpoint).
ACTIONS = {
    'forward': (0.22, 0.0,
                'Drive straight at full speed: waypoint straight ahead and front sensors clear.'),
    'creep': (0.08, 0.0,
              'Drive straight slowly: waypoint ahead but something far in front.'),
    'veer_left': (0.12, 0.8,
                  'Curve forward to the left: waypoint ahead-left, or obstacle on the right.'),
    'veer_right': (0.12, -0.8,
                   'Curve forward to the right: waypoint ahead-right, or obstacle on the left.'),
    'turn_left': (0.0, 1.2,
                  'Spin in place to the left: waypoint left or behind, or front blocked and '
                  'left more open.'),
    'turn_right': (0.0, -1.2,
                   'Spin in place to the right: waypoint right or behind, or front blocked and '
                   'right more open.'),
    'back_up': (-0.08, 0.0,
                'Reverse slowly: only when the bumper is pressed or the front is very close.'),
}
ACTION_INSTRUCTIONS = (
    "Choose the robot's motion for the next half second. Head for the next waypoint and "
    'never drive toward an obstacle that a sensor reports as near or very close.')

DETOUR_CRITERIA = {
    'none': 'Nothing blocks the way toward the waypoint.',
    'left': 'Go around the obstacle on the left side; the left sensors are more open.',
    'right': 'Go around the obstacle on the right side; the right sensors are more open.',
}
DETOUR_INSTRUCTIONS = (
    'Does an obstacle stand between the robot and the waypoint, and if so, which side should '
    'the robot go around it on? The robot keeps this plan for the next few seconds.')
DETOUR_TEXT = {'none': 'none',
               'left': 'going around an obstacle on its left',
               'right': 'going around an obstacle on its right'}

SENSOR_LABELS = [s.label for s in create3.IR_SENSORS]
LEVEL = {'clear': 0, 'far': 1, 'near': 2, 'very close': 3, 'unknown': 2}
DIRECTION_LABELS = ['straight ahead', 'ahead, slightly left', 'ahead, slightly right',
                    'to the left', 'to the right', 'behind, on the left',
                    'behind, on the right', 'directly behind']
DISTANCE_LABELS = ['very close (under 0.6 m)', 'close (0.6 to 1.5 m)', 'medium (1.5 to 4 m)',
                   'far (over 4 m)']
PROGRESS_LABELS = ['just started', 'getting closer', 'no progress', 'getting farther']

REFLEX = (-0.08, 0.0)      # back-up command after a bump or cliff
REFLEX_TIME = 0.5          # s
UNKNOWN_RANGE = 0.08       # m, assumed range for a missing IR reading (cautious)


@dataclass
class Snapshot:
    t: float                  # seconds
    x: float                  # map frame, m
    y: float
    yaw: float                # rad
    ir: dict                  # create3 sensor name -> raw intensity; missing = unknown
    hazards: frozenset = frozenset()   # bump zone frame_ids, 'cliff', 'backup_limit'


@dataclass
class Query:
    t: float
    state: dict
    questions: dict
    generation: int = 0       # goal generation the query was asked under


@dataclass
class Decision:
    t: float
    action: str
    source: str               # code | laya | fallback:low_confidence | fallback:error | ...
    confidence: float = None
    probabilities: dict = field(default_factory=dict)
    detour: str = 'none'      # detour plan in force after this decision
    latency: float = None
    error: str = None
    usage: dict = None
    query: Query = None


@dataclass
class Command:
    v: float
    w: float
    source: str               # decision | reflex | safety | stale | idle | goal | step


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def proximity_label(distance):
    if distance is None:
        return 'unknown'
    if distance >= 0.19:
        return 'clear'
    if distance >= 0.10:
        return 'far'
    if distance >= 0.05:
        return 'near'
    return 'very close'


def bearing_label(angle):
    """Relative bearing (rad, left positive) -> words."""
    deg = math.degrees(angle)
    side = 'left' if deg > 0 else 'right'
    a = abs(deg)
    if a <= 12:
        return 'straight ahead'
    if a <= 40:
        return f'ahead, slightly {side}'
    if a <= 100:
        return f'to the {side}'
    if a <= 160:
        return f'behind, on the {side}'
    return 'directly behind'


def distance_label(distance):
    if distance < 0.6:
        return DISTANCE_LABELS[0]
    if distance < 1.5:
        return DISTANCE_LABELS[1]
    if distance < 4.0:
        return DISTANCE_LABELS[2]
    return DISTANCE_LABELS[3]


def make_state(direction, distance, sensors, bumps=(), progress='just started',
               detour='none', recent=(), cliff=False):
    """The labelled state both drivers see. `sensors` maps sensor label -> proximity label."""
    state = {
        'robot': 'iRobot Create 3: a round 34 cm floor robot that can drive forward, '
                 'back up, and turn in place.',
        'task': 'Follow waypoints along a route planned around known walls, without '
                'colliding. Obstacle sensors only see about 20 cm ahead of the bumper.',
        'next_waypoint': {'direction': direction, 'distance': distance},
        'obstacle_sensors_left_to_right': {label: sensors[label] for label in SENSOR_LABELS},
        'bumper': f"pressed on the {', '.join(bumps)}" if bumps else 'not pressed',
        'progress_last_5_seconds': progress,
        'detour_plan': DETOUR_TEXT[detour],
        'recent_actions_oldest_first': list(recent) or ['none yet'],
    }
    if cliff:
        state['cliff_sensor'] = 'floor edge detected under the front of the robot'
    return state


def _read(state):
    s = {k: LEVEL.get(v, 2) for k, v in state['obstacle_sensors_left_to_right'].items()}
    plan = next((side for side, text in DETOUR_TEXT.items()
                 if side != 'none' and state['detour_plan'] == text), None)
    return s, plan


def _openness(s):
    """Lower is more open: summed obstacle levels on each side."""
    return (s['far left'] + s['left'] + s['front left'],
            s['far right'] + s['right'] + s['front right'])


def rule_action(state):
    """The code driver: a deterministic policy over the labelled state."""
    s, plan = _read(state)
    direction = state['next_waypoint']['direction']
    if state['bumper'] != 'not pressed' or 'cliff_sensor' in state:
        return 'back_up'
    front = max(s['front left'], s['front'], s['front right'])
    if front >= 2:
        left, right = _openness(s)
        side = plan or ('left' if left < right else 'right' if right < left else
                        'right' if 'right' in direction else 'left')
        return f'turn_{side}'
    if plan:
        # Passing an obstacle: keep it beside the robot instead of turning back into it.
        obstacle_side = 'right' if plan == 'left' else 'left'
        seen = max(s[f'far {obstacle_side}'], s[obstacle_side], s[f'front {obstacle_side}'])
        if obstacle_side in direction and seen >= 1:
            return 'forward' if front == 0 else f'veer_{plan}'
    if direction == 'directly behind':
        return f"turn_{plan or 'left'}"
    if direction.startswith(('to the', 'behind')):
        return 'turn_left' if 'left' in direction else 'turn_right'
    if direction.startswith('ahead'):
        side = 'left' if 'left' in direction else 'right'
        if max(s[side], s[f'front {side}']) >= 2:
            return 'creep' if front >= 1 else 'forward'
        return f'veer_{side}'
    return 'creep' if front >= 1 else 'forward'


def rule_detour(state):
    """The code driver's detour plan: pass on the more open side when something is ahead."""
    s, plan = _read(state)
    direction = state['next_waypoint']['direction']
    front = max(s['front left'], s['front'], s['front right'])
    if front >= 1 and (direction == 'straight ahead' or direction.startswith('ahead')):
        left, right = _openness(s)
        return ('left' if left < right else 'right' if right < left else
                'right' if 'right' in direction else 'left')
    if plan and state['progress_last_5_seconds'] != 'getting closer':
        return plan
    return 'none'


class LayaPilot:
    def __init__(self, goal=None, *, planner=None, driver='code', decision_ttl=2.0,
                 min_confidence=0.3, detour_confidence=0.4, detour_hold=4.0,
                 goal_tolerance=0.25, progress_window=5.0, seed=None):
        if driver not in ('code', 'laya'):
            raise ValueError(f"driver must be 'code' or 'laya', not {driver!r}")
        self.planner = planner
        self.driver = driver
        self.decision_ttl = decision_ttl
        self.min_confidence = min_confidence
        self.detour_confidence = detour_confidence
        self.detour_hold = detour_hold
        self.goal_tolerance = goal_tolerance
        self.progress_window = progress_window
        self.rng = random.Random(seed)
        self.generation = 0
        self.set_goal(goal)

    @property
    def uses_model(self):
        return self.driver == 'laya'

    def set_goal(self, goal, pace=1.0):
        """Drive to goal (map x, y), or stand still for None. `pace` scales forward speeds."""
        self.goal = None if goal is None else (float(goal[0]), float(goal[1]))
        self.pace = pace
        self.route = Route(self.planner, self.goal) if self.planner and self.goal else None
        self.decision = None
        self.recent = deque(maxlen=4)
        self.progress = deque()          # (t, remaining route length)
        self.detour = 'none'
        self.detour_until = 0.0
        self.reflex_until = -1.0
        self.wants_decision = self.goal is not None
        self.generation += 1

    # -- perception ------------------------------------------------------------------

    def ranges(self, snap):
        """Sensor name -> estimated distance in m (None when the reading is missing)."""
        return {s.name: (create3.intensity_to_range(snap.ir[s.name]) if s.name in snap.ir
                         else None) for s in create3.IR_SENSORS}

    def waypoint(self, snap):
        return self.route.target(snap.x, snap.y) if self.route else self.goal

    def waypoint_vector(self, snap):
        wx, wy = self.waypoint(snap)
        dx, dy = wx - snap.x, wy - snap.y
        return math.hypot(dx, dy), wrap(math.atan2(dy, dx) - snap.yaw)

    def remaining(self, snap):
        if self.route:
            return self.route.remaining(snap.x, snap.y)
        return math.hypot(self.goal[0] - snap.x, self.goal[1] - snap.y)

    def reached(self, snap):
        if self.goal is None:
            return True
        return math.hypot(self.goal[0] - snap.x, self.goal[1] - snap.y) < self.goal_tolerance

    def _progress_label(self, t, remaining):
        self.progress.append((t, remaining))
        while len(self.progress) > 1 and self.progress[1][0] <= t - self.progress_window:
            self.progress.popleft()
        t0, r0 = self.progress[0]
        if t - t0 < self.progress_window * 0.8:
            return 'just started'
        if remaining < r0 - 0.15:
            return 'getting closer'
        if remaining > r0 + 0.15:
            return 'getting farther'
        return 'no progress'

    def build_state(self, snap):
        distance, bearing = self.waypoint_vector(snap)
        ranges = self.ranges(snap)
        if snap.t >= self.detour_until:
            self.detour = 'none'
        return make_state(
            bearing_label(bearing), distance_label(distance),
            {s.label: proximity_label(ranges[s.name]) for s in create3.IR_SENSORS},
            [create3.BUMP_LABELS[h] for h in sorted(snap.hazards) if h in create3.BUMP_LABELS],
            self._progress_label(snap.t, self.remaining(snap)), self.detour, self.recent,
            'cliff' in snap.hazards)

    def build_questions(self):
        # Shuffle option order on every request so no option gains from a position bias.
        actions = list(ACTIONS)
        self.rng.shuffle(actions)
        sides = list(DETOUR_CRITERIA)
        self.rng.shuffle(sides)
        return {
            'action': {'type': 'choice', 'instructions': ACTION_INSTRUCTIONS,
                       'criteria': {a: ACTIONS[a][2] for a in actions}},
            'detour_side': {'type': 'choice', 'instructions': DETOUR_INSTRUCTIONS,
                            'criteria': {s: DETOUR_CRITERIA[s] for s in sides}},
        }

    # -- decisions -------------------------------------------------------------------

    def decide_locally(self, snap):
        """The code driver: decide from the labelled state with no model call."""
        self.wants_decision = False
        state = self.build_state(snap)
        self._hold_detour(rule_detour(state), snap.t)
        decision = Decision(snap.t, rule_action(state), 'code',
                            query=Query(snap.t, state, {}, self.generation))
        return self._commit(decision)

    def prepare(self, snap):
        self.wants_decision = False
        return Query(snap.t, self.build_state(snap), self.build_questions(), self.generation)

    def apply(self, query, now, response=None, error=None, latency=None):
        """Turn Laya's answer (or a failure) into the active decision. Raises fatal errors.

        Returns None when the goal changed while the query was in flight.
        """
        if query.generation != self.generation or self.goal is None:
            return None
        if error is not None and not error.retryable:
            self.decision = None
            raise error
        decision = None
        if error is None:
            decision = self._from_answers(now, response.get('answers', {}), query.state)
        if decision is None:
            reason = 'error' if error is not None else 'invalid'
            decision = Decision(now, rule_action(query.state), f'fallback:{reason}',
                                error=str(error) if error else 'unusable answer')
        decision.latency = latency
        decision.query = query
        decision.usage = response.get('usage') if response else None
        return self._commit(decision)

    def decide(self, snap, client):
        """Synchronous prepare -> Laya -> apply, for callers without their own threading."""
        query = self.prepare(snap)
        response, error = None, None
        try:
            response = client.decide(query.state, query.questions)
        except LayaError as e:
            error = e
        return self.apply(query, snap.t, response, error)

    def _commit(self, decision):
        decision.detour = self.detour
        self.decision = decision
        self.recent.append(decision.action)
        return decision

    def _hold_detour(self, side, now):
        if side == 'none':
            self.detour, self.detour_until = 'none', 0.0
        elif side != self.detour or now >= self.detour_until:
            self.detour, self.detour_until = side, now + self.detour_hold

    def _from_answers(self, now, answers, state):
        action = answers.get('action') or {}
        probs = action.get('probabilities') or {}
        choice = action.get('choice')
        if choice not in ACTIONS or not probs:
            return None
        confidence = top_confidence(probs)

        detour = answers.get('detour_side') or {}
        side = detour.get('choice')
        side_conf = (top_confidence(detour['probabilities'])
                     if detour.get('probabilities') else None)
        if side in DETOUR_CRITERIA and side_conf is not None \
                and side_conf >= self.detour_confidence:
            self._hold_detour(side, now)

        source = 'laya'
        if confidence < self.min_confidence:
            choice, source = rule_action(state), 'fallback:low_confidence'
        return Decision(now, choice, source, confidence=confidence, probabilities=probs)

    # -- control (call at 20 Hz or faster) -------------------------------------------

    def command(self, snap):
        if self.reached(snap):
            return Command(0.0, 0.0, 'goal')
        if snap.t < self.reflex_until:
            return Command(*REFLEX, 'reflex')
        if collision(snap):
            # Bump or cliff: back off, forget the current plan, and decide again.
            self.reflex_until = snap.t + REFLEX_TIME
            self.recent.append('back_up')
            self.decision = None
            self.wants_decision = True
            return Command(*REFLEX, 'reflex')
        if self.decision is None:
            return Command(0.0, 0.0, 'idle')
        if snap.t - self.decision.t > self.decision_ttl:
            return Command(0.0, 0.0, 'stale')

        v, w, _ = ACTIONS[self.decision.action]
        if v <= 0:
            return Command(v, w, 'decision')
        v *= self.pace
        limit = forward_limit(self.ranges(snap), w)
        if v <= limit:
            return Command(v, w, 'decision')
        if limit > 0 or w != 0:
            return Command(min(v, limit), w, 'safety')
        return Command(0.0, open_side_turn(self.ranges(snap)), 'safety')


def collision(snap):
    """True when the bumper is pressed or a cliff / wheel drop is detected."""
    return any(h == 'cliff' or h in create3.BUMP_LABELS for h in snap.hazards)


def _known(ranges):
    return {k: (UNKNOWN_RANGE if r is None else r) for k, r in ranges.items()}


def forward_limit(ranges, w=0.0):
    """Highest safe forward speed given IR ranges, watching the side the robot turns toward."""
    ranges = _known(ranges)
    watched = list(create3.FRONT_SENSORS)
    if w > 0:
        watched += create3.LEFT_SENSORS
    elif w < 0:
        watched += create3.RIGHT_SENSORS
    distance = min(ranges[n] for n in watched)
    if distance >= 0.19:
        return create3.MAX_WHEEL_SPEED
    if distance >= 0.10:
        return 0.12
    if distance >= 0.05:
        return 0.05
    return 0.0


def open_side_turn(ranges, rate=1.0):
    ranges = _known(ranges)
    left = sum(ranges[n] for n in create3.LEFT_SENSORS)
    right = sum(ranges[n] for n in create3.RIGHT_SENSORS)
    return rate if left >= right else -rate
