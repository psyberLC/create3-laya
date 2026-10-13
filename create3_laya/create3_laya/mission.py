"""Runs a list of steps on the robot.

`go_to` steps hand the driving decisions to the pilot (and so to Laya). The small fixed
motions (rotate, drive, wait) are done by code with odometry feedback, because they are
arithmetic, not judgment. Docking is delegated to whoever runs the mission: the ROS node
calls the Create 3 Dock and Undock actions; the 2D sim has no dock model and marks them
done.
"""

import math
from dataclasses import dataclass, field

from .pilot import Command, collision, forward_limit, wrap

STEP_TIMEOUT = {'go_to': 300.0, 'go_relative': 300.0, 'rotate': 20.0, 'face': 20.0,
                'drive': 20.0,                  # plus 10 s per metre, see MissionRunner.command
                'dock': 90.0, 'undock': 60.0}
DRIVE_SPEED = 0.25          # m/s for straight drives (the Create 3's safe maximum is 0.306)


@dataclass
class Step:
    kind: str                  # go_to | go_relative | rotate | drive | dock | undock | wait | stop
    target: tuple = None       # (x, y) for go_to; set when a go_relative step starts
    offset: tuple = None       # (forward, left) in metres from the pose where go_relative starts
    queries: tuple = ()        # detector phrases for find ('place' holds the object label)
    place: str = None
    amount: float = 0.0        # rad (left positive) for rotate, m (signed) for drive, s for wait
    pace: float = 1.0
    clause: str = ''
    bearing: float = None      # find: the asked-for direction, rad, left positive; None: any
    near: str = None           # find: a reference ("the shelves with red barrels")
    near_queries: tuple = ()   # find: detector phrases for the reference

    def describe(self):
        if self.kind == 'go_to':
            return f"go to {self.place or self.target}" + (' (slowly)' if self.pace < 1 else '')
        if self.kind == 'go_relative':
            forward, left = self.offset
            parts = []
            if forward:
                parts.append(f'{abs(forward):.1f} m {"forward" if forward > 0 else "back"}')
            if left:
                parts.append(f'{abs(left):.1f} m {"left" if left > 0 else "right"}')
            return 'go ' + ' and '.join(parts)
        if self.kind == 'rotate':
            side = 'left' if self.amount > 0 else 'right'
            return f'rotate {math.degrees(abs(self.amount)):.0f} deg {side}'
        if self.kind == 'drive':
            way = 'forward' if self.amount > 0 else 'backward'
            return f'drive {abs(self.amount):.2f} m {way}'
        if self.kind == 'wait':
            return f"wait {self.amount:.0f} s"
        if self.kind == 'find':
            side = {0.0: ' ahead', math.pi / 2: ' on the left', -math.pi / 2: ' on the right',
                    math.pi: ' behind'}.get(self.bearing, '')
            near = f' near the {self.near}' if self.near else ''
            return f'find the {self.place}{near}{side}'
        if self.kind == 'face':
            return f'turn to face {self.place or self.target}'
        return self.kind


@dataclass
class Event:
    t: float
    step: int
    status: str                # started | done | failed | stopped | replaced
    detail: str = ''
    mission: int = 0


@dataclass
class MissionRunner:
    pilot: object
    # kind -> callable(snap, first_call) returning 'running' | 'done' | 'failed', for steps
    # done by someone else (the ROS node maps 'dock' and 'undock' to Create 3 actions).
    action_handlers: dict = field(default_factory=dict)
    on_interrupt: object = None     # callable(step): a step was stopped, replaced or failed
    steps: list = field(default_factory=list)
    index: int = 0
    status: str = 'idle'            # idle | running | done | failed | stopped
    events: list = field(default_factory=list)
    mission: int = 0                # increments with every start()

    def start(self, steps, snap):
        if self.active:
            self._event(snap, 'replaced', 'a new task replaced this one')
            self._interrupted()
        self.mission += 1
        self.steps, self.index, self.status = list(steps), 0, 'running'
        self._begin(snap)

    def cancel(self, snap, detail='cancelled'):
        if self.status == 'running':
            self._finish(snap, 'stopped', detail)

    @property
    def active(self):
        return self.status == 'running'

    @property
    def current(self):
        return self.steps[self.index] if self.active else None

    # -- decisions are only needed while a go_to step drives -------------------------

    @property
    def wants_decision(self):
        step = self.current
        return self.needs_decisions() and self.pilot.wants_decision

    def needs_decisions(self):
        """True while the pilot drives a go step (not when a host handler, e.g. Nav2, does)."""
        step = self.current
        return (step is not None and step.kind in ('go_to', 'go_relative')
                and step.kind not in self.action_handlers)

    def prepare(self, snap):
        return self.pilot.prepare(snap)

    def apply(self, query, now, response=None, error=None, latency=None):
        return self.pilot.apply(query, now, response, error, latency)

    # -- control ---------------------------------------------------------------------

    def command(self, snap):
        step = self.current
        if step is None:
            return Command(0.0, 0.0, 'idle')
        limit = STEP_TIMEOUT.get(step.kind, math.inf)
        if step.kind == 'drive':
            limit += 10.0 * abs(step.amount)
        if snap.t - self._started > limit:
            self._finish(snap, 'failed', f'{step.describe()} timed out')
            return Command(0.0, 0.0, 'idle')
        if step.kind in self.action_handlers:
            result = self._run_external(step, snap)
        else:
            result = getattr(self, f'_run_{step.kind}')(step, snap)
        if result is None:              # step finished; start the next one on the next tick
            return Command(0.0, 0.0, 'step')
        return result

    def _run_go_relative(self, step, snap):
        return self._run_go_to(step, snap)

    def _run_go_to(self, step, snap):
        if self.pilot.reached(snap):
            return self._advance(snap)
        return self.pilot.command(snap)

    def _run_face(self, step, snap):
        error = wrap(math.atan2(step.target[1] - snap.y, step.target[0] - snap.x) - snap.yaw)
        if abs(error) <= 0.05:
            return self._advance(snap)
        rate = max(0.25, min(1.0, abs(error) / 0.3)) * 0.9
        return Command(0.0, math.copysign(rate, error), 'step')

    def _run_rotate(self, step, snap):
        self._turned += wrap(snap.yaw - self._last_yaw)
        self._last_yaw = snap.yaw
        remaining = abs(step.amount) - abs(self._turned)
        if remaining <= 0.015:
            return self._advance(snap)
        # Slow over the last ~20 deg so a 20 Hz loop stops within about a degree.
        rate = max(0.15, min(1.0, remaining / 0.35)) * 0.9 * step.pace
        return Command(0.0, math.copysign(rate, step.amount), 'step')

    def _run_drive(self, step, snap):
        # Distance along the heading the step started with; steer back onto that line.
        x0, y0 = self._start_pose
        heading = self._start_yaw + (0.0 if step.amount > 0 else math.pi)
        travelled = (snap.x - x0) * math.cos(heading) + (snap.y - y0) * math.sin(heading)
        remaining = abs(step.amount) - travelled
        if remaining <= 0.01:
            return self._advance(snap)
        if step.amount < 0 and 'backup_limit' in snap.hazards:
            return self._advance(snap, f'stopped after {travelled:.2f} m: the Create 3 limits '
                                       'reversing to about 15 cm')
        if collision(snap):
            return self._fail(snap, f'{step.describe()} stopped: bumper or cliff')
        speed = max(0.04, min(DRIVE_SPEED, 0.8 * remaining)) * step.pace
        if step.amount > 0:
            limit = forward_limit(self.pilot.ranges(snap))
            if limit <= 0.0:
                self._blocked_since = self._blocked_since or snap.t
                if snap.t - self._blocked_since > 3.0:
                    return self._fail(snap, f'{step.describe()} blocked by an obstacle after '
                                            f'{max(0.0, travelled):.2f} m')
                return Command(0.0, 0.0, 'safety')
            self._blocked_since = None
            speed = min(speed, limit)
        # Off the line by `side` metres and `error` radians: turn back toward it.
        c, s = math.cos(self._start_yaw), math.sin(self._start_yaw)
        side = -(snap.x - x0) * s + (snap.y - y0) * c
        error = wrap(self._start_yaw - snap.yaw) - math.copysign(1.5 * side, step.amount)
        turn = max(-0.4, min(0.4, 2.0 * error))
        return Command(math.copysign(speed, step.amount), turn, 'step')

    def _run_wait(self, step, snap):
        if snap.t - self._started >= step.amount:
            return self._advance(snap)
        return Command(0.0, 0.0, 'step')

    # Steps only a host can do; reached when no handler is registered for them.
    def _run_dock(self, step, snap):
        return self._advance(snap, 'no dock in this simulator; skipped')

    def _run_undock(self, step, snap):
        return self._advance(snap, 'no dock in this simulator; skipped')

    def _run_find(self, step, snap):
        return self._fail(snap, f'finding "{step.place}" needs a camera, and this robot has none')

    def _run_external(self, step, snap):
        """Hand the step to a host handler: handler(step, snap, first_call) returns 'running',
        'done' or 'failed', optionally as (status, detail) or ('running', Command) when the
        handler drives through cmd_vel itself. A plain 'running' yields an 'external' command,
        which hosts must not publish: something else (an action server, Nav2) is driving."""
        result = self.action_handlers[step.kind](step, snap, self._first_external_call)
        self._first_external_call = False
        status, extra = result if isinstance(result, tuple) else (result, None)
        if status == 'done':
            return self._advance(snap, extra or '')
        if status == 'failed':
            return self._fail(snap, extra or f'{step.describe()} failed')
        return extra if isinstance(extra, Command) else Command(0.0, 0.0, 'external')

    def _run_stop(self, step, snap):
        self._finish(snap, 'stopped', 'stop requested')
        return None

    # -- bookkeeping -----------------------------------------------------------------

    def _begin(self, snap):
        step = self.steps[self.index]
        self._started = snap.t
        self._start_pose = (snap.x, snap.y)
        self._start_yaw = snap.yaw
        self._turned, self._last_yaw = 0.0, snap.yaw
        self._blocked_since = None
        self._first_external_call = True
        if step.kind == 'go_relative':
            forward, left = step.offset
            c, s = math.cos(snap.yaw), math.sin(snap.yaw)
            step.target = (snap.x + c * forward - s * left, snap.y + s * forward + c * left)
        if step.kind in ('go_to', 'go_relative'):
            self.pilot.set_goal(step.target, step.pace)
        else:
            self.pilot.set_goal(None)
        self._event(snap, 'started', step.describe())

    def _advance(self, snap, detail=''):
        self._event(snap, 'done', detail)
        self.index += 1
        if self.index >= len(self.steps):
            self._finish(snap, 'done')
        else:
            self._begin(snap)
        return None

    def _fail(self, snap, detail):
        self._finish(snap, 'failed', detail)
        return None

    def _finish(self, snap, status, detail=''):
        if status != 'done':
            self._event(snap, status, detail)
            self._interrupted()
        self.status = status
        self.pilot.set_goal(None)

    def _interrupted(self):
        if self.on_interrupt is not None and self.index < len(self.steps):
            self.on_interrupt(self.steps[self.index])

    def _event(self, snap, status, detail=''):
        self.events.append(Event(snap.t, self.index, status, detail, self.mission))
