#!/usr/bin/env python3
"""Give a simulated Create 3 tasks in plain English; Laya interprets them and the robot acts.

The robot drives create3_sim's maze (loaded from its maze.sdf) as a 2D simulation, so no ROS
or Gazebo is needed. Start the local Laya server first (scripts/serve_laya.ps1).

  python sim2d/run_sim.py --task "go to the east hall, then come back to the dock"
  python sim2d/run_sim.py --task "explore two rooms" --task "@30 stop"    # interrupt at t=30 s
  python sim2d/run_sim.py --interactive                                   # type tasks
  python sim2d/run_sim.py --driver laya --driving-model create3-driving  # Laya also drives
  python sim2d/run_sim.py --mock --task "..."                             # no server needed

Each run writes runs/<time>/ with run.json, decisions.jsonl and replay.html.
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'create3_laya'))
sys.path.insert(0, str(HERE))

from create3_laya import create3  # noqa: E402
from create3_laya.laya_client import LayaClient, LayaError  # noqa: E402
from create3_laya.mission import MissionRunner, Step  # noqa: E402
from create3_laya.mock_laya import MockLaya  # noqa: E402
from create3_laya.pilot import LayaPilot  # noqa: E402
from create3_laya.planner import GridPlanner  # noqa: E402
from create3_laya.tasks import TaskInterpreter, load_places  # noqa: E402
from create3_laya.world import World  # noqa: E402
from robot import SimCreate3  # noqa: E402

DT = 0.02                   # physics and control step, s
CODE_PERIOD = 0.2           # code driver decision period, s
FRAME_EVERY = 0.1           # replay sampling, s
MOCK_LATENCY = (0.05, 0.15)  # simulated delay for mock answers, s (GPU Laya is ~70-90 ms)
SOURCES = ['decision', 'reflex', 'safety', 'stale', 'idle', 'goal', 'step']
PLACES = ROOT / 'create3_laya' / 'config' / 'maze_places.json'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog='\n'.join(__doc__.split('\n')[4:]))
    p.add_argument('--task', action='append', default=[],
                   help='instruction; prefix "@SECONDS " to send it at that sim time')
    p.add_argument('--interactive', action='store_true', help='type tasks at a prompt')
    p.add_argument('--goal', nargs=2, type=float, metavar=('X', 'Y'),
                   help='skip language: drive straight to this map point')
    p.add_argument('--driver', choices=['code', 'laya'], default='code',
                   help='who picks motions along the route (default: code)')
    p.add_argument('--driving-model', default='create3-driving',
                   help='Laya checkpoint for --driver laya (default: create3-driving)')
    p.add_argument('--task-model', default='typed-decisions',
                   help='Laya checkpoint for task understanding (default: typed-decisions)')
    p.add_argument('--mock', action='store_true', help='offline heuristic stand-in for Laya')
    p.add_argument('--url', help='Laya server (default: $LAYA_URL or http://127.0.0.1:8000)')
    p.add_argument('--start', nargs=3, type=float, default=(0.0, 0.0, 0.0),
                   metavar=('X', 'Y', 'YAW'))
    p.add_argument('--world', default=str(HERE / 'worlds' / 'maze.sdf'))
    p.add_argument('--max-time', type=float, default=600.0, help='sim seconds per run')
    p.add_argument('--period', type=float, default=0.5, help='seconds between Laya decisions')
    p.add_argument('--min-confidence', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default=str(ROOT / 'runs'))
    p.add_argument('--quiet', action='store_true')
    return p.parse_args(argv)


class Simulation:
    def __init__(self, args):
        self.args = args
        self.world = World.from_sdf(args.world)
        self.planner = GridPlanner(self.world)
        self.places = load_places(PLACES)
        self.robot = SimCreate3(self.world, *args.start)
        self.pilot = LayaPilot(planner=self.planner, driver=args.driver,
                               min_confidence=args.min_confidence, seed=args.seed)
        self.runner = MissionRunner(self.pilot)
        self.rng = random.Random(args.seed)
        if args.mock:
            self.task_client = self.drive_client = MockLaya(seed=args.seed)
        else:
            # The first request after Laya was paged out of GPU memory can take ~20 s.
            self.task_client = LayaClient(args.url, args.task_model, timeout=30)
            self.drive_client = LayaClient(args.url, args.driving_model, timeout=2)
        self.interpreter = TaskInterpreter(self.places, self.task_client, seed=args.seed)
        self.t = 0.0
        self.queue = []                    # (steps, task record) waiting for the current mission
        self.frames, self.decisions, self.tasks, self.routes = [], [], [], []
        self.pending = None
        self.next_decision = 0.0
        self.next_frame = 0.0
        self.next_report = 10.0
        self.event_count = 0

    # -- tasks -----------------------------------------------------------------------

    def submit(self, text):
        busy = self.runner.active or bool(self.queue)
        start = time.perf_counter()
        plan = self.interpreter.interpret(text, busy=busy)
        wall = time.perf_counter() - start
        snap = self.robot.sense(self.t)
        if not plan.runnable:
            action = 'not run'
        elif plan.stops:
            self.runner.cancel(snap, f'"{text}"')
            self.queue.clear()
            action = 'stopped the robot'
        elif busy and plan.timing == 'after':
            action = 'queued after the current task'
        else:
            self.queue.clear()
            self.runner.start(plan.steps, snap)
            action = 'replaced the current task' if busy else 'started'
        record = {
            't': round(self.t, 2), 'text': text, 'action': action, 'summary': plan.summary(),
            'steps': [s.describe() for s in plan.steps], 'clarification': plan.clarification,
            'refusal': plan.refusal, 'busy': busy, 'timing': plan.timing,
            'timing_confidence': _round(plan.timing_confidence), 'wall_ms': round(wall * 1000, 1),
            'clauses': [{'text': c.text, 'skill': c.skill,
                         'skill_confidence': _round(c.skill_confidence), 'place': c.place,
                         'place_confidence': _round(c.place_confidence),
                         'direction': c.direction, 'pace': c.pace, 'numbers': c.numbers,
                         'destination': c.destination, 'resolved': c.resolved,
                         'skill_probabilities': c.answers.get('skill', {}).get('probabilities'),
                         'place_probabilities': c.answers.get('place', {}).get('probabilities')}
                        for c in plan.clauses],
        }
        started = action in ('started', 'replaced the current task')
        record['mission'] = self.runner.mission if started else None
        if action.startswith('queued'):
            self.queue.append((plan.steps, record))
        self.tasks.append(record)
        if not self.args.quiet:
            print(f'[t={self.t:6.1f}s] "{text}"\n    -> {action}: {plan.summary()}')
        return plan

    def drive_to(self, goal):
        self.runner.start([Step('go_to', target=tuple(goal), place=f'point {goal}')],
                          self.robot.sense(self.t))

    # -- simulation loop -------------------------------------------------------------

    def run(self, schedule=(), until_idle=True):
        """Advance until missions and the schedule are used up, or time runs out."""
        schedule = sorted(schedule)
        limit = self.t + self.args.max_time
        while self.t <= limit:
            while schedule and schedule[0][0] <= self.t:
                self.submit(schedule.pop(0)[1])
            self.tick()
            if until_idle and not self.runner.active and not self.queue and not schedule:
                return True
            self.t = round(self.t + DT, 6)
        return False

    def tick(self):
        """One control and physics step of DT at the current time (the caller advances t)."""
        snap = self.robot.sense(self.t)
        self._decide(snap)
        command = self.runner.command(snap)
        if not self.runner.active and self.queue:
            steps, record = self.queue.pop(0)
            self.runner.start(steps, snap)
            record['mission'] = self.runner.mission
            record['action'] += f' (started at t={self.t:.1f} s)'
        self.robot.step(command.v, command.w, DT)
        self._record(snap, command)
        return snap, command

    def _decide(self, snap):
        if not self.runner.needs_decisions():
            self.pending = None
            return
        if not self.pilot.uses_model:
            if self.t >= self.next_decision or self.pilot.wants_decision:
                self._log_decision(self.pilot.decide_locally(snap), None, None)
                self.next_decision = self.t + CODE_PERIOD
            return
        if self.pending is None and (self.t >= self.next_decision or self.pilot.wants_decision):
            query = self.runner.prepare(snap)
            start = time.perf_counter()
            response, error = None, None
            try:
                response = self.drive_client.decide(query.state, query.questions)
            except LayaError as e:
                error = e
            wall = time.perf_counter() - start
            latency = self.rng.uniform(*MOCK_LATENCY) if self.args.mock else wall
            self.pending = (self.t + latency, query, response, error, latency, wall)
            self.next_decision = self.t + self.args.period
        if self.pending is not None and self.t >= self.pending[0]:
            _, query, response, error, latency, wall = self.pending
            self.pending = None
            decision = self.runner.apply(query, self.t, response, error, latency)
            if decision is not None:
                self._log_decision(decision, wall, response)

    def _log_decision(self, d, wall, response):
        self.decisions.append({
            't': round(d.t, 3), 'action': d.action, 'source': d.source,
            'confidence': _round(d.confidence), 'probabilities': d.probabilities,
            'detour': d.detour, 'latency': _round(d.latency), 'wall_seconds': _round(wall),
            'error': d.error, 'usage': d.usage, 'server_model': (response or {}).get('model'),
            'state': d.query.state if d.query else None,
        })

    def _record(self, snap, command):
        if self.pilot.route is not None and self.pilot.route.path is not None:
            path = self.pilot.route.path
            if not self.routes or self.routes[-1]['id'] != id(path):
                self.routes.append({'t': round(self.t, 2), 'id': id(path),
                                    'path': [[round(x, 2), round(y, 2)] for x, y in path]})
        if self.t >= self.next_frame - 1e-9:
            step = self.runner.current
            self.frames.append([round(snap.t, 2), round(snap.x, 4), round(snap.y, 4),
                                round(snap.yaw, 4),
                                [snap.ir[s.name] for s in create3.IR_SENSORS],
                                round(command.v, 3), round(command.w, 3),
                                SOURCES.index(command.source), sorted(snap.hazards),
                                self.runner.index if step else -1,
                                self.runner.mission if step else -1])
            self.next_frame += FRAME_EVERY
        while self.event_count < len(self.runner.events):
            e = self.runner.events[self.event_count]
            self.event_count += 1
            if not self.args.quiet and e.status != 'started':
                print(f'[t={e.t:6.1f}s]    step {e.step + 1} {e.status}'
                      + (f': {e.detail}' if e.detail else ''))
        if not self.args.quiet and self.t >= self.next_report and self.runner.active:
            print(f'[t={self.t:6.1f}s]    at ({snap.x:5.2f}, {snap.y:5.2f}) doing: '
                  f'{self.runner.current.describe()}')
            self.next_report += 15.0
        elif self.t >= self.next_report:
            self.next_report += 15.0

    # -- output ----------------------------------------------------------------------

    def summary(self):
        latencies = [d['latency'] for d in self.decisions if d['latency'] is not None]
        statuses = Counter(e.status for e in self.runner.events if e.status != 'started')
        return {
            'sim_seconds': round(self.t, 2),
            'mission_status': self.runner.status,
            'driver': self.args.driver,
            'backend': 'mock' if self.args.mock else self.task_client.url,
            'task_model': None if self.args.mock else self.args.task_model,
            'driving_model': (self.args.driving_model if self.args.driver == 'laya'
                              and not self.args.mock else None),
            'tasks': len(self.tasks),
            'step_results': dict(statuses),
            'final_pose': [round(self.robot.x, 3), round(self.robot.y, 3),
                           round(self.robot.yaw, 3)],
            'distance_driven_m': round(self.robot.odometer, 2),
            'blocked_steps': self.robot.collisions,
            'decisions': len(self.decisions),
            'decision_sources': dict(Counter(d['source'] for d in self.decisions)),
            'latency_p50_ms': _ms(_percentile(latencies, 0.5)),
            'latency_p95_ms': _ms(_percentile(latencies, 0.95)),
            'task_wall_ms': [t['wall_ms'] for t in self.tasks],
        }

    def write(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        data = {
            'summary': self.summary(),
            'boxes': [[round(c, 4) for xy in b.corners() for c in xy] for b in self.world.boxes],
            'places': {n: p['xy'] for n, p in self.places.items()},
            'robot_radius': create3.ROBOT_RADIUS,
            'ir_sensors': [[s.label, s.x, s.y, s.yaw] for s in create3.IR_SENSORS],
            'ir_max_range': create3.IR_MAX_RANGE,
            'sources': SOURCES,
            'frames': self.frames,
            'decisions': self.decisions,
            'tasks': self.tasks,
            'routes': [{'t': r['t'], 'path': r['path']} for r in self.routes],
            'events': [{'t': round(e.t, 2), 'mission': e.mission, 'step': e.step,
                        'status': e.status, 'detail': e.detail} for e in self.runner.events],
            'steps_by_task': [t['steps'] for t in self.tasks],
        }
        (folder / 'run.json').write_text(json.dumps(data), encoding='utf-8')
        with open(folder / 'decisions.jsonl', 'w', encoding='utf-8') as f:
            for d in self.decisions:
                f.write(json.dumps(d) + '\n')
        template = (HERE / 'replay_template.html').read_text(encoding='utf-8')
        payload = json.dumps(data).replace('</', '<\\/')
        (folder / 'replay.html').write_text(template.replace('"__RUN_DATA__"', payload),
                                            encoding='utf-8')
        return folder / 'replay.html'


def _round(value, digits=4):
    return None if value is None else round(value, digits)


def _ms(seconds):
    return None if seconds is None else round(seconds * 1000, 1)


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def parse_schedule(tasks):
    schedule = []
    for text in tasks:
        at = 0.0
        if text.startswith('@'):
            stamp, _, text = text[1:].partition(' ')
            at = float(stamp)
        schedule.append((at, text.strip()))
    return schedule


def check_server(sim):
    if sim.args.mock:
        return
    try:
        sim.task_client.decide({'instruction': 'stop'},
                               {'probe': {'type': 'choice', 'instructions': 'Is this a test?',
                                          'criteria': {'yes': 'yes', 'no': 'no'}}})
    except LayaError as e:
        sys.exit(f'{e}\nStart the local model first:  powershell -File scripts\\serve_laya.ps1'
                 f'\n(or add --mock to run without it)')


def main():
    args = parse_args()
    if not (args.task or args.interactive or args.goal):
        sys.exit('give --task "...", --interactive, or --goal X Y (see --help)')
    sim = Simulation(args)
    check_server(sim)
    folder = Path(args.out) / datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.goal:
        sim.drive_to(args.goal)
        sim.run()
    if args.task:
        sim.run(parse_schedule(args.task))
    if args.interactive:
        names = ', '.join(n.replace('_', ' ') for n in sim.places)
        print(f'Places: {names}.\nType a task (empty line or "quit" to exit).')
        while True:
            try:
                text = input('task> ').strip()
            except EOFError:
                break
            if text.lower() in ('', 'quit', 'exit'):
                break
            sim.submit(text)
            sim.run()
            print(f'    replay: {sim.write(folder)}')
    replay = sim.write(folder)
    print(json.dumps(sim.summary(), indent=2))
    print(f'replay: {replay}')
    return 0 if sim.runner.status in ('done', 'stopped', 'idle') else 1


if __name__ == '__main__':
    sys.exit(main())
