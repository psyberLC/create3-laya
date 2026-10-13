"""What the launcher drives, kept free of Tk so it can be tested: the two local model servers,
the 2D simulator stepped live on a thread, and Gazebo simulators running in WSL.

Every simulator reports through the same status events the ROS node publishes on task_status
({'event': 'task' | 'step' | 'mission' | 'seen' | 'error' | 'heartbeat', ...}), plus
{'event': 'session', 'status': 'ready' | 'ended'} from the launcher itself.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'create3_laya'))
sys.path.insert(0, str(ROOT / 'sim2d'))

from create3_laya import create3  # noqa: E402
from create3_laya.laya_client import LayaError  # noqa: E402

DISTRO = 'Ubuntu-24.04'
WSL_USER = 'ros'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


# -- processes ---------------------------------------------------------------------------

class ManagedProcess:
    """A child process (no console window) whose output lines go to sink(name, line); the
    sink gets line=None once the process has exited."""

    def __init__(self, name, args, sink, stdin=False, env=None):
        self.name = name
        self.proc = subprocess.Popen(
            args, cwd=ROOT, env=env, creationflags=NO_WINDOW,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        threading.Thread(target=self._pump, args=(sink,), daemon=True).start()

    def _pump(self, sink):
        for raw in self.proc.stdout:
            sink(self.name, raw.decode('utf-8', 'replace').rstrip('\r\n'))
        self.proc.wait()
        sink(self.name, None)

    @property
    def running(self):
        return self.proc.poll() is None

    def send(self, line):
        self.proc.stdin.write((line + '\n').encode('utf-8'))
        self.proc.stdin.flush()

    def close_input(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass

    def kill_tree(self):
        if self.running:
            kill_tree(self.proc.pid)


def http_json(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except (OSError, ValueError, urllib.error.URLError):
        return None


# -- model servers -----------------------------------------------------------------------

def kill_tree(pid):
    subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True,
                   creationflags=NO_WINDOW)


def listening_pid(port):
    """PID of the process listening on a local TCP port, or None (netstat, any locale)."""
    out = subprocess.run(['netstat', '-ano', '-p', 'TCP'], capture_output=True, text=True,
                         creationflags=NO_WINDOW).stdout
    for parts in (line.split() for line in out.splitlines()):
        # A listening socket's foreign address is 0.0.0.0:0, whatever the state column says.
        if len(parts) == 5 and parts[1].endswith(f':{port}') and parts[2] == '0.0.0.0:0':
            return int(parts[4])
    return None


class Service:
    """One of the local model servers, started through its scripts\\serve_*.ps1. Its output
    goes to runs\\logs\\<name>.log (not a pipe), so it keeps running after the launcher quits."""

    def __init__(self, name, script, port, health_path, describe):
        self.name, self.script, self.port = name, script, port
        self.health_url = f'http://127.0.0.1:{port}{health_path}'
        self.describe = describe          # health JSON -> short text
        self.log = ROOT / 'runs' / 'logs' / f'{name.lower()}.log'
        self.proc = None
        self.state, self.detail = 'unknown', ''
        self._tail_stop = threading.Event()

    @property
    def owned(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def available(self):
        return self.state in ('ready', 'external')

    def start(self, sink):
        if self.owned or self.available:
            return
        self.log.parent.mkdir(parents=True, exist_ok=True)
        start = self.log.stat().st_size if self.log.exists() else 0
        with open(self.log, 'ab') as out:
            out.write(f'\n== started by the launcher {datetime.now():%Y-%m-%d %H:%M:%S}\n'
                      .encode())
            self.proc = subprocess.Popen(
                ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                 str(ROOT / 'scripts' / self.script)],
                cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                creationflags=NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        self.state, self.detail = 'starting', 'loading the model'
        self._tail_stop.set()
        self._tail_stop = threading.Event()
        threading.Thread(target=self._tail, args=(start, sink, self._tail_stop),
                         daemon=True).start()

    def _tail(self, offset, sink, stop):
        with open(self.log, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(offset)
            while not stop.is_set():
                line = f.readline()
                if line:
                    sink(self.name, line.rstrip('\r\n'))
                else:
                    time.sleep(0.3)

    def stop(self):
        """Stop the server, whoever started it."""
        pid = self.proc.pid if self.owned else listening_pid(self.port)
        if pid:
            kill_tree(pid)

    def poll(self):
        """Refresh state from the health endpoint; returns True if it changed."""
        info = http_json(self.health_url)
        before = (self.state, self.detail)
        if info is not None:
            self.state = 'ready' if self.owned else 'external'
            self.detail = self.describe(info)
        elif self.owned:
            self.state, self.detail = 'starting', 'loading the model'
        elif self.proc is not None and self.proc.returncode not in (None, 0, 1):
            self.state, self.detail = 'failed', f'exited with code {self.proc.returncode}'
        else:
            self.state, self.detail = 'stopped', ''
        return (self.state, self.detail) != before


def warm_up(models, timeout=120.0):
    """One small request per checkpoint, so the first real task isn't the one that waits while
    weights page back into GPU memory (or a checkpoint loads). Returns the seconds it took."""
    from create3_laya.laya_client import LayaClient
    start = time.perf_counter()
    for model in models:
        LayaClient(model=model, timeout=timeout, max_retries=0).decide(
            {'instruction': 'stop'}, {'probe': {'type': 'choice', 'instructions': 'Is this a test?',
                                                'criteria': {'yes': 'yes', 'no': 'no'}}})
    return time.perf_counter() - start


def laya_service():
    def describe(info):
        loaded = ', '.join(info.get('loaded') or []) or 'no checkpoint loaded yet'
        return f"{info.get('device', '?')} · {loaded}"
    return Service('Laya', 'serve_laya.ps1', 8000, '/health', describe)


def vision_service():
    def describe(info):
        return f"{info.get('device', '?')} · OWLv2"
    return Service('Vision', 'serve_vision.ps1', 8001, '/v1/health', describe)


# -- WSL -----------------------------------------------------------------------------------

def wsl_path(path):
    """C:\\Users\\x\\file -> /mnt/c/Users/x/file"""
    p = Path(path).resolve()
    return f'/mnt/{p.drive.rstrip(":").lower()}/' + '/'.join(p.parts[1:])


def wsl_status():
    """(ok, message): can this machine run the Gazebo simulators?"""
    try:
        listing = subprocess.run(['wsl.exe', '-l', '-q'], capture_output=True, timeout=30,
                                 creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False, 'WSL is not available'
    if DISTRO not in listing.decode('utf-16-le', 'ignore').replace('\x00', '').split():
        return False, f'{DISTRO} is not installed (README: "Gazebo on Windows")'
    try:
        built = subprocess.run(
            ['wsl.exe', '-d', DISTRO, '-u', WSL_USER, '--', 'test', '-f',
             f'/home/{WSL_USER}/create3_ws/install/setup.bash'],
            capture_output=True, timeout=60, creationflags=NO_WINDOW).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False, f'{DISTRO} did not respond'
    if not built:
        return False, 'ROS workspace not built (run scripts/wsl/setup_ros_jazzy.sh)'
    return True, f'{DISTRO} with ROS 2 Jazzy'


class GazeboSession:
    """A Gazebo simulator in WSL (scripts/wsl/sim_session.sh): prompts in, status events out."""

    SCRIPT = ROOT / 'scripts' / 'wsl' / 'sim_session.sh'

    def __init__(self, robot, on_event, on_log, driver='code', gui=True):
        self.robot, self.on_event, self.on_log = robot, on_event, on_log
        self.ready = False
        env = ['env', f'DRIVER={driver}', f'GUI={"true" if gui else "false"}']
        self.process = ManagedProcess(
            f'Gazebo ({robot})',
            ['wsl.exe', '-d', DISTRO, '-u', WSL_USER, '--', *env, 'bash',
             wsl_path(self.SCRIPT), robot],
            self._line, stdin=True)

    def _line(self, name, line):
        if line is None:
            self.ready = False
            self.on_event({'event': 'session', 'status': 'ended'})
        elif line == '@@ready':
            self.ready = True
            self.on_event({'event': 'session', 'status': 'ready'})
        elif line.startswith('@@{'):
            try:
                self.on_event(json.loads(line[2:]))
            except ValueError:
                self.on_log(name, line)
        else:
            self.on_log(name, line)

    def send(self, text):
        self.process.send(text)

    def set_speed(self, factor):
        """Ask Gazebo for `factor` x real time (it runs as fast as it can up to that). Blocks
        for the WSL round trip (~1-2 s); returns the script's message."""
        result = subprocess.run(
            ['wsl.exe', '-d', DISTRO, '-u', WSL_USER, '--', 'bash', wsl_path(self.SCRIPT),
             'speed', f'{factor:g}'],
            capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW)
        message = (result.stdout or result.stderr).strip().splitlines()
        return result.returncode == 0, message[-1] if message else ''

    def stop(self, grace=45.0):
        """Close the prompt pipe so the session shuts Gazebo down; force it after `grace` s."""
        self.process.close_input()
        try:
            self.process.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            stop_leftover_sessions()
            self.process.kill_tree()


def stop_leftover_sessions():
    subprocess.run(['wsl.exe', '-d', DISTRO, '-u', WSL_USER, '--', 'bash',
                    wsl_path(GazeboSession.SCRIPT), 'stop'],
                   capture_output=True, timeout=120, creationflags=NO_WINDOW)


# -- 2D simulator --------------------------------------------------------------------------

OUTCOMES = {'started': 'started', 'replaced the current task': 'replaced current task',
            'stopped the robot': 'stopped', 'queued after the current task': 'queued'}


class LiveSim2D:
    """sim2d's maze stepped in real time on a thread. Prompts may arrive at any moment, as on
    the robot: "stop", "go to the west room instead", "when you are done, ...". """

    def __init__(self, on_event, driver='code', mock=False, speed=1.0):
        import run_sim
        self.run_sim = run_sim
        args = ['--quiet', '--driver', driver] + (['--mock'] if mock else [])
        self.sim = run_sim.Simulation(run_sim.parse_args(args))
        self.on_event = on_event
        self.speed = speed
        self.prompts = queue.Queue()
        self.view = None                  # latest drawable state, replaced as a whole
        self.trail = deque(maxlen=900)
        self.replay = None
        self._resync = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # static scene for drawing
    def scene(self):
        return {'boxes': [b.corners() for b in self.sim.world.boxes],
                'places': {n: p['xy'] for n, p in self.sim.places.items()},
                'robot_radius': create3.ROBOT_RADIUS}

    def start(self):
        self._thread.start()

    def send(self, text):
        self.prompts.put(text)

    def set_speed(self, speed):
        self.speed, self._resync = speed, True

    def stop(self, folder=None):
        """Stop the clock and write the replay (to runs\\<time>\\ by default); returns its path."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        folder = Path(folder or ROOT / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.replay = self.sim.write(folder)
        return self.replay

    def _loop(self):
        sim, rs = self.sim, self.run_sim
        if not sim.args.mock:
            try:
                sim.task_client.decide({'instruction': 'stop'}, {'probe': {
                    'type': 'choice', 'instructions': 'Is this a test?',
                    'criteria': {'yes': 'yes', 'no': 'no'}}})
            except LayaError as e:
                self.on_event({'event': 'error', 'detail': f'Laya is not reachable: {e}'})
                self.on_event({'event': 'session', 'status': 'ended'})
                return
        self.on_event({'event': 'session', 'status': 'ready'})
        seen_events, last = 0, (0, 'idle')
        wall0, sim0, next_view = time.perf_counter(), sim.t, 0.0
        while not self._stop.is_set():
            if not self.prompts.empty():
                self._submit(self.prompts.get())
                self._resync = True       # Laya's thinking time must not make the clock race
            if self._resync:
                wall0, sim0, self._resync = time.perf_counter(), sim.t, False
            snap, command = sim.tick()
            sim.t = round(sim.t + rs.DT, 6)
            for e in sim.runner.events[seen_events:]:
                self.on_event({'event': 'step', 'mission': e.mission, 'step': e.step,
                               'status': e.status, 'detail': e.detail, 't': e.t})
            seen_events = len(sim.runner.events)
            state = (sim.runner.mission, sim.runner.status)
            if state[0] != last[0] and last[1] == 'running':
                # The mission ended and a queued one started within this one tick; its last
                # event says how it ended ('replaced' needs no report).
                ending = next((e.status for e in reversed(sim.runner.events)
                               if e.mission == last[0]), None)
                if ending in ('done', 'failed', 'stopped'):
                    self.on_event({'event': 'mission', 'mission': last[0], 'status': ending})
            if state != last and state[1] in ('done', 'failed', 'stopped'):
                self.on_event({'event': 'mission', 'mission': state[0], 'status': state[1]})
            last = state
            if sim.t >= next_view:
                self._publish_view(snap, command)
                next_view = sim.t + 0.05
            ahead = (sim.t - sim0) / self.speed - (time.perf_counter() - wall0)
            if ahead > 0.002:
                time.sleep(ahead)
            elif ahead < -0.5:            # fell behind (a slow model call): resync, don't race
                wall0, sim0 = time.perf_counter(), sim.t

    def _submit(self, text):
        sim = self.sim
        try:
            plan = sim.submit(text)
        except LayaError as e:
            self.on_event({'event': 'error', 'text': text, 'detail': f'Laya unavailable: {e}'})
            return
        record = sim.tasks[-1]
        if not plan.runnable:
            outcome = 'refused' if plan.refusal else 'needs clarification'
        else:
            outcome = OUTCOMES.get(record['action'], record['action'])
        self.on_event({'event': 'task', 'text': text, 'outcome': outcome,
                       'summary': plan.summary(), 'mission': record['mission'],
                       'clauses': record['clauses']})

    def _publish_view(self, snap, command):
        sim, pilot = self.sim, self.sim.pilot
        if not self.trail or (abs(self.trail[-1][0] - snap.x) + abs(self.trail[-1][1] - snap.y)
                              > 0.03):
            self.trail.append((snap.x, snap.y))
        route = None
        if pilot.route is not None and pilot.route.path is not None and sim.runner.active:
            route = list(pilot.route.path)
        ranges = pilot.ranges(snap)
        self.view = {
            't': sim.t, 'x': snap.x, 'y': snap.y, 'yaw': snap.yaw,
            'v': command.v, 'source': command.source,
            'route': route, 'trail': list(self.trail),
            'ir': [(s.x, s.y, s.yaw, ranges.get(s.name)) for s in create3.IR_SENSORS],
            'hazards': sorted(snap.hazards),
            'step': sim.runner.current.describe() if sim.runner.active else None,
        }


# -- transcript --------------------------------------------------------------------------

def describe_event(event):
    """(text, style) for the conversation pane, or None for events it doesn't show."""
    kind = event.get('event')
    if kind == 'task':
        outcome, summary = event.get('outcome'), event.get('summary', '')
        if outcome == 'needs clarification':
            return summary, 'ask'
        if outcome == 'refused':
            return summary, 'refuse'
        if outcome == 'stopped':
            return 'Stopping.', 'robot'
        if outcome == 'queued':
            return f'Queued until the current task ends: {summary}', 'robot'
        if outcome == 'replaced current task':
            return f'Dropping the current task. Plan: {summary}', 'robot'
        return f'Plan: {summary}', 'robot'
    if kind == 'step':
        n, status, detail = event.get('step', 0) + 1, event.get('status'), event.get('detail')
        if status == 'started':
            return f'  step {n}: {detail}', 'step'
        if status == 'done':
            return f'  step {n} done' + (f' ({detail})' if detail else ''), 'step'
        return f'  step {n} {status}' + (f': {detail}' if detail else ''), 'problem'
    if kind == 'mission':
        status = event.get('status')
        return ({'done': ('Task done.', 'done'), 'failed': ('Task failed.', 'problem'),
                 'stopped': ('Task stopped.', 'robot')}.get(status, (f'Task {status}.', 'robot')))
    if kind == 'seen':
        return (f"  saw a {event.get('label')} {event.get('distance', 0):.1f} m away "
                f"(score {event.get('score', 0):.2f})", 'step')
    if kind == 'error':
        return event.get('detail', 'error'), 'problem'
    if kind == 'session':
        return ({'ready': ('Simulator ready. Type a task below.', 'info'),
                 'ended': ('Simulator stopped.', 'info')}.get(event.get('status')))
    return None
