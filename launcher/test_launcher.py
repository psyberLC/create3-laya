"""Tests for the launcher's back ends (no window, no Laya server, no WSL needed).

  .venv-laya\\Scripts\\python.exe launcher\\test_launcher.py
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as be  # noqa: E402
import theme  # noqa: E402


def wait_for(events, predicate, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        for e in list(events):
            if predicate(e):
                return e
        time.sleep(0.05)
    raise AssertionError(f'no matching event in {timeout} s; got {events[-8:]}')


class Pieces(unittest.TestCase):
    def test_wsl_path(self):
        self.assertEqual(be.wsl_path(r'C:\Users\me\Projects and Docs\x.sh'),
                         '/mnt/c/Users/me/Projects and Docs/x.sh')

    def test_transcript_lines(self):
        ask = be.describe_event({'event': 'task', 'outcome': 'needs clarification',
                                 'summary': 'Where should I go?'})
        self.assertEqual(ask, ('Where should I go?', 'ask'))
        self.assertEqual(be.describe_event({'event': 'step', 'step': 0, 'status': 'started',
                                            'detail': 'go to east_hall'}),
                         ('  step 1: go to east_hall', 'step'))
        self.assertEqual(be.describe_event({'event': 'mission', 'status': 'done'})[1], 'done')
        self.assertEqual(be.describe_event({'event': 'step', 'step': 1, 'status': 'failed',
                                            'detail': 'blocked'})[1], 'problem')
        self.assertIsNone(be.describe_event({'event': 'heartbeat'}))

    def test_theme_choice_is_remembered(self):
        from pathlib import Path
        path = Path(tempfile.mkdtemp()) / 'runs' / 'launcher.json'
        self.assertEqual(theme.load_choice(path), 'System')        # nothing saved yet
        theme.save_choice(path, 'Dark')
        self.assertEqual(theme.load_choice(path), 'Dark')
        path.write_text('{"theme": "Purple"}', encoding='utf-8')
        self.assertEqual(theme.load_choice(path), 'System')        # unknown: the default
        path.write_text('not json', encoding='utf-8')
        self.assertEqual(theme.load_choice(path), 'System')
        self.assertTrue(theme.is_dark('Dark'))
        self.assertFalse(theme.is_dark('Light'))
        self.assertEqual(set(theme.LIGHT), set(theme.DARK) - {
            'button', 'button_hover', 'button_pressed', 'border', 'disabled', 'accent'})


class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({'status': 'ok', 'device': 'cuda', 'loaded': ['typed-decisions']})
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


class Services(unittest.TestCase):
    def test_health_and_port_owner(self):
        server = HTTPServer(('127.0.0.1', 0), Health)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            svc = be.Service('Test', 'none.ps1', port, '/health',
                             lambda info: ', '.join(info['loaded']))
            self.assertTrue(svc.poll())
            self.assertEqual((svc.state, svc.detail), ('external', 'typed-decisions'))
            self.assertTrue(svc.available)
            self.assertEqual(be.listening_pid(port), os.getpid())
        finally:
            server.shutdown()
            server.server_close()
        svc.poll()
        self.assertEqual(svc.state, 'stopped')


class LiveSim(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.sim = be.LiveSim2D(self.events.append, mock=True, speed=50.0)
        self.sim.start()
        wait_for(self.events, lambda e: e == {'event': 'session', 'status': 'ready'})

    def tearDown(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(self.sim.stop(folder).exists())

    def test_runs_a_task_and_draws(self):
        self.sim.send('go to the east room')
        task = wait_for(self.events, lambda e: e['event'] == 'task')
        self.assertEqual(task['outcome'], 'started')
        wait_for(self.events, lambda e: e['event'] == 'mission' and e['status'] == 'done')
        view = self.sim.view
        self.assertLess(abs(view['x'] - self.sim.sim.places['east_room']['xy'][0]), 0.5)
        self.assertEqual(len(view['ir']), 7)
        self.assertGreater(len(view['trail']), 10)

    def test_queued_task_follows_a_reported_end(self):
        self.sim.send('go to the west room')
        wait_for(self.events, lambda e: e['event'] == 'step' and e['status'] == 'started')
        self.sim.send('when you are done, go to the south room')
        queued = wait_for(self.events, lambda e: e['event'] == 'task'
                          and e['text'].startswith('when'))
        self.assertEqual(queued['outcome'], 'queued')
        def missions_reported():
            return [e for e in list(self.events) if e['event'] == 'mission']
        wait_for(self.events, lambda e: len(missions_reported()) >= 2)
        missions = missions_reported()
        self.assertEqual([(m['mission'], m['status']) for m in missions[:2]],
                         [(1, 'done'), (2, 'done')])

    def test_stop_interrupts(self):
        self.sim.send('go to the north west room')
        wait_for(self.events, lambda e: e['event'] == 'step' and e['status'] == 'started')
        self.sim.send('stop')
        wait_for(self.events, lambda e: e['event'] == 'mission' and e['status'] == 'stopped')
        self.assertTrue(any(e['event'] == 'task' and e['outcome'] == 'stopped'
                            for e in self.events))


if __name__ == '__main__':
    unittest.main()
