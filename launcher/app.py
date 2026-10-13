"""create3-laya launcher: start the local models and a simulator, then tell the robot what to do.

  Launcher.cmd                                        (double-click; no console window)
  .venv-laya\\Scripts\\pythonw.exe launcher\\app.py
  .venv-laya\\Scripts\\python.exe launcher\\app.py --sim 2d --start --prompt "go to the east hall"
  .venv-laya\\Scripts\\python.exe launcher\\app.py --theme dark
"""

import argparse
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as be  # noqa: E402
import theme  # noqa: E402

MAZE_EXAMPLES = [
    'go to the east hall, then come back to the dock',
    'go to the north west room',
    'drive forward 1 meter, then turn left 90 degrees',
    'explore two rooms',
    'go to the south room slowly',
    'when you are done, go to the west room',
    'go to the east room instead',
    'stop',
    'pick up the red cup',
]
DEPOT_EXAMPLES = [
    'go 2 meters forward',
    'go to where the pallets are stored',
    'find the forklift',
    'come back to where you started',
    'go back to the pallets',
    'turn right 90 degrees',
    'return to the dock',
    'stop',
]
SIMS = {
    '2d': ('2D maze', "create3_sim's maze in a 2D simulator, drawn live. No WSL needed.",
           MAZE_EXAMPLES),
    'create3': ('Create 3 in Gazebo', 'The same maze in Gazebo, through WSL.', MAZE_EXAMPLES),
    'tb4': ('TurtleBot 4 in Gazebo', 'A depot with no place names: camera, lidar, SLAM, Nav2.',
            DEPOT_EXAMPLES),
}
STATE_COLORS = {'ready': '#2e9d4f', 'external': '#2e9d4f', 'starting': '#d99a00',
                'failed': '#c62828', 'stopped': '#9e9e9e', 'unknown': '#9e9e9e'}
SIM_STATES = {'stopped': 'Not running.', 'waiting': 'Waiting for the models…',
              'starting': 'Starting…', 'ready': 'Running. Type a task below.',
              'stopping': 'Stopping…'}
TIPS = ('Enter sends   ·   Up/Down: earlier prompts   ·   Esc clears   ·   '
        '"Stop robot" halts at once')
DRIVERS = ['code rules', 'fine-tuned Laya']
SPEEDS = ['1x', '2x', '4x']
HAS_DRIVING_MODEL = (be.ROOT / 'models' / 'create3-driving' / 'model.safetensors').exists()
SETTINGS = be.ROOT / 'runs' / 'launcher.json'


class Launcher:
    def __init__(self, root, opts):
        self.root, self.opts = root, opts
        self.inbox = queue.Queue()          # (kind, data) from worker threads to the Tk thread
        self.services = {'laya': be.laya_service(), 'vision': be.vision_service()}
        self.wsl = (None, 'checking WSL…')
        self.backend = None                 # LiveSim2D | GazeboSession
        self.sim_kind = None
        self.sim_state = 'stopped'
        self.pending = None                 # (kind, needs, deadline) while models start
        self.prompts_on_ready = list(opts.prompt or [])
        self.history, self.history_pos = [], 0
        self.scene = None
        self.tf = None
        self.sightings = {}
        self.beat, self.speed = None, None      # last heartbeat (wall, sim t); Gazebo speed
        self.last_replay = None
        self.closing = False
        self.theme_choice = opts.theme or theme.load_choice(SETTINGS)
        self.dark = theme.is_dark(self.theme_choice)
        self.p = theme.apply_ttk(root, self.dark)
        self._build()
        self._apply_theme(force=True)
        self.sim_var.set(opts.sim)
        self.mock_var.set(opts.mock)
        self.driver_var.set(DRIVERS[1] if opts.driver == 'laya' and HAS_DRIVING_MODEL
                            else DRIVERS[0])
        self.speed_var.set(f'{opts.speed:g}x' if f'{opts.speed:g}x' in SPEEDS else SPEEDS[0])
        self._on_sim_choice()
        threading.Thread(target=self._poll_services, daemon=True).start()
        threading.Thread(target=lambda: self.inbox.put(('wsl', be.wsl_status())),
                         daemon=True).start()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        root.after(50, self._pump)
        root.after(100, self._redraw)
        root.after(3000, self._follow_system_theme)
        if opts.start:
            root.after(400, self._auto_start)

    # -- layout --------------------------------------------------------------------------

    def _build(self):
        root = self.root
        root.title('create3-laya')
        root.geometry('1240x820')
        root.minsize(1000, 660)

        models = ttk.LabelFrame(root, text='Local models', padding=(10, 4, 10, 6))
        models.pack(fill='x', padx=10, pady=(10, 4))
        self.service_rows = {}
        for col, (key, svc) in enumerate(self.services.items()):
            cell = ttk.Frame(models)
            cell.grid(row=0, column=col, sticky='w', padx=(0, 40))
            dot = tk.Canvas(cell, width=14, height=14, highlightthickness=0, bg=self.p['bg'])
            dot.grid(row=0, column=0, padx=(0, 6))
            oval = dot.create_oval(2, 2, 12, 12, fill='#9e9e9e', outline='')
            ttk.Label(cell, text=svc.name, style='Head.TLabel').grid(row=0, column=1)
            text = tk.StringVar(value='checking…')
            ttk.Label(cell, textvariable=text, width=40, style='Hint.TLabel').grid(
                row=0, column=2, padx=(8, 8))
            start = ttk.Button(cell, text='Start', width=7,
                               command=lambda s=svc: self._start_service(s))
            stop = ttk.Button(cell, text='Stop', width=7,
                              command=lambda s=svc: self._stop_service(s))
            start.grid(row=0, column=3)
            stop.grid(row=0, column=4, padx=(4, 0))
            self.service_rows[key] = (dot, oval, text, start, stop)
        ttk.Label(models, text='The simulators start what they need.',
                  style='Hint.TLabel').grid(row=0, column=2, sticky='e')
        models.columnconfigure(2, weight=1)

        # Packed before the body so a small window squeezes the body, not the prompt.
        self.status_var = tk.StringVar(value='Pick a simulator and press Start.')
        status = ttk.Frame(root, padding=(12, 0, 10, 6))
        status.pack(side='bottom', fill='x')
        ttk.Label(status, textvariable=self.status_var, style='Hint.TLabel').pack(
            side='left', fill='x', expand=True)
        self.theme_var = tk.StringVar(value=self.theme_choice)
        self.theme_box = ttk.Combobox(status, textvariable=self.theme_var, state='readonly',
                                      values=theme.CHOICES, width=8)
        self.theme_box.pack(side='right')
        self.theme_box.bind('<<ComboboxSelected>>', lambda e: self._pick_theme())
        ttk.Label(status, text='Theme', style='Hint.TLabel').pack(side='right', padx=(0, 6))
        prompt = ttk.Frame(root, padding=(10, 4, 10, 4))
        prompt.pack(side='bottom', fill='x')
        self.prompt_var = tk.StringVar()
        self.entry = ttk.Entry(prompt, textvariable=self.prompt_var, font=('Segoe UI', 12))
        self.entry.pack(side='left', fill='x', expand=True, ipady=4)
        self.entry.bind('<Return>', lambda e: self.send())
        self.entry.bind('<Up>', lambda e: self._history(-1))
        self.entry.bind('<Down>', lambda e: self._history(1))
        self.entry.bind('<Escape>', lambda e: self.prompt_var.set(''))
        self.send_button = ttk.Button(prompt, text='Send', style='Go.TButton', command=self.send)
        self.send_button.pack(side='left', padx=(8, 0), ipady=3)
        self.halt_button = ttk.Button(prompt, text='Stop robot',
                                      command=lambda: self.send('stop'))
        self.halt_button.pack(side='left', padx=(6, 0), ipady=3)

        body = ttk.PanedWindow(root, orient='horizontal')
        body.pack(fill='both', expand=True, padx=10, pady=4)
        left = ttk.Frame(body, width=330)
        body.add(left, weight=0)
        right = ttk.Notebook(body)
        body.add(right, weight=1)

        sim = ttk.LabelFrame(left, text='Simulator', padding=(10, 6))
        sim.pack(fill='x')
        self.sim_var = tk.StringVar(value='2d')
        self.sim_radios = []
        for key, (label, hint, _) in SIMS.items():
            r = ttk.Radiobutton(sim, text=label, value=key, variable=self.sim_var,
                                command=self._on_sim_choice)
            r.pack(anchor='w', pady=(4, 0))
            ttk.Label(sim, text=hint, style='Hint.TLabel', wraplength=290).pack(
                anchor='w', padx=(22, 0))
            self.sim_radios.append(r)
        opts = ttk.Frame(sim)
        opts.pack(fill='x', pady=(10, 4))
        ttk.Label(opts, text='Driving').grid(row=0, column=0, sticky='w')
        self.driver_var = tk.StringVar(value=DRIVERS[0])
        self.driver_box = ttk.Combobox(opts, textvariable=self.driver_var, state='readonly',
                                       values=DRIVERS if HAS_DRIVING_MODEL else DRIVERS[:1],
                                       width=18)
        self.driver_box.grid(row=0, column=1, sticky='w', padx=(8, 0))
        ttk.Label(opts, text='Speed').grid(row=1, column=0, sticky='w', pady=(4, 0))
        self.speed_var = tk.StringVar(value=SPEEDS[0])
        self.speed_box = ttk.Combobox(opts, textvariable=self.speed_var, state='readonly',
                                      values=SPEEDS, width=6)
        self.speed_box.grid(row=1, column=1, sticky='w', padx=(8, 0), pady=(4, 0))
        self.speed_box.bind('<<ComboboxSelected>>', lambda e: self._apply_speed())
        self.mock_var = tk.BooleanVar(value=False)
        self.mock_check = ttk.Checkbutton(opts, text='Offline: keyword stand-in, not Laya',
                                          variable=self.mock_var)
        self.mock_check.grid(row=2, column=0, columnspan=2, sticky='w', pady=(6, 0))
        self.gzgui_var = tk.BooleanVar(value=True)
        self.gzgui_check = ttk.Checkbutton(opts, text='Show the Gazebo window',
                                           variable=self.gzgui_var)
        self.gzgui_check.grid(row=3, column=0, columnspan=2, sticky='w', pady=(2, 0))
        self.wsl_var = tk.StringVar(value='')
        ttk.Label(sim, textvariable=self.wsl_var, style='Hint.TLabel', wraplength=290).pack(
            anchor='w', pady=(2, 6))
        buttons = ttk.Frame(sim)
        buttons.pack(fill='x')
        self.start_button = ttk.Button(buttons, text='Start', style='Go.TButton',
                                       command=self.start_sim)
        self.start_button.pack(side='left', fill='x', expand=True)
        self.stop_button = ttk.Button(buttons, text='Stop', command=self.stop_sim)
        self.stop_button.pack(side='left', fill='x', expand=True, padx=(6, 0))
        self.sim_status = tk.StringVar(value=SIM_STATES['stopped'])
        ttk.Label(sim, textvariable=self.sim_status, wraplength=290).pack(anchor='w',
                                                                          pady=(6, 0))
        self.replay_button = ttk.Button(sim, text='Open the last replay',
                                        command=self._open_replay)

        examples = ttk.LabelFrame(left, text='Try saying', padding=(10, 6))
        examples.pack(fill='both', expand=True, pady=(8, 0))
        self.examples = tk.Listbox(examples, activestyle='none', borderwidth=0,
                                   highlightthickness=0, font=('Segoe UI', 10))
        self.examples.pack(fill='both', expand=True)
        self.examples.bind('<<ListboxSelect>>', self._example_picked)
        self.examples.bind('<Double-Button-1>', lambda e: self.send())
        ttk.Label(examples, text='Click to edit, double-click to send.',
                  style='Hint.TLabel').pack(anchor='w', pady=(4, 0))

        robot_tab = ttk.PanedWindow(right, orient='vertical')
        right.add(robot_tab, text='  Robot  ')
        self.canvas = tk.Canvas(robot_tab, highlightthickness=0, height=430)
        robot_tab.add(self.canvas, weight=3)
        self.canvas.bind('<Configure>', lambda e: self._draw_static())
        talk = ttk.Frame(robot_tab)
        robot_tab.add(talk, weight=2)
        self.talk = tk.Text(talk, wrap='word', font=('Segoe UI', 10), borderwidth=0, height=9,
                            padx=10, pady=8, state='disabled')
        bar = ttk.Scrollbar(talk, command=self.talk.yview)
        self.talk.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        self.talk.pack(fill='both', expand=True)
        self.talk.tag_configure('you', font=('Segoe UI Semibold', 10), spacing1=8)
        self.talk.tag_configure('info', font=('Segoe UI', 9, 'italic'), spacing1=4)

        logs = ttk.Frame(right)
        right.add(logs, text='  Logs  ')
        self.logs = tk.Text(logs, wrap='none', font=('Consolas', 9), state='disabled',
                            borderwidth=0, padx=6, pady=4)
        lbar = ttk.Scrollbar(logs, command=self.logs.yview)
        self.logs.configure(yscrollcommand=lbar.set)
        lbar.pack(side='right', fill='y')
        self.logs.pack(fill='both', expand=True)

        self.say('Pick a simulator and press Start. The launcher starts the local models it '
                 'needs, then you can tell the robot what to do in plain English.', 'info')

    # -- look ------------------------------------------------------------------------------

    def _apply_theme(self, force=False):
        dark = theme.is_dark(self.theme_choice)
        if dark == self.dark and not force:
            return
        self.dark = dark
        self.p = p = theme.apply_ttk(self.root, dark)
        for dot, *_ in self.service_rows.values():
            dot.configure(bg=p['bg'])
        self.examples.configure(bg=p['field'], fg=p['fg'], selectbackground=p['select_bg'],
                                selectforeground=p['select_fg'])
        self.canvas.configure(bg=p['canvas'])
        self.talk.configure(bg=p['field'], fg=p['robot_text'], insertbackground=p['fg'],
                            selectbackground=p['select_bg'])
        for tag in ('you', 'step', 'done', 'ask', 'refuse', 'problem', 'info'):
            self.talk.tag_configure(tag, foreground=p[tag])
        self.talk.tag_configure('robot', foreground=p['robot_text'])
        self.logs.configure(bg=p['log_bg'], fg=p['log_fg'], insertbackground=p['log_fg'])
        theme.style_popdowns(self.root, (self.driver_box, self.speed_box, self.theme_box), p)
        theme.title_bar(self.root, dark)
        self._draw_static()

    def _pick_theme(self):
        self.theme_choice = self.theme_var.get()
        theme.save_choice(SETTINGS, self.theme_choice)
        self._apply_theme()
        self.entry.focus_set()

    def _follow_system_theme(self):
        if self.theme_choice == 'System':
            self._apply_theme()
        self.root.after(3000, self._follow_system_theme)

    # -- services ------------------------------------------------------------------------

    def _poll_services(self):
        while not self.closing:
            changed = False
            for svc in self.services.values():
                changed |= svc.poll()
            if changed:
                self.inbox.put(('service', None))
            time.sleep(1.5)

    def _refresh_services(self):
        for key, svc in self.services.items():
            dot, oval, text, start, stop = self.service_rows[key]
            dot.itemconfigure(oval, fill=STATE_COLORS.get(svc.state, '#9e9e9e'))
            label = {'ready': 'ready', 'external': 'running', 'starting': 'starting',
                     'failed': 'failed', 'stopped': 'stopped'}.get(svc.state, 'checking…')
            text.set(f'{label} · {svc.detail}' if svc.detail else label)
            start.state(['!disabled'] if svc.state in ('stopped', 'failed') else ['disabled'])
            stop.state(['!disabled'] if svc.state in ('ready', 'external', 'starting')
                       else ['disabled'])

    def _start_service(self, svc):
        svc.start(self._log_sink)
        self.say(f'Starting the {svc.name} server (it loads its model; up to a minute).', 'info')
        self._refresh_services()

    def _stop_service(self, svc):
        reasons = []
        if svc.state == 'external':
            reasons.append(f'The {svc.name} server was started outside this launcher.')
        if self.sim_state != 'stopped' and (svc.name == 'Laya' or self.sim_kind == 'tb4'):
            reasons.append('The running simulator uses it.')
        if reasons and not messagebox.askyesno('Stop server',
                                               ' '.join(reasons) + ' Stop it anyway?'):
            return
        threading.Thread(target=svc.stop, daemon=True).start()
        self.say(f'Stopping the {svc.name} server.', 'info')

    # -- simulator -----------------------------------------------------------------------

    def _on_sim_choice(self):
        kind = self.sim_var.get()
        self.examples.delete(0, 'end')
        for text in SIMS[kind][2]:
            self.examples.insert('end', text)
        self._refresh_options()

    def _refresh_options(self):
        kind, idle = self.sim_var.get(), self.sim_state == 'stopped'
        for r in self.sim_radios:
            r.state(['!disabled'] if idle else ['disabled'])
        self.mock_check.state(['!disabled'] if idle and kind == '2d' else ['disabled'])
        self.gzgui_check.state(['!disabled'] if idle and kind == 'tb4' else ['disabled'])
        # ttk keeps 'disabled' until it is cleared explicitly.
        self.speed_box.state(['!disabled', 'readonly'])
        self.driver_box.state(['!disabled', 'readonly'] if idle else ['disabled'])
        self.start_button.state(['!disabled'] if idle else ['disabled'])
        self.stop_button.state(['disabled'] if idle else ['!disabled'])
        ready = self.sim_state == 'ready'
        for w in (self.send_button, self.halt_button):
            w.state(['!disabled'] if ready else ['disabled'])
        ok, msg = self.wsl
        self.wsl_var.set('' if kind == '2d' else
                         f"WSL: {msg}" if ok is not False else f'Not available: {msg}')

    def _set_sim_state(self, state, note=None):
        self.sim_state = state
        self.sim_status.set(note or SIM_STATES[state])
        self.status_var.set(TIPS if state == 'ready' else
                            'Pick a simulator and press Start.' if state == 'stopped' else '')
        self._refresh_options()
        if not self.scene:
            self._draw_static()

    def _auto_start(self):
        """--start: a Gazebo simulator waits for the WSL check (a few seconds) to finish."""
        if self.sim_var.get() != '2d' and self.wsl[0] is None:
            self.root.after(500, self._auto_start)
            return
        self.start_sim()

    def start_sim(self):
        if self.sim_state != 'stopped':
            return
        kind = self.sim_var.get()
        mock = kind == '2d' and self.mock_var.get()
        if kind != '2d':
            ok, msg = self.wsl
            if ok is None:
                messagebox.showinfo('Gazebo', 'Still checking WSL. Try again in a moment.')
                return
            if not ok:
                messagebox.showerror('Gazebo unavailable', msg)
                return
        needs = ([] if mock else ['laya']) + (['vision'] if kind == 'tb4' else [])
        for key in needs:
            svc = self.services[key]
            if not svc.available and not svc.owned:
                self._start_service(svc)
        self.pending = (kind, needs, time.time() + 300)
        self._set_sim_state('waiting')
        self._check_pending()

    def _check_pending(self):
        if self.pending is None:
            return
        kind, needs, deadline = self.pending
        missing = [self.services[k] for k in needs if not self.services[k].available]
        if not missing:
            self.pending = None
            if 'laya' not in needs:
                self._launch(kind)
                return
            models = ['typed-decisions'] + (['create3-driving'] if self._driver() == 'laya'
                                            else [])
            self.sim_status.set('Warming up Laya…')

            def work():
                try:
                    self.inbox.put(('warmed', (kind, None, be.warm_up(models))))
                except be.LayaError as e:
                    self.inbox.put(('warmed', (kind, str(e), 0.0)))
            threading.Thread(target=work, daemon=True).start()
            return
        if any(s.state == 'failed' for s in missing) or time.time() > deadline:
            self.pending = None
            self._set_sim_state('stopped')
            self.say(' and '.join(s.name for s in missing) + ' did not start. The Logs tab '
                     'has its output.', 'problem')
            return
        self.sim_status.set('Waiting for ' + ' and '.join(s.name for s in missing)
                            + ' to load…')
        self.root.after(500, self._check_pending)

    def _driver(self):
        return 'laya' if self.driver_var.get() == DRIVERS[1] else 'code'

    def _after_warm_up(self, kind, error, seconds):
        if self.sim_state != 'waiting':          # stopped while warming up
            return
        if error:
            self._set_sim_state('stopped')
            self.say(f'Laya did not answer: {error}', 'problem')
            return
        if seconds > 3:
            self.say(f'Laya took {seconds:.0f} s to wake up (its weights had been moved out of '
                     'GPU memory); it answers in a fraction of a second now.', 'info')
        self._launch(kind)

    def _launch(self, kind):
        driver = self._driver()
        self.sim_kind, self.sightings = kind, {}
        self.beat, self.speed = None, None
        self.say(f'Starting: {SIMS[kind][0]}'
                 + (', fine-tuned Laya driving' if driver == 'laya' else '') + '.', 'info')
        if kind == '2d':
            try:
                self.backend = be.LiveSim2D(self._event_sink, driver=driver,
                                            mock=self.mock_var.get(), speed=self._speed())
            except Exception as e:      # noqa: BLE001 -- show any setup failure in the window
                self.say(f'The 2D simulator could not start: {e}', 'problem')
                self._set_sim_state('stopped')
                return
            self.scene = self.backend.scene()
            self._set_sim_state('starting')
            self.backend.start()
        else:
            self.scene = None
            gui = self.gzgui_var.get() or kind == 'create3'
            try:
                self.backend = be.GazeboSession(kind, self._event_sink, self._log_sink,
                                                driver=driver, gui=gui)
            except OSError as e:
                self.say(f'WSL could not be started: {e}', 'problem')
                self._set_sim_state('stopped')
                return
            self._set_sim_state('starting', 'Starting Gazebo in WSL (about a minute)…')
            self.say('Gazebo is starting in WSL. Its window opens on its own'
                     + ('' if gui else ' (hidden for speed)') + '; progress is in the Logs tab.',
                     'info')
        self._draw_static()

    def stop_sim(self):
        if self.sim_state == 'waiting':          # models still loading, or Laya warming up
            self.pending = None
            self._set_sim_state('stopped')
            return
        if self.backend is None or self.sim_state == 'stopping':
            return
        backend = self.backend
        self._set_sim_state('stopping')

        def work():
            result = backend.stop()
            self.inbox.put(('stopped', (backend, result)))
        threading.Thread(target=work, daemon=True).start()

    def _after_stop(self, backend, result):
        if backend is self.backend:
            self.backend = None
        if isinstance(backend, be.LiveSim2D) and result:
            self.last_replay = result
            self.replay_button.pack(fill='x', pady=(6, 0))
            self.say(f'Simulator stopped. Replay saved to {result}.', 'info')
        else:
            self.say('Simulator stopped.', 'info')
        self._set_sim_state('stopped')

    def _open_replay(self):
        if self.last_replay:
            os.startfile(self.last_replay)

    def _speed(self):
        return float(self.speed_var.get().rstrip('x'))

    def _apply_speed(self):
        if isinstance(self.backend, be.LiveSim2D):
            self.backend.set_speed(self._speed())
        elif isinstance(self.backend, be.GazeboSession) and self.sim_state == 'ready':
            backend, factor = self.backend, self._speed()

            def work():
                ok, message = backend.set_speed(factor)
                self.inbox.put(('speed', (ok, message, factor)))
            threading.Thread(target=work, daemon=True).start()

    def _after_speed(self, ok, message, factor):
        if ok and factor > 1:
            self.say(f'Gazebo now runs up to {factor:g}x real time; the status line shows '
                     'what it reaches (about 2x with its window open).', 'info')
        elif not ok:
            self.say(f'Could not change the Gazebo speed: {message}', 'problem')

    # -- talking to the robot --------------------------------------------------------------

    def send(self, text=None):
        text = (text if text is not None else self.prompt_var.get()).strip()
        if not text:
            return
        if self.sim_state != 'ready' or self.backend is None:
            self.root.bell()
            self.status_var.set('Start a simulator first. The prompt works once it is ready.')
            return
        self.say(f'You: {text}', 'you')
        if not self.history or self.history[-1] != text:
            self.history.append(text)
        self.history_pos = len(self.history)
        self.prompt_var.set('')
        self.status_var.set(f'Laya is reading "{text}"…')
        try:
            self.backend.send(text)
        except OSError as e:
            self.say(f'Could not reach the simulator: {e}', 'problem')

    def _history(self, delta):
        if not self.history:
            return 'break'
        self.history_pos = max(0, min(len(self.history), self.history_pos + delta))
        self.prompt_var.set(self.history[self.history_pos]
                            if self.history_pos < len(self.history) else '')
        self.entry.icursor('end')
        return 'break'

    def _example_picked(self, _event):
        sel = self.examples.curselection()
        if sel:
            self.prompt_var.set(self.examples.get(sel[0]))
            self.entry.focus_set()
            self.entry.icursor('end')

    def say(self, text, style='robot'):
        self.talk.configure(state='normal')
        self.talk.insert('end', text + '\n', style)
        self.talk.configure(state='disabled')
        self.talk.see('end')

    # -- events from back ends -------------------------------------------------------------

    def _event_sink(self, event):
        self.inbox.put(('event', event))

    def _log_sink(self, name, line):
        self.inbox.put(('log', (name, '[exited]' if line is None else line)))

    def _pump(self):
        try:
            for _ in range(500):
                kind, data = self.inbox.get_nowait()
                if kind == 'log':
                    self._append_log(*data)
                elif kind == 'event':
                    self._on_event(data)
                elif kind == 'service':
                    self._refresh_services()
                elif kind == 'wsl':
                    self.wsl = data
                    self._refresh_options()
                elif kind == 'stopped':
                    self._after_stop(*data)
                elif kind == 'warmed':
                    self._after_warm_up(*data)
                elif kind == 'speed':
                    self._after_speed(*data)
                elif kind == 'quit':
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(50, self._pump)

    def _on_event(self, event):
        kind = event.get('event')
        if kind == 'heartbeat':
            # Sent every 2 s of simulated time, so their spacing gives Gazebo's speed.
            now, t = time.time(), event.get('t')
            if self.beat is not None and None not in (t, self.beat[1]) and t > self.beat[1]:
                self.speed = (t - self.beat[1]) / max(1e-3, now - self.beat[0])
            self.beat = (now, t)
            if self.sim_state == 'ready':
                self.sim_status.set(SIM_STATES['ready'] + (
                    f'\nSimulation speed: {self.speed:.2f}x real time.' if self.speed else ''))
                if not self.scene:              # the Gazebo info text repeats the status
                    self._draw_static()
            return
        if kind == 'session':
            if event.get('status') == 'ready' and self.sim_state == 'starting':
                self._set_sim_state('ready')
                self.beat = (time.time(), None)     # the crawl watchdog counts from here
                self.say('Ready. Tell the robot what to do.', 'info')
                self.entry.focus_set()
                if self._speed() != 1.0:            # a new world starts at real time
                    self._apply_speed()
                for text in self.prompts_on_ready:
                    self.send(text)
                self.prompts_on_ready = []
            elif event.get('status') == 'ended' and self.sim_state not in ('stopping',
                                                                         'stopped'):
                self.backend = None
                self._set_sim_state('stopped')
                if not self.closing:
                    self.say('The simulator exited. The Logs tab has its output.', 'problem')
            return
        if kind in ('task', 'error') and self.sim_state == 'ready':
            self.status_var.set(TIPS)
        if kind == 'seen':
            self.sightings[event.get('label')] = event
            self._draw_static()
        line = be.describe_event(event)
        if line:
            self.say(*line)

    def _append_log(self, name, line):
        self.logs.configure(state='normal')
        self.logs.insert('end', f'[{name}] {line}\n')
        lines = int(self.logs.index('end-1c').split('.')[0])
        if lines > 6000:
            self.logs.delete('1.0', f'{lines - 5000}.0')
        self.logs.configure(state='disabled')
        self.logs.see('end')

    # -- drawing ---------------------------------------------------------------------------

    def _draw_static(self):
        c = self.canvas
        c.delete('all')
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50 or h < 50:
            return
        if not self.scene:
            self.tf = None
            self._draw_info(w, h)
            return
        pts = [p for box in self.scene['boxes'] for p in box]
        minx, maxx = min(p[0] for p in pts), max(p[0] for p in pts)
        miny, maxy = min(p[1] for p in pts), max(p[1] for p in pts)
        m = 18
        s = min((w - 2 * m) / (maxx - minx), (h - 2 * m) / (maxy - miny))
        self.tf = (s, (w - s * (maxx - minx)) / 2 - s * minx,
                   (h + s * (maxy - miny)) / 2 + s * miny)
        for box in self.scene['boxes']:
            c.create_polygon([v for p in box for v in self._px(*p)], fill=self.p['wall'],
                             outline='')
        for name, (x, y) in self.scene['places'].items():
            px, py = self._px(x, y)
            color = self.p['dock'] if name == 'dock' else self.p['place']
            c.create_oval(px - 3, py - 3, px + 3, py + 3, fill=color, outline='')
            c.create_text(px, py + 11, text=name.replace('_', ' '), fill=color,
                          font=('Segoe UI', 8))

    def _draw_info(self, w, h):
        c, fg, muted = self.canvas, self.p['fg'], self.p['muted']
        if self.sim_kind in ('create3', 'tb4') and self.sim_state != 'stopped':
            lines = [(SIMS[self.sim_kind][0], ('Segoe UI Semibold', 14), fg),
                     (self.sim_status.get(), ('Segoe UI', 10), muted),
                     ('The robot is in the Gazebo window.', ('Segoe UI', 10), muted)]
            for label, s in sorted(self.sightings.items()):
                lines.append((f"Seen: {label}, {s.get('distance', 0):.1f} m away at "
                              f"({s.get('x', 0):.1f}, {s.get('y', 0):.1f}), "
                              f"score {s.get('score', 0):.2f}", ('Segoe UI', 10),
                              self.p['seen']))
        else:
            lines = [('No simulator running', ('Segoe UI Semibold', 14), fg),
                     ('The 2D maze is drawn here live; the Gazebo simulators open their own '
                      'window.', ('Segoe UI', 10), muted)]
        y = h / 2 - 14 * len(lines)
        for text, font, color in lines:
            c.create_text(w / 2, y, text=text, font=font, fill=color, width=w - 60)
            y += 30

    def _px(self, x, y):
        s, ox, oy = self.tf
        return ox + s * x, oy - s * y

    def _redraw(self):
        if isinstance(self.backend, be.LiveSim2D) and self.backend.view and self.tf:
            self._draw_robot(self.backend.view)
        elif (isinstance(self.backend, be.GazeboSession) and self.sim_state == 'ready'
              and self.beat is not None and time.time() - self.beat[0] > 15):
            self.sim_status.set(f'Gazebo is crawling: no word from the robot for '
                                f'{time.time() - self.beat[0]:.0f} s. Is another program '
                                'using the GPU?')
        self.root.after(80, self._redraw)

    def _draw_robot(self, v):
        c, s, pal = self.canvas, self.tf[0], self.p
        c.delete('dyn')
        if len(v['trail']) > 1:
            c.create_line([q for p in v['trail'] for q in self._px(*p)], fill=pal['trail'],
                          width=2, tags='dyn')
        if v['route'] and len(v['route']) > 1:
            c.create_line([q for p in v['route'] for q in self._px(*p)], fill=pal['route'],
                          width=2, dash=(5, 3), tags='dyn')
        x, y, yaw = v['x'], v['y'], v['yaw']
        cy, sy = math.cos(yaw), math.sin(yaw)
        for mx, my, myaw, rng in v['ir']:
            ox, oy = x + cy * mx - sy * my, y + sy * mx + cy * my
            length = rng if rng is not None else be.create3.IR_MAX_RANGE
            color = (pal['ir_clear'] if rng is None or rng >= be.create3.IR_MAX_RANGE
                     else pal['ir_close'] if rng < 0.08 else pal['ir_near'])
            c.create_line(*self._px(ox, oy), *self._px(ox + math.cos(yaw + myaw) * length,
                                                        oy + math.sin(yaw + myaw) * length),
                          fill=color, width=2, tags='dyn')
        r = self.scene['robot_radius'] * s
        px, py = self._px(x, y)
        bumped = any(h.startswith('bump') for h in v['hazards'])
        c.create_oval(px - r, py - r, px + r, py + r,
                      fill=pal['robot_bumped'] if bumped else pal['robot'],
                      outline=pal['robot_edge'], width=2, tags='dyn')
        c.create_line(px, py, px + cy * r, py - sy * r, fill=pal['robot_edge'], width=2,
                      tags='dyn')
        doing = v['step'] or 'idle'
        c.create_text(10, 10, anchor='nw', text=f"t = {v['t']:.1f} s    {doing}",
                      font=('Segoe UI', 9), fill=pal['hud'], tags='dyn')

    # -- closing ---------------------------------------------------------------------------

    def on_close(self):
        if self.closing:
            return
        running = [s for s in self.services.values() if s.available or s.owned]
        stop_servers = False
        if running:
            answer = messagebox.askyesnocancel(
                'Quit', 'Stop the model servers too?\n\nYes: stop '
                + ' and '.join(s.name for s in running)
                + '.\nNo: leave them running, so the next start is quick.')
            if answer is None:
                return
            stop_servers = answer
        self.closing = True
        self.status_var.set('Shutting down…')
        if self.backend is not None:
            self._set_sim_state('stopping')
        threading.Thread(target=self._shutdown, args=(stop_servers, running),
                         daemon=True).start()

    def _shutdown(self, stop_servers, running):
        if self.backend is not None:
            try:
                self.backend.stop()
            except Exception:           # noqa: BLE001 -- quitting regardless
                pass
        if stop_servers:
            for svc in running:
                svc.stop()
        self.inbox.put(('quit', None))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--sim', choices=list(SIMS), default='2d')
    p.add_argument('--mock', action='store_true', help='2D only: no Laya server')
    p.add_argument('--driver', choices=['code', 'laya'], default='code')
    p.add_argument('--speed', type=float, default=1.0,
                   help='1, 2 or 4 x real time (Gazebo: as fast as it can, up to that)')
    p.add_argument('--start', action='store_true', help='start the simulator right away')
    p.add_argument('--prompt', action='append', help='send once ready (repeatable)')
    p.add_argument('--theme', choices=theme.CHOICES, type=str.capitalize,
                   help='System (follows Windows), Light or Dark; remembered when picked in '
                        'the window')
    opts = p.parse_args()
    try:                                # crisp text on high-DPI screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    Launcher(root, opts)
    root.mainloop()


if __name__ == '__main__':
    main()
