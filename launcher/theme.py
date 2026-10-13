"""Light and dark looks for the launcher.

Light keeps ttk's native Windows theme ('vista'), which can't be recoloured; dark is built on
'clam', which can. Colours that aren't ttk's (canvas, chat, lists) come from the palettes.
"""

import json
import sys
from tkinter import ttk

CHOICES = ('System', 'Light', 'Dark')

# Keys shared by both palettes; 'bg' None means "whatever the native theme uses".
LIGHT = {
    'bg': None, 'field': '#ffffff', 'canvas': '#fbfbfa', 'fg': '#202124', 'muted': '#5f6368',
    'select_bg': '#dbe7fb', 'select_fg': '#111111',
    # the robot view
    'wall': '#4b5563', 'place': '#7c8591', 'dock': '#0f766e', 'trail': '#a8c7fa',
    'route': '#1a73e8', 'ir_clear': '#d1d5db', 'ir_near': '#f59e0b', 'ir_close': '#ef4444',
    'robot': '#e8eaed', 'robot_bumped': '#fde68a', 'robot_edge': '#202124', 'hud': '#3c4043',
    'seen': '#1e7e34',
    # the conversation
    'you': '#1a4fb5', 'robot_text': '#202124', 'step': '#5f6368', 'done': '#1e7e34',
    'ask': '#9a5b00', 'refuse': '#a61b1b', 'problem': '#c62828', 'info': '#5f6368',
    # the logs stay a dark console in both looks
    'log_bg': '#1e1e1e', 'log_fg': '#d4d4d4',
}
DARK = {
    'bg': '#202124', 'field': '#2b2c30', 'canvas': '#17181a', 'fg': '#e8eaed',
    'muted': '#9aa0a6', 'select_bg': '#34476b', 'select_fg': '#ffffff',
    'button': '#303134', 'button_hover': '#3c4043', 'button_pressed': '#4a4d52',
    'border': '#3c4043', 'disabled': '#6b6f75', 'accent': '#8ab4f8',
    'wall': '#596170', 'place': '#9aa0a6', 'dock': '#2dd4bf', 'trail': '#35507d',
    'route': '#8ab4f8', 'ir_clear': '#3c4043', 'ir_near': '#fbbf24', 'ir_close': '#f87171',
    'robot': '#3c4043', 'robot_bumped': '#8a6d1a', 'robot_edge': '#e8eaed', 'hud': '#bdc1c6',
    'seen': '#81c995',
    'you': '#8ab4f8', 'robot_text': '#e8eaed', 'step': '#9aa0a6', 'done': '#81c995',
    'ask': '#fdd663', 'refuse': '#f28b82', 'problem': '#f28b82', 'info': '#9aa0a6',
    'log_bg': '#141414', 'log_fg': '#d4d4d4',
}


def system_prefers_dark():
    """Windows' "choose your app mode" setting; False elsewhere or if it can't be read."""
    if sys.platform != 'win32':
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize') as k:
            return winreg.QueryValueEx(k, 'AppsUseLightTheme')[0] == 0
    except OSError:
        return False


def is_dark(choice):
    return choice == 'Dark' or (choice == 'System' and system_prefers_dark())


def load_choice(path):
    try:
        choice = json.loads(path.read_text(encoding='utf-8')).get('theme')
    except (OSError, ValueError, AttributeError):
        return 'System'
    return choice if choice in CHOICES else 'System'


def save_choice(path, choice):
    try:
        data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    except (OSError, ValueError):
        data = {}
    data['theme'] = choice
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    except OSError:
        pass                            # a read-only folder just means it isn't remembered


def apply_ttk(root, dark):
    """Switch ttk to the look and (re)define the launcher's own styles. Returns the palette,
    with 'bg' filled in."""
    style = ttk.Style(root)
    if not dark:
        style.theme_use('vista' if 'vista' in style.theme_names() else 'default')
        p = dict(LIGHT, bg=style.lookup('TFrame', 'background') or '#f0f0f0')
        style.configure('Head.TLabel', font=('Segoe UI Semibold', 10))
        style.configure('Hint.TLabel', foreground=p['muted'])
        style.configure('Go.TButton', font=('Segoe UI Semibold', 10))
        root.configure(bg=p['bg'])
        return p
    p = dict(DARK)
    style.theme_use('clam')
    bg, fg, field, edge = p['bg'], p['fg'], p['field'], p['border']
    style.configure('.', background=bg, foreground=fg, fieldbackground=field,
                    bordercolor=edge, lightcolor=bg, darkcolor=bg, troughcolor=field,
                    focuscolor=p['accent'], selectbackground=p['select_bg'],
                    selectforeground=p['select_fg'], insertcolor=fg, arrowcolor=fg,
                    font=('Segoe UI', 9))
    style.map('.', foreground=[('disabled', p['disabled'])])
    style.configure('TButton', background=p['button'], bordercolor=edge,
                    lightcolor=p['button'], darkcolor=p['button'], padding=(8, 3))
    style.map('TButton',
              background=[('disabled', bg), ('pressed', p['button_pressed']),
                          ('active', p['button_hover'])],
              lightcolor=[('pressed', p['button_pressed']), ('active', p['button_hover'])],
              darkcolor=[('pressed', p['button_pressed']), ('active', p['button_hover'])],
              bordercolor=[('focus', p['accent'])])
    for check in ('TCheckbutton', 'TRadiobutton'):
        style.configure(check, indicatorbackground=field, indicatorforeground=fg,
                        indicatormargin=(0, 1, 6, 1))
        style.map(check, background=[('active', bg)],
                  indicatorbackground=[('disabled', bg), ('pressed', p['button_hover']),
                                       ('selected', field)],
                  indicatorforeground=[('disabled', p['disabled']), ('selected', p['accent'])])
    style.configure('TEntry', fieldbackground=field, bordercolor=edge, lightcolor=field,
                    darkcolor=field, padding=3)
    style.map('TEntry', bordercolor=[('focus', p['accent'])],
              lightcolor=[('focus', p['accent'])])
    style.configure('TCombobox', fieldbackground=field, background=p['button'],
                    bordercolor=edge, lightcolor=field, darkcolor=field, arrowsize=12)
    style.map('TCombobox',
              fieldbackground=[('readonly', field), ('disabled', bg)],
              foreground=[('disabled', p['disabled'])],
              background=[('active', p['button_hover']), ('disabled', bg)],
              selectbackground=[('readonly', field)], selectforeground=[('readonly', fg)])
    style.configure('TLabelframe', background=bg, bordercolor=edge, lightcolor=bg,
                    darkcolor=bg)
    style.configure('TLabelframe.Label', background=bg, foreground=p['muted'])
    style.configure('TNotebook', background=bg, bordercolor=edge, lightcolor=bg, darkcolor=bg,
                    tabmargins=(2, 4, 2, 0))
    style.configure('TNotebook.Tab', background=bg, foreground=p['muted'], bordercolor=edge,
                    lightcolor=bg, darkcolor=bg, padding=(10, 4))
    style.map('TNotebook.Tab', background=[('selected', p['button'])],
              foreground=[('selected', fg)], lightcolor=[('selected', p['button'])])
    style.configure('TScrollbar', background=p['button'], troughcolor=bg, bordercolor=bg,
                    lightcolor=p['button'], darkcolor=p['button'], arrowcolor=p['muted'])
    style.map('TScrollbar', background=[('active', p['button_hover'])])
    style.configure('TPanedwindow', background=bg)
    style.configure('Sash', sashthickness=6, gripcount=0, background=bg, lightcolor=bg,
                    darkcolor=bg)
    style.configure('Head.TLabel', font=('Segoe UI Semibold', 10))
    style.configure('Hint.TLabel', foreground=p['muted'])
    style.configure('Go.TButton', font=('Segoe UI Semibold', 10))
    root.configure(bg=bg)
    return p


def style_popdowns(root, comboboxes, p):
    """A combobox's drop-down list is a plain Tk listbox; recolour the ones that exist and set
    the defaults for those not opened yet."""
    for key, value in (('background', p['field']), ('foreground', p['fg']),
                       ('selectBackground', p['select_bg']),
                       ('selectForeground', p['select_fg'])):
        root.option_add(f'*TCombobox*Listbox.{key}', value)
    for box in comboboxes:
        try:
            popdown = root.tk.call('ttk::combobox::PopdownWindow', box)
            root.tk.call(f'{popdown}.f.l', 'configure', '-background', p['field'],
                         '-foreground', p['fg'], '-selectbackground', p['select_bg'],
                         '-selectforeground', p['select_fg'])
        except Exception:               # noqa: BLE001 -- cosmetic only
            pass


def title_bar(root, dark):
    """Dark or light window frame on Windows 10 (20H1+) and 11; does nothing elsewhere."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        for attribute in (20, 19):      # DWMWA_USE_IMMERSIVE_DARK_MODE, and its pre-20H1 id
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
        # Repaint the frame now rather than on the next focus change.
        flags = 0x0001 | 0x0002 | 0x0004 | 0x0020   # NOSIZE | NOMOVE | NOZORDER | FRAMECHANGED
        ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, flags)
    except (AttributeError, OSError):
        pass
