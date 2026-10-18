"""macOS integration: App Nap and thread priority, global hotkeys, sounds, and the preview window's events."""
from __future__ import annotations

import ctypes
import subprocess
import threading

_activity = None  # must stay referenced, or macOS ends the activity


def disable_app_nap() -> bool:
    """Keep full timer precision and priority when the preview window isn't focused.

    macOS puts unfocused apps into App Nap: timers get coalesced and the process is
    deprioritized, which makes the 60 Hz output to X-Plane uneven (visible as jitter).
    """
    global _activity
    try:
        from Foundation import NSActivityLatencyCritical, NSActivityUserInitiatedAllowingIdleSystemSleep, NSProcessInfo
    except ImportError:
        return False
    _activity = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
        NSActivityUserInitiatedAllowingIdleSystemSleep | NSActivityLatencyCritical, "Head tracking for X-Plane")
    return True


def set_thread_interactive():
    """Schedule the calling thread like UI work, so it isn't starved by X-Plane's load.

    Threads it creates afterwards (e.g. MediaPipe's inference workers) inherit the class.
    """
    try:
        ctypes.CDLL("/usr/lib/libSystem.dylib").pthread_set_qos_class_self_np(0x21, 0)  # USER_INTERACTIVE
    except (OSError, AttributeError):
        pass


def play_sound(cfg: dict, name: str):
    if cfg.get("sounds"):
        try:
            subprocess.Popen(["afplay", f"/System/Library/Sounds/{name}.aiff"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass


MAC_KEYCODES = {
    **dict(zip("asdfhgzxcv", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29,
    "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "space": 49, "return": 36, "escape": 53, "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "f9": 101, "f10": 109, "f11": 103, "f12": 111, "f13": 105, "f14": 107, "f15": 113,
}


def start_global_hotkeys(bindings: dict[str, str], actions: dict) -> str:
    """Listen-only Quartz event tap so hotkeys work while X-Plane has focus.

    Needs Input Monitoring permission for the terminal app. Returns a status string.
    """
    try:
        import Quartz
    except ImportError:
        return "global hotkeys unavailable (pyobjc-framework-Quartz not installed)"

    mods = {
        "ctrl": Quartz.kCGEventFlagMaskControl,
        "shift": Quartz.kCGEventFlagMaskShift,
        "alt": Quartz.kCGEventFlagMaskAlternate,
        "opt": Quartz.kCGEventFlagMaskAlternate,
        "cmd": Quartz.kCGEventFlagMaskCommand,
    }
    all_mods = 0
    for m in mods.values():
        all_mods |= m

    table = {}
    for action, combo in bindings.items():
        if not combo or action not in actions:
            continue
        *mod_names, key = [s.strip().lower() for s in combo.split("+")]
        if key not in MAC_KEYCODES or any(m not in mods for m in mod_names):
            print(f"[hotkeys] can't parse {action}={combo!r}, ignoring")
            continue
        mask = 0
        for m in mod_names:
            mask |= mods[m]
        table[(mask, MAC_KEYCODES[key])] = actions[action]

    holder = {}

    def callback(proxy, etype, event, refcon):
        if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(holder["tap"], True)
        elif etype == Quartz.kCGEventKeyDown:
            if not Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat):
                kc = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                fn = table.get((Quartz.CGEventGetFlags(event) & all_mods, kc))
                if fn:
                    threading.Thread(target=fn, daemon=True).start()
        return event

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap, Quartz.kCGEventTapOptionListenOnly,
        Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown), callback, None)
    if tap is None:
        return ("global hotkeys OFF: allow your terminal app in System Settings > Privacy & Security > "
                "Input Monitoring, then restart")
    holder["tap"] = tap

    def loop():
        src = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), src, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        Quartz.CFRunLoopRun()

    threading.Thread(target=loop, daemon=True, name="hotkeys").start()
    return "global hotkeys: " + ", ".join(f"{a}={c}" for a, c in bindings.items())


def window_visible(title: str) -> bool:
    """Is any part of this app's window with this title showing on screen (not minimized, or
    covered by e.g. a full-screen X-Plane)?"""
    try:
        from AppKit import NSApplication, NSWindowOcclusionStateVisible
    except ImportError:
        return True
    for window in NSApplication.sharedApplication().windows():
        if window.title() == title:
            return bool(window.occlusionState() & NSWindowOcclusionStateVisible)
    return True


def accept_first_click():
    """macOS normally swallows the first click on a window that isn't in front, so clicking a
    button while X-Plane had focus needed a second click. This process only has the preview
    window, so opting all its views in is safe."""
    try:
        import objc
        from AppKit import NSView
    except ImportError:
        return

    def acceptsFirstMouse_(self, _event):
        return True

    objc.classAddMethods(NSView, [objc.selector(acceptsFirstMouse_, selector=b"acceptsFirstMouse:",
                                                signature=b"Z@:@")])


def pump_events() -> int:
    """Handle every waiting window event right away; returns the last key pressed (-1 if none).
    OpenCV's waitKey() handles one event and then sleeps ~10 ms, so mouse drags queued up and
    the sliders trailed behind the mouse until macOS started merging the queued events."""
    from AppKit import NSApplication, NSDefaultRunLoopMode, NSEventMaskAny, NSEventTypeKeyDown

    app, key = NSApplication.sharedApplication(), -1
    while True:
        event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(NSEventMaskAny, None, NSDefaultRunLoopMode, True)
        if event is None:
            break
        if event.type() == NSEventTypeKeyDown and event.characters():
            key = ord(event.characters()[0])  # handled by us, like waitKey() did (no system beep)
            continue
        app.sendEvent_(event)
    app.updateWindows()
    return key
