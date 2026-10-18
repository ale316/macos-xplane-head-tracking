"""macOS integration: scheduling, camera frame-rate lock, global hotkeys, sounds."""
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
        from Foundation import (NSActivityLatencyCritical, NSActivityUserInitiatedAllowingIdleSystemSleep,
                                NSProcessInfo)
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


def lock_camera_fps(index: int, width: int, height: int, want_fps: float):
    """Pin the camera to its fastest format at this size, at up to `want_fps`.

    Mac cameras otherwise let auto-exposure stretch frames in dim light (30 -> 15 fps, which
    doubles their latency), and OpenCV never picks 60 fps formats on its own. Must be called
    after OpenCV has opened the camera. Returns (device to unlock on exit or None, message).
    """
    try:
        import CoreMedia as CM
        devs = capture_devices()
    except ImportError:
        return None, "camera fps not locked (pyobjc-framework-AVFoundation not installed)"
    if index >= len(devs):
        return None, "camera fps not locked (device not found)"
    dev = devs[index]
    best, best_fps = None, 0.0
    for f in dev.formats():
        dims = CM.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
        top = max((r.maxFrameRate() for r in f.videoSupportedFrameRateRanges()), default=0.0)
        if (dims.width, dims.height) == (width, height) and top > best_fps:
            best, best_fps = f, top
    if best is None:
        return None, f"camera fps not locked (no {width}x{height} format)"
    fps = int(min(want_fps, best_fps))
    ok, _ = dev.lockForConfiguration_(None)
    if not ok:
        return None, "camera fps not locked (camera busy)"
    try:
        if best != dev.activeFormat():
            dev.setActiveFormat_(best)
        frame = CM.CMTimeMake(1, fps)
        dev.setActiveVideoMinFrameDuration_(frame)
        dev.setActiveVideoMaxFrameDuration_(frame)
    except Exception as e:  # out-of-range durations raise; keep the camera's own setting
        dev.unlockForConfiguration()
        return None, f"camera fps not locked ({e})"
    # Stay locked: other apps (or the session) can't renegotiate the frame rate meanwhile.
    return dev, f"{dev.localizedName()} locked at {fps} fps"


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


def app_focused() -> bool:
    try:
        from AppKit import NSApplication
    except ImportError:
        return True
    return bool(NSApplication.sharedApplication().isActive())


def capture_devices() -> list:
    """Cameras, in the order OpenCV's AVFoundation backend numbers them."""
    import AVFoundation as AV

    return (list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
            + list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed)))


def camera_names() -> list[str]:
    try:
        return [str(d.localizedName()) for d in capture_devices()]
    except ImportError:
        return []


def pick_camera(cfg: dict, names: list[str]) -> int:
    """The saved camera by name if it's connected (numbers shift as cameras come and go)."""
    if cfg.get("camera_name") in names:
        return names.index(cfg["camera_name"])
    index = int(cfg["camera_index"])
    return index if index < len(names) or not names else 0


def list_cameras():
    for i, name in enumerate(camera_names()):
        print(f"{i}: {name}")
