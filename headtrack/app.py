"""Command line entry point: wires the tracker, the X-Plane link and the preview together."""
from __future__ import annotations

import argparse
import multiprocessing
import queue
import threading
import time
from pathlib import Path

from .config import CONFIG_PATH, MODEL_PATH, load_config, save_settings, set_look_range
from .macos import disable_app_nap, list_cameras, play_sound, set_thread_interactive, start_global_hotkeys
from .preview import preview_process, snapshot, status_lines
from .state import Shared, close_diag, diag, enable_diag
from .tracker import Tracker
from .xplane import XPlaneLink


def run_engine(sh: Shared, cfg: dict, status_q, preview):
    """Main thread of the tracking process: feeds the preview, or prints status when headless."""
    n, last_print = 0, 0.0
    while not sh.quit:
        time.sleep(0.05)
        if preview is not None and not preview.is_alive():
            print("preview window closed; still tracking (Ctrl+C to quit)", flush=True)
            preview = None
        if preview is not None:
            n += 1
            if not sh.preview_on and n % 10:
                continue  # preview off: just a 2 Hz status heartbeat, no video
            # full-rate video only while you're looking at the preview
            st = snapshot(sh, cfg, with_frame=sh.preview_on and (sh.preview_focused or n % 4 == 0))
            try:
                status_q.put_nowait(st)
            except queue.Full:
                pass
        elif time.monotonic() - last_print >= 1.0:
            last_print = time.monotonic()
            with sh.lock:
                lines = status_lines(sh, cfg)
                raw = sh.raw.copy()
            print(" | ".join(lines), "| yaw/pitch/roll %+5.1f %+5.1f %+5.1f" % tuple(raw[:3]), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="run.sh", description="Webcam head tracking for X-Plane 12 (macOS)")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH, help="settings file (default: config.json here)")
    ap.add_argument("--camera", type=int, help="camera index (see --list-cameras)")
    ap.add_argument("--host", help="X-Plane host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, help="X-Plane UDP port (default 49000)")
    ap.add_argument("--no-preview", action="store_true", help="run without the camera window")
    ap.add_argument("--diag", type=Path, help="log frame / send / focus timing to this CSV (for debugging lag)")
    ap.add_argument("--list-cameras", action="store_true", help="list cameras and their numbers")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return 0

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera_index"] = args.camera
    if args.host:
        cfg["xplane_host"] = args.host
    if args.port:
        cfg["xplane_port"] = args.port
    if not MODEL_PATH.exists():
        print(f"Missing {MODEL_PATH}. Start with: bash run.sh (it downloads the face model).")
        return 1

    sh = Shared()
    sh.lean = bool(cfg["lean_enabled"])
    sh.smoothing = float(cfg["smoothing"])
    tracker = Tracker(cfg, sh)
    link = XPlaneLink(cfg, sh)
    if args.diag:
        enable_diag(args.diag)
        print(f"writing timing diagnostics to {args.diag}", flush=True)

    def toggle_pause():
        with sh.lock:
            sh.paused = not sh.paused
            paused = sh.paused
        play_sound(cfg, "Bottle" if paused else "Pop")
        print("paused" if paused else "resumed", flush=True)

    def toggle_lean():
        with sh.lock:
            sh.lean = not sh.lean
            lean = sh.lean
        play_sound(cfg, "Glass" if lean else "Submarine")
        print("leaning on" if lean else "leaning off", flush=True)

    def set_range(deg: float, save: bool):
        set_look_range(cfg, deg)
        if save:
            save_settings(args.config, cfg)
            print(f"look range +/-{cfg['axes']['yaw']['output_max']:.0f} deg left/right, "
                  f"+/-{cfg['axes']['pitch']['output_max']:.0f} up/down (saved)", flush=True)

    def set_smoothing(pct: float, save: bool):
        with sh.lock:
            sh.smoothing = cfg["smoothing"] = float(pct)
        if save:
            save_settings(args.config, cfg)
            print(f"smoothness {pct:.0f}% (saved)", flush=True)

    def set_rotation(mode):
        tracker.set_rotation_mode(mode)
        cfg["camera_rotation"] = mode
        save_settings(args.config, cfg)
        print(f"camera rotation: {mode} (saved)", flush=True)

    actions = {"recenter": tracker.request_recenter, "pause": toggle_pause, "lean": toggle_lean}
    set_thread_interactive()
    if not disable_app_nap():
        print("warning: couldn't disable App Nap")
    hotkey_status = start_global_hotkeys(cfg["hotkeys"], actions)
    print(hotkey_status, flush=True)

    def handle_commands(cmd_q):
        while True:
            cmd, *arg = cmd_q.get()
            if cmd == "quit":
                with sh.lock:
                    sh.quit = True
                return
            if cmd == "focus":
                sh.preview_focused = arg[0]
                diag("focus", float(arg[0]))
            elif cmd == "range":
                set_range(*arg)
            elif cmd == "smoothing":
                set_smoothing(*arg)
            elif cmd == "preview":
                sh.preview_on = arg[0]
            elif cmd == "rotation":
                set_rotation(arg[0])
            elif cmd in actions:
                actions[cmd]()

    preview, status_q = None, None
    if not args.no_preview:
        ctx = multiprocessing.get_context("spawn")
        status_q, cmd_q = ctx.Queue(maxsize=2), ctx.Queue()
        preview = ctx.Process(target=preview_process, name="preview", daemon=True,
                              args=(cfg, status_q, cmd_q, "" if "OFF" not in hotkey_status else "(global hotkeys off)"))
        preview.start()
        threading.Thread(target=handle_commands, args=(cmd_q,), daemon=True, name="commands").start()

    tracker.start()
    link.start()
    try:
        run_engine(sh, cfg, status_q, preview)
    except KeyboardInterrupt:
        pass
    finally:
        with sh.lock:
            sh.quit = True
        if preview is not None and preview.is_alive():
            try:
                status_q.put(None, timeout=0.5)  # ask the preview to close
            except queue.Full:
                pass
            preview.join(2)
            if preview.is_alive():
                preview.terminate()
        link.join(2)
        tracker.join(2)
        close_diag()
    return 0
