"""Command line entry point: wires the tracker, the X-Plane link and the preview together."""
from __future__ import annotations

import argparse
import multiprocessing
import platform
import queue
import sys
import threading
import time
from pathlib import Path

from . import __version__
from .airpods import HeadphoneMotion
from .camera import camera_to_use, list_cameras, pick_camera
from .config import CONFIG_PATH, MODEL_PATH, ConfigError, load_config, save_settings, set_look_range
from .macos import disable_app_nap, play_sound, set_thread_interactive, start_global_hotkeys
from .preview import preview_process, snapshot, status_lines
from .state import Shared, close_diag, diag, enable_diag
from .tracker import Tracker
from .xplane import XPlaneLink


class Controls:
    """Carries out commands from the preview window and the global hotkeys, and saves the
    settings they change."""

    def __init__(self, cfg: dict, sh: Shared, tracker: Tracker, config_path: Path):
        self.cfg, self.sh, self.tracker, self.config_path = cfg, sh, tracker, config_path
        self.hotkey_actions = {"recenter": tracker.request_recenter, "pause": self.toggle_pause,
                               "lean": self.toggle_lean}
        self.airpods: HeadphoneMotion | None = None

    def handle(self, cmd: str, *args):
        """One command from the preview, e.g. ("range", 90, True)."""
        handler = {
            **self.hotkey_actions,
            "quit": self.quit,
            "range": self.set_range,  # (degrees, save)
            "smoothing": self.set_smoothing,  # (0-100, save)
            "camera": self.select_camera,  # (index in the camera list,)
            "cameras": self.refresh_cameras,
            "rotation": self.set_rotation,  # ("auto" or 0/90/180/270,)
            "airpods": self.set_airpods,  # (on,)
            "preview": self.set_preview,  # (on,)
            "visible": self.set_preview_visible,  # (on screen,)
            "perf": lambda *perf: diag("preview", *perf),
        }.get(cmd)
        if handler is not None:
            handler(*args)

    def serve(self, cmd_q):
        """Handle the preview's commands until it asks to quit."""
        while not self.sh.quit:
            self.handle(*cmd_q.get())

    def quit(self):
        with self.sh.lock:
            self.sh.quit = True

    def toggle_pause(self):
        with self.sh.lock:
            self.sh.paused = paused = not self.sh.paused
        play_sound(self.cfg, "Bottle" if paused else "Pop")
        print("paused" if paused else "resumed", flush=True)

    def toggle_lean(self):
        with self.sh.lock:
            self.sh.lean = lean = not self.sh.lean
        play_sound(self.cfg, "Glass" if lean else "Submarine")
        print("leaning on" if lean else "leaning off", flush=True)

    def set_range(self, degrees: float, save: bool):
        set_look_range(self.cfg, degrees)
        if save:  # on release, not on every step of a drag
            save_settings(self.config_path, self.cfg)
            axes = self.cfg["axes"]
            print(f"look range +/-{axes['yaw']['output_max']:.0f} deg left/right, "
                  f"+/-{axes['pitch']['output_max']:.0f} up/down (saved)", flush=True)

    def set_smoothing(self, pct: float, save: bool):
        with self.sh.lock:
            self.sh.smoothing = self.cfg["smoothing"] = float(pct)
        if save:
            save_settings(self.config_path, self.cfg)
            print(f"smoothness {pct:.0f}% (saved)", flush=True)

    def set_rotation(self, mode):
        self.tracker.set_rotation_mode(mode)
        self.cfg["camera_rotation"] = mode
        save_settings(self.config_path, self.cfg)
        print(f"camera rotation: {mode} (saved)", flush=True)

    def set_airpods(self, on: bool):
        """Turn the view with AirPods head tracking (on) or with the camera."""
        if on and self.airpods is None:
            self.airpods = HeadphoneMotion(self.sh)
            self.airpods.start()
        elif not on and self.airpods is not None:
            self.airpods.stop()
            self.airpods = None
        if self.cfg["airpods"] != on:
            self.cfg["airpods"] = on
            save_settings(self.config_path, self.cfg)
            print(f"AirPods head tracking {'on' if on else 'off'} (saved)", flush=True)

    def set_preview(self, on: bool):
        self.sh.preview_on = on

    def set_preview_visible(self, visible: bool):
        self.sh.preview_visible = visible
        diag("visible", float(visible))

    def select_camera(self, index: int):
        cameras = self.sh.cam_list
        if index >= len(cameras) or cameras[index].id == self.sh.cam_id:
            return
        self.tracker.request_camera(cameras[index])
        self.cfg["camera_id"], self.cfg["camera_name"] = cameras[index].id, cameras[index].name
        save_settings(self.config_path, self.cfg)

    def refresh_cameras(self):
        cameras = list_cameras()
        with self.sh.lock:
            self.sh.cam_list = cameras

    def watch_cameras(self):
        """Follow cameras coming and going: switch to the chosen one when it shows up (an iPhone
        connecting late), or to another one if the camera in use disappears."""
        while not self.sh.quit:
            time.sleep(3)
            cameras = list_cameras()
            with self.sh.lock:
                self.sh.cam_list, current = cameras, self.sh.cam_id
            switch = camera_to_use(self.cfg, cameras, current)
            if switch is not None and switch != self.tracker.camera:  # not the one it's still opening
                print(f"camera: switching to {switch.name}", flush=True)
                self.tracker.request_camera(switch)


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
            # full-rate video only while the preview is on screen
            st = snapshot(sh, cfg, with_frame=sh.preview_on and (sh.preview_visible or n % 4 == 0))
            try:
                status_q.put_nowait(st)
            except queue.Full:
                pass
        elif time.monotonic() - last_print >= 1.0:
            last_print = time.monotonic()
            with sh.lock:
                lines = status_lines(sh, cfg)
                raw = sh.raw.copy()
            print(" | ".join(lines), f"| yaw/pitch/roll {raw[0]:+5.1f} {raw[1]:+5.1f} {raw[2]:+5.1f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="run.sh", description="Webcam head tracking for X-Plane 12 (macOS)")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH, help="settings file (default: config.json here)")
    ap.add_argument("--camera", type=int, help="camera number for this run (see --list-cameras)")
    ap.add_argument("--host", help="X-Plane host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, help="X-Plane UDP port (default 49000)")
    ap.add_argument("--no-preview", action="store_true", help="run without the camera window")
    ap.add_argument("--diag", type=Path, help="log tracking and preview timing to this CSV (for debugging)")
    ap.add_argument("--list-cameras", action="store_true", help="list cameras and their numbers")
    ap.add_argument("--version", action="version", version=f"X-Plane Head Tracker {__version__}")
    args = ap.parse_args()

    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("Sorry, X-Plane Head Tracker only runs on Macs with Apple Silicon (M1 or newer).")
        return 1

    if args.list_cameras:
        for i, cam in enumerate(list_cameras()):
            print(f"{i}: {cam.name}")
        return 0

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(e)
        return 1
    cameras = list_cameras()
    if args.camera is not None and 0 <= args.camera < len(cameras):
        camera = cameras[args.camera]
    else:
        if args.camera is not None:
            print(f"There's no camera {args.camera} (see --list-cameras); using the saved one.")
        camera = pick_camera(cfg, cameras)
    if args.host:
        cfg["xplane_host"] = args.host
    if args.port:
        cfg["xplane_port"] = args.port
    if not MODEL_PATH.exists():
        print(f"Missing {MODEL_PATH}. Start with: bash run.sh (it downloads the face model).")
        return 1

    sh = Shared()
    sh.cam_list = cameras
    sh.lean = bool(cfg["lean_enabled"])
    sh.smoothing = float(cfg["smoothing"])
    tracker = Tracker(cfg, sh, camera)
    link = XPlaneLink(cfg, sh)
    controls = Controls(cfg, sh, tracker, args.config)
    if args.diag:
        enable_diag(args.diag)
        print(f"writing diagnostics to {args.diag}", flush=True)

    set_thread_interactive()
    if not disable_app_nap():
        print("warning: couldn't disable App Nap")
    hotkey_status = start_global_hotkeys(cfg["hotkeys"], controls.hotkey_actions)
    print(hotkey_status, flush=True)

    preview, status_q = None, None
    if not args.no_preview:
        ctx = multiprocessing.get_context("spawn")
        status_q, cmd_q = ctx.Queue(maxsize=2), ctx.Queue()
        preview = ctx.Process(target=preview_process, name="preview", daemon=True,
                              args=(cfg, status_q, cmd_q, "" if "OFF" not in hotkey_status else "(global hotkeys off)"))
        preview.start()
        threading.Thread(target=controls.serve, args=(cmd_q,), daemon=True, name="commands").start()

    tracker.start()
    link.start()
    controls.set_airpods(cfg["airpods"])
    threading.Thread(target=controls.watch_cameras, daemon=True, name="cameras").start()
    try:
        run_engine(sh, cfg, status_q, preview)
    except KeyboardInterrupt:
        pass
    finally:
        controls.quit()
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
