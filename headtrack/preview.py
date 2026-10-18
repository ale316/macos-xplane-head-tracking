"""Preview window: camera, simulated cockpit view, sliders and buttons.

Runs in its own process. macOS throttles any background process that has a window (we
measured 4x slower face tracking and 200 ms gaps in updates to X-Plane while X-Plane was in
front), so the tracking engine has no window; at worst this preview redraws slowly.
The two processes talk through queues: snapshots of the engine's state one way (`snapshot`),
commands such as ("range", 90, True) the other (handled by app.Controls).
"""
from __future__ import annotations

import copy
import multiprocessing
import queue
import signal
import textwrap
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from .cockpit import CockpitView
from .config import AXES
from .macos import accept_first_click, pump_events, window_visible
from .state import Shared
from .xplane import VIEW_3D_COCKPIT


def status_lines(sh: Shared, cfg: dict) -> list[str]:
    if sh.error:  # wrapped to fit the window; the full traceback is in the terminal
        return textwrap.wrap(f"ERROR: {sh.error}", 72, max_lines=3, placeholder=" ... (see terminal)")
    if not sh.xp_connected:
        xp = f"X-Plane: waiting for {cfg['xplane_host']}:{cfg['xplane_port']}"
    elif sh.xp_view not in (None, VIEW_3D_COCKPIT) and cfg["only_in_cockpit_view"]:
        xp = "X-Plane: connected, not in 3D cockpit view (idle)"
    else:
        xp = "X-Plane: connected" + (" - DRIVING VIEW" if sh.xp_active else "")
    if sh.paused:
        tr = "Tracking: PAUSED"
    elif sh.recentering:
        tr = "Tracking: recentering - look at the screen"
    elif not sh.face_ok:
        tr = "Tracking: no face - holding view"
    elif sh.hand_at_face:
        tr = f"Tracking: hand at face - {'holding view' if cfg['hand_at_face'] == 'hold' else 'steadied'}"
    else:
        tr = "Tracking: OK"
    if sh.cam_target_fps and 0 < sh.cam_fps < 0.75 * sh.cam_target_fps:
        xp += f"   camera at {sh.cam_fps:.0f} of {sh.cam_target_fps} fps: Video Effects on? (Control Center)"
    look = f"look +/-{cfg['axes']['yaw']['output_max']:.0f}/{cfg['axes']['pitch']['output_max']:.0f} deg"
    cam = f"cam {sh.cam_fps:.0f} fps" + (" (locked)" if sh.cam_locked else "")
    if sh.cam_rotation:
        cam += f" rot {sh.cam_rotation}"
    n = sh.noise
    return [xp,
            f"{tr}   {look}   smooth {sh.smoothing:.0f}%   lean {'ON' if sh.lean else 'OFF'}   {cam}",
            f"noise: yaw {n[0]:.2f} pit {n[1]:.2f} roll {n[2]:.2f} deg  x {n[3]:.2f} y {n[4]:.2f} z {n[5]:.2f} cm"]


STATUS_FIELDS = ("error", "xp_connected", "xp_view", "xp_active", "xp_fov", "paused", "recentering", "face_ok",
                 "hand_at_face", "hand_boxes", "smoothing", "lean", "cam_fps", "cam_locked", "cam_target_fps",
                 "cam_name", "cam_rotation", "cam_rotation_mode", "noise", "raw", "target", "landmarks")


def snapshot(sh: Shared, cfg: dict, with_frame: bool) -> dict:
    """Picklable copy of what the preview shows; the camera frame goes as JPEG."""
    with sh.lock:
        st = {k: getattr(sh, k) for k in STATUS_FIELDS}
        st["cam_names"] = [c.name for c in sh.cam_list]
        st["cam_index"] = next((i for i, c in enumerate(sh.cam_list) if c.id == sh.cam_id), None)
        frame = sh.frame  # the tracker replaces (never mutates) frames, so no copy needed
    st = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in st.items()}
    st["axes"] = copy.deepcopy(cfg["axes"])  # ranges can change at runtime
    st["jpeg"] = None
    if with_frame and frame is not None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        st["jpeg"] = buf.tobytes() if ok else None
    return st


# Preview layout (pixels). The window is drawn by hand so it can have a two-handled slider.
UI_W, CAM_H, SIM_H, CTRL_H = 800, 300, 300, 182
RANGE_ROW, SMOOTH_ROW, CAMERA_ROW, BUTTON_ROW = 28, 68, 98, 142  # y within the controls area
RANGE_MIN, RANGE_MAX = 30.0, 180.0
SLIDER_X0, SLIDER_X1 = 150, 640
WHITE, GREY, DIM, ACCENT = (230, 230, 230), (200, 200, 200), (150, 150, 150), (0, 200, 255)  # BGR


def text(img, s: str, xy: tuple[int, int], scale: float = 0.5, color=WHITE, thickness: int = 1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class PreviewUI:
    """Widgets and drawing for the preview window. Sends commands; never touches the engine."""

    def __init__(self, cfg: dict, cmd_q, hotkey_status: str):
        self.cfg, self.cmd_q, self.hotkey_status = cfg, cmd_q, hotkey_status
        self.preview_on = True
        self.range = cfg["axes"]["yaw"]["output_max"]
        self.smoothing = float(cfg["smoothing"])
        self.drag = None  # "range" | "smoothing" while the mouse button is held on a slider
        self.rotation_mode = cfg["camera_rotation"]
        self.menu_open = False  # camera list
        self.menu_items: list[tuple[tuple[int, int, int, int], tuple]] = []
        self.buttons: list[tuple[tuple[int, int, int, int], tuple]] = []
        self.cockpit = None  # built in the background (takes ~0.5 s) so the window responds at once
        threading.Thread(target=self._build_cockpit, daemon=True, name="cockpit").start()
        self.dirty = True
        self.clicked = False  # user input since the last redraw
        self.mouse_events, self.input_t = 0, None  # for the --diag performance report
        self.pending: dict[str, tuple[float, float]] = {}  # slider -> (value sent, until when to trust it)
        self.quit = False

    # ---- input
    def on_mouse(self, event, x, y, flags, _param):
        top = self.controls_top()
        self.mouse_events += 1
        if self.input_t is None and (event != cv2.EVENT_MOUSEMOVE or self.drag):
            self.input_t = time.monotonic()  # to measure how long until it shows on screen
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            self.clicked = True
        if event == cv2.EVENT_LBUTTONDOWN and self.menu_open:
            self.menu_open, self.dirty = False, True
            for rect, cmd in self.menu_items:
                if inside(rect, x, y):
                    self.cmd_q.put(cmd)
                    return
            if any(cmd == ("camera_menu",) and inside(rect, x, y) for rect, cmd in self.buttons):
                return  # clicking the camera button again just closes the list
        if event == cv2.EVENT_LBUTTONDOWN:
            on_sliders = SLIDER_X0 - 15 <= x <= SLIDER_X1 + 15
            if on_sliders and abs(y - (top + RANGE_ROW)) < 18:
                self.drag = "range"
            elif on_sliders and abs(y - (top + SMOOTH_ROW)) < 18:
                self.drag = "smoothing"
            else:
                for rect, cmd in self.buttons:
                    if inside(rect, x, y):
                        self.press(cmd)
        if self.drag and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            done = event == cv2.EVENT_LBUTTONUP
            if self.drag == "range":  # both handles move together: distance from the middle sets the range
                half = (SLIDER_X1 - SLIDER_X0) / 2
                value = round(min(RANGE_MAX, max(RANGE_MIN, abs(x - (SLIDER_X0 + half)) / half * RANGE_MAX)))
            else:
                value = round(min(100.0, max(0.0, (x - SLIDER_X0) / (SLIDER_X1 - SLIDER_X0) * 100)))
            setattr(self, self.drag, value)
            self.cmd_q.put((self.drag, value, done))  # done: the engine saves the setting
            self.pending[self.drag] = (value, time.monotonic() + 1.0)
            if done:
                self.drag = None
            self.dirty = True

    def press(self, cmd: tuple):
        if cmd == ("preview",):
            self.preview_on = not self.preview_on
            self.cmd_q.put(("preview", self.preview_on))
        elif cmd == ("camera_menu",):
            self.menu_open = not self.menu_open
            self.cmd_q.put(("cameras",))  # refresh the list, e.g. an iPhone that just connected
        elif cmd == ("rotation",):  # cycle auto -> 0 -> 90 -> 180 -> 270 -> auto
            modes = ["auto", 0, 90, 180, 270]
            cur = self.rotation_mode if self.rotation_mode in modes else "auto"
            self.rotation_mode = modes[(modes.index(cur) + 1) % len(modes)]
            self.cmd_q.put(("rotation", self.rotation_mode))
        else:
            self.cmd_q.put(cmd)
            self.quit = cmd == ("quit",)
        self.dirty = True

    def sync(self, st: dict):
        """Adopt the engine's values, except for a slider that's being dragged or whose new value
        the engine hasn't confirmed yet (older status updates would make it jump back)."""
        for name, value in (("range", st["axes"]["yaw"]["output_max"]), ("smoothing", st["smoothing"])):
            if self.drag == name:
                continue
            sent = self.pending.get(name)
            if sent is not None:
                if abs(value - sent[0]) > 0.5 and time.monotonic() < sent[1]:
                    continue  # the engine hasn't caught up yet: keep showing what was set
                del self.pending[name]
            setattr(self, name, value)
        self.rotation_mode = st["cam_rotation_mode"]

    def controls_top(self) -> int:
        return CAM_H + SIM_H if self.preview_on else 30

    # ---- drawing
    def draw(self, st: dict | None, frame) -> np.ndarray:
        top = self.controls_top()
        img = np.full((top + CTRL_H, UI_W, 3), 30, np.uint8)
        if self.preview_on:
            self.draw_camera(img[:CAM_H], st, frame)
            self.draw_sim(img[CAM_H:CAM_H + SIM_H], st)
        else:
            text(img, "Preview off - no video or drawing work while you fly", (10, 20), color=(170, 170, 170))
        self.draw_controls(img, top, st)
        return img

    def draw_camera(self, area, st, frame):
        cw = CAM_H * 4 // 3  # the camera area; a portrait image is drawn narrower inside it
        if frame is None:
            text(area, "Starting camera...", (20, CAM_H // 2), 0.7, (255, 255, 255), 2)
        else:
            fw = min(cw, round(CAM_H * frame.shape[1] / frame.shape[0]))
            cam = cv2.resize(cv2.flip(frame, 1), (fw, CAM_H))  # mirrored so it feels natural
            if st is not None and st["landmarks"] is not None:
                for x, y in st["landmarks"][::3]:
                    cv2.circle(cam, (int((1 - x) * fw), int(y * CAM_H)), 1, (80, 220, 120), -1)
            for x0, y0, x1, y1 in (st or {}).get("hand_boxes") or []:
                color = (0, 140, 255) if st["hand_at_face"] else GREY
                cv2.rectangle(cam, (int((1 - x1) * fw), int(y0 * CAM_H)), (int((1 - x0) * fw), int(y1 * CAM_H)),
                              color, 2)
            area[:, (cw - fw) // 2:(cw - fw) // 2 + fw] = cam
        if st is None:
            return
        x0 = cw + 12
        wrapped = []
        for line in status_lines(SimpleNamespace(**st), dict(self.cfg, axes=st["axes"])):
            wrapped += textwrap.wrap(line, 48) or [""]
        for i, line in enumerate(wrapped[:7]):
            text(area, line, (x0, 22 + 19 * i), 0.45, (80, 80, 255) if st["error"] else WHITE)
        units = ["deg", "deg", "deg", "cm", "cm", "cm"]
        for i, a in enumerate(AXES):
            text(area, f"{a:>5} {st['raw'][i]:+6.1f} {units[i]:<3} -> {st['target'][i]:+7.2f}", (x0, 170 + 20 * i),
                 0.42, GREY)

    def _build_cockpit(self):
        self.cockpit = CockpitView()
        self.dirty = True

    def draw_sim(self, area, st):
        if self.cockpit is None:
            text(area, "Preparing the simulated view...", (20, SIM_H // 2), 0.6, (170, 170, 170))
            return
        out = st["target"] if st is not None else np.zeros(6)
        fov = (st or {}).get("xp_fov") or 75.0
        zoom = min(2.0, max(0.7, 1 + 2 * out[5]))  # leaning in/out (no parallax in this stand-in)
        area[:] = self.cockpit.render(out[0], out[1], out[2], UI_W, SIM_H, fov, zoom)
        cv2.rectangle(area, (0, 0), (UI_W, 24), (20, 20, 20), -1)
        side = "right" if out[0] >= 0 else "left"
        vert = "up" if out[1] >= 0 else "down"
        text(area, f"Simulated view (X-Plane FOV {fov:.0f} deg): looking {abs(out[0]):.0f} {side}, "
                   f"{abs(out[1]):.0f} {vert}", (8, 17), 0.45)

    def draw_controls(self, img, top, st):
        mid = (SLIDER_X0 + SLIDER_X1) // 2
        half = (SLIDER_X1 - SLIDER_X0) / 2
        # look range: two handles, symmetric around the middle
        y = top + RANGE_ROW
        off = int(self.range / RANGE_MAX * half)
        ticks = {int(mid + sgn * deg / RANGE_MAX * half): str(deg) for deg in (90, 180) for sgn in (-1, 1)}
        self.draw_slider(img, y, "Look range", (mid - off, mid + off), f"+/-{self.range:.0f} deg", ticks)
        # smoothness: single handle
        y = top + SMOOTH_ROW
        hx = int(SLIDER_X0 + self.smoothing / 100 * (SLIDER_X1 - SLIDER_X0))
        self.draw_slider(img, y, "Smoothness", (hx,), f"{self.smoothing:.0f}%")
        text(img, "snappy", (SLIDER_X0 - 5, y + 22), 0.38, DIM)
        text(img, "steady", (SLIDER_X1 - 40, y + 22), 0.38, DIM)
        # camera row: which camera (opens a list) and its rotation
        self.buttons = []
        y = top + CAMERA_ROW
        text(img, "Camera", (10, y + 21))
        name = (st or {}).get("cam_name") or "..."
        name = name if len(name) <= 30 else name[:29] + "..."
        x = self.button(img, SLIDER_X0, y, f"{name}    ", ("camera_menu",))
        tri = np.array([[x - 22, y + 13], [x - 10, y + 13], [x - 16, y + 20]], np.int32)
        cv2.fillPoly(img, [tri], (240, 240, 240))  # dropdown arrow
        self.button(img, x + 10, y, rotation_label(st), ("rotation",))
        # main buttons
        paused = bool(st and st["paused"])
        lean = bool(st is None or st["lean"])
        x, y = 10, top + BUTTON_ROW
        for label, cmd in ((f"Preview: {'ON' if self.preview_on else 'OFF'}", ("preview",)),
                           ("Resume" if paused else "Pause", ("pause",)), ("Recenter", ("recenter",)),
                           (f"Lean: {'ON' if lean else 'OFF'}", ("lean",)), ("Quit", ("quit",))):
            x = self.button(img, x, y, label, cmd) + 10
        text(img, self.hotkey_status, (x + 6, y + 21), 0.4, DIM)
        self.menu_items = []
        if self.menu_open:
            self.draw_camera_menu(img, top + CAMERA_ROW, st)

    @staticmethod
    def draw_slider(img, y: int, label: str, handles: tuple, value: str, ticks: dict | None = None):
        """A track highlighted between its handles (or from its start, with one handle), with
        optional labelled tick marks {x: label}."""
        text(img, label, (10, y + 5))
        cv2.line(img, (SLIDER_X0, y), (SLIDER_X1, y), (90, 90, 90), 4)
        for tx, tick in (ticks or {}).items():
            cv2.line(img, (tx, y - 8), (tx, y + 8), (120, 120, 120), 1)
            text(img, tick, (tx - 10, y + 22), 0.38, DIM)
        cv2.line(img, (handles[0] if len(handles) > 1 else SLIDER_X0, y), (handles[-1], y), (0, 170, 255), 4)
        for hx in handles:
            cv2.circle(img, (hx, y), 9, ACCENT, -1)
        text(img, value, (SLIDER_X1 + 20, y + 5))

    def draw_camera_menu(self, img, button_y: int, st):
        """The list of cameras, opening upwards from the camera button."""
        names = (st or {}).get("cam_names") or []
        current = (st or {}).get("cam_index")
        if not names:
            names, current = ["(no cameras found)"], None
        w = max(cv2.getTextSize(n, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for n in names) + 60
        x0, y1 = SLIDER_X0, button_y - 4
        y0 = max(0, y1 - 30 * len(names) - 8)
        cv2.rectangle(img, (x0, y0), (x0 + w, y1), (55, 55, 55), -1)
        cv2.rectangle(img, (x0, y0), (x0 + w, y1), DIM, 1)
        for i, n in enumerate(names):
            top = y0 + 4 + 30 * i
            cv2.circle(img, (x0 + 18, top + 15), 7, GREY, 1, cv2.LINE_AA)
            if i == current:
                cv2.circle(img, (x0 + 18, top + 15), 4, ACCENT, -1, cv2.LINE_AA)
            text(img, n, (x0 + 34, top + 20), color=(240, 240, 240))
            if current is not None:
                self.menu_items.append(((x0, top, x0 + w, top + 30), ("camera", i)))

    def button(self, img, x: int, y: int, label: str, cmd: tuple) -> int:
        """Draws a button and registers it for clicks; returns its right edge."""
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        rect = (x, y, x + tw + 24, y + 32)
        cv2.rectangle(img, rect[:2], rect[2:], (70, 70, 70), -1)
        cv2.rectangle(img, rect[:2], rect[2:], (120, 120, 120), 1)
        text(img, label, (x + 12, y + 21), color=(240, 240, 240))
        self.buttons.append((rect, cmd))
        return rect[2]


def inside(rect: tuple[int, int, int, int], x: int, y: int) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def rotation_label(st) -> str:
    if st is None:
        return "Rotation: ..."
    if st["cam_rotation_mode"] == "auto":
        return f"Rotation: auto ({st['cam_rotation']})"
    return f"Rotation: {st['cam_rotation_mode']}"


KEYS = {ord("c"): ("recenter",), ord(" "): ("recenter",), ord("p"): ("pause",), ord("l"): ("lean",),
        ord("v"): ("preview",), ord("q"): ("quit",), 27: ("quit",)}


def preview_process(cfg: dict, status_q, cmd_q, hotkey_status: str):
    """Preview window (separate process): draws engine snapshots, sends user input back."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C goes to the engine, which closes us
    engine = multiprocessing.parent_process()
    ui = PreviewUI(cfg, cmd_q, hotkey_status)
    accept_first_click()
    win = "X-Plane Head Tracker"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, ui.on_mouse)
    st, frame = None, None
    last_draw, was_visible = 0.0, None
    perf = {"since": time.monotonic(), "redraws": 0, "gap": 0.0, "lag": 0.0}  # reported once a second
    while (engine is None or engine.is_alive()) and not ui.quit:
        loop_start = time.monotonic()
        key = pump_events()  # mouse and keys first, so a drag shows up in this very redraw
        if key in KEYS:
            ui.press(KEYS[key])
        try:
            while True:  # only the newest snapshot matters
                st = status_q.get_nowait()
                if st is None:
                    return
                ui.sync(st)
                if st["jpeg"] is not None:
                    frame = cv2.imdecode(np.frombuffer(st["jpeg"], np.uint8), cv2.IMREAD_COLOR)
                ui.dirty = True
        except queue.Empty:
            pass
        visible = window_visible(win)
        if visible != was_visible:
            cmd_q.put(("visible", visible))
            was_visible = visible
        # Full rate while the preview is on screen, ~5 fps while it's hidden (e.g. behind a
        # full-screen X-Plane), and only on input while it's switched off.
        interval = 0.0 if visible else 0.2
        due = ui.dirty if not ui.preview_on else time.monotonic() - last_draw >= interval
        if not (due or ui.clicked or ui.drag):  # clicks always get an immediate redraw
            time.sleep(0.02)
            continue
        now = time.monotonic()
        perf["gap"] = max(perf["gap"], now - last_draw) if perf["redraws"] else 0.0
        last_draw, ui.dirty, ui.clicked = now, False, False
        cv2.imshow(win, ui.draw(st, frame))
        perf["redraws"] += 1
        if ui.input_t is not None:  # input -> on screen
            perf["lag"], ui.input_t = max(perf["lag"], time.monotonic() - ui.input_t), None
        if now - perf["since"] >= 1.0:  # redraws/s, slowest gap and input lag (ms), visible, mouse events
            cmd_q.put(("perf", perf["redraws"], perf["gap"] * 1000, perf["lag"] * 1000, float(visible),
                       ui.mouse_events))
            perf.update(since=now, redraws=0, gap=0.0, lag=0.0)
            ui.mouse_events = 0
        time.sleep(max(0.0, 1 / 60 - (time.monotonic() - loop_start)))  # up to 60 redraws a second
    cv2.destroyAllWindows()
