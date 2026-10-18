"""Preview window: camera, simulated cockpit view, sliders and buttons.

Runs in its own process. macOS throttles any background process that has a window (we
measured 4x slower face tracking and 200 ms gaps in updates to X-Plane while X-Plane was in
front), so the tracking engine has no window; at worst this preview redraws slowly.
"""
from __future__ import annotations

import copy
import math
import multiprocessing
import queue
import signal
import textwrap
import time
from types import SimpleNamespace

import numpy as np

from .config import AXES
from .macos import app_focused
from .pose import rot_x, rot_y, rot_z
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
    cam = f"cam {sh.cam_fps:.0f} fps" + (" (locked)" if "locked at" in sh.cam_info else "")
    if sh.cam_rotation:
        cam += f" rot {sh.cam_rotation}"
    n = sh.noise
    return [xp,
            f"{tr}   {look}   smooth {sh.smoothing:.0f}%   lean {'ON' if sh.lean else 'OFF'}   {cam}",
            f"noise: yaw {n[0]:.2f} pit {n[1]:.2f} roll {n[2]:.2f} deg  x {n[3]:.2f} y {n[4]:.2f} z {n[5]:.2f} cm"]


STATUS_FIELDS = ("error", "xp_connected", "xp_view", "xp_active", "xp_fov", "paused", "recentering", "face_ok",
                 "hand_at_face", "hand_boxes", "smoothing", "lean", "cam_fps", "cam_info", "cam_target_fps",
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
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        st["jpeg"] = buf.tobytes() if ok else None
    return st


class CockpitView:
    """Stand-in for X-Plane's 3D cockpit, to show in the preview what the head pose does to
    the view: a procedural 360-degree cockpit (window posts, glareshield and gauges, cabin
    walls, sky, ground grid, runway) rendered for a head direction and field of view."""

    def __init__(self, width: int = 2048):
        h = width // 2
        lon = (np.arange(width) + 0.5) / width * 360 - 180  # 0 = straight ahead, + = right
        lat = 90 - (np.arange(h) + 0.5) / h * 180  # + = up
        LON, LAT = np.meshgrid(lon, lat)
        a = np.abs(LON)
        img = np.zeros((h, width, 3), np.float32)
        sky = np.clip(LAT / 60, 0, 1)[..., None]
        img[:] = (1 - sky) * np.array([235, 215, 190]) + sky * np.array([200, 140, 70])  # BGR
        below = LAT < 0
        d = 2.0 / np.tan(np.radians(np.maximum(-LAT, 0.3)))  # distance on the ground, eye 2 m up
        gx, gz = d * np.sin(np.radians(LON)), d * np.cos(np.radians(LON))
        # 25 m grid, offset half a cell so no line runs through the viewer (it would show as a wedge)
        grid = (np.abs((gx / 25 + 0.5) % 1 - 0.5) > 0.47) | (np.abs((gz / 25 + 0.5) % 1 - 0.5) > 0.47)
        ground = np.where(grid[..., None], [70, 110, 80], [80, 140, 95]).astype(np.float32)
        haze = np.clip(-LAT / 4, 0, 1)[..., None]
        ground = haze * ground + (1 - haze) * np.array([200, 200, 190])
        runway = (np.abs(gx) < 12) & (gz > 0)
        ground[runway] = [110, 110, 110]
        ground[runway & (np.abs(gx) < 0.6) & ((gz / 12) % 1 < 0.5)] = [235, 235, 235]
        img[below] = ground[below]
        cabin, frame = np.array([60, 60, 66], np.float32), np.array([35, 35, 38], np.float32)
        img[LAT > np.where(a < 70, 28, 22)] = cabin * 0.8  # roof
        img[(a > 55) & (LAT < -8)] = cabin  # side walls below the side windows
        rear = a > 125
        img[rear] = cabin * 0.9  # back of the cabin
        for c, half in ((0, 0.8), (27, 2.5), (60, 3.5), (110, 4.0)):  # windscreen divider, A/B/C posts
            img[(np.abs(a - c) < half) & ~rear] = frame
        img[(np.abs(LAT - 28) < 1.2) & (a < 70)] = frame
        edge = -10 - 6 * (a / 55) ** 2  # curved glareshield
        img[(a < 55) & (LAT < edge)] = [45, 45, 48]
        img[(np.abs(LAT - edge) < 0.8) & (a < 55)] = [25, 25, 25]
        for gl in (-30, -18, -6):  # six-pack
            for gb in (-19, -30):
                r = np.hypot((LON - gl) * np.cos(np.radians(gb)), LAT - gb)
                img[r < 5] = [20, 20, 20]
                img[(r > 4.6) & (r < 5)] = [200, 200, 200]
                img[r < 0.6] = [230, 230, 230]
        for gl in (10, 22, 34):  # radio stack / screens
            img[(np.abs(LON - gl) < 5) & (np.abs(LAT + 24) < 9)] = [70, 50, 30]
        self.map = img.astype(np.uint8)
        self._rays_key, self._rays = None, None

    def render(self, yaw: float, pitch: float, roll: float, w: int, h: int, fov_deg: float, zoom: float = 1.0):
        """yaw right+, pitch up+, roll right+ (degrees); horizontal FOV; zoom > 1 = leaning in."""
        import cv2

        fov = 2 * math.degrees(math.atan(math.tan(math.radians(fov_deg) / 2) / zoom))
        if self._rays_key != (w, h, round(fov, 1)):
            f = (w / 2) / math.tan(math.radians(fov) / 2)
            u, v = np.meshgrid(np.arange(w) - w / 2 + 0.5, h / 2 - np.arange(h) - 0.5)
            r = np.stack([u, v, np.full_like(u, f)], -1).reshape(-1, 3)
            self._rays = (r / np.linalg.norm(r, axis=1, keepdims=True)).astype(np.float32)
            self._rays_key = (w, h, round(fov, 1))
        R = rot_y(yaw) @ rot_x(-pitch) @ rot_z(-roll)
        d = self._rays @ R.T.astype(np.float32)
        mh, mw = self.map.shape[:2]
        mx = np.arctan2(d[:, 0], d[:, 2]) * (mw / (2 * math.pi)) + mw / 2
        my = mh / 2 - np.arcsin(np.clip(d[:, 1], -1, 1)) * (mh / math.pi)
        return cv2.remap(self.map, mx.reshape(h, w).astype(np.float32), my.reshape(h, w).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# Preview layout (pixels). The window is drawn by hand so it can have a two-handled slider.
UI_W, CAM_H, SIM_H, CTRL_H = 800, 300, 300, 182
RANGE_MIN, RANGE_MAX = 30.0, 180.0
SLIDER_X0, SLIDER_X1 = 150, 640


class PreviewUI:
    """Widgets and drawing for the preview window. Sends commands; never touches the engine."""

    def __init__(self, cfg: dict, cmd_q, hotkey_status: str):
        self.cfg, self.cmd_q, self.hotkey_status = cfg, cmd_q, hotkey_status
        self.preview_on = True
        self.range = cfg["axes"]["yaw"]["output_max"]
        self.smoothing = float(cfg["smoothing"])
        self.drag = None  # "range" | "smooth" while the mouse button is held on a slider
        self.rotation_mode = cfg["camera_rotation"]
        self.menu_open = False  # camera list
        self.menu_items: list[tuple[tuple[int, int, int, int], tuple]] = []
        self.buttons: list[tuple[tuple[int, int, int, int], tuple]] = []
        self.cockpit = None
        self.dirty = True
        self.clicked = False  # user input since the last redraw
        self.quit = False

    # ---- input
    def on_mouse(self, event, x, y, flags, _param):
        import cv2

        top = self.controls_top()
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            self.clicked = True
        if event == cv2.EVENT_LBUTTONDOWN and self.menu_open:
            self.menu_open, self.dirty = False, True
            for (x0, y0, x1, y1), cmd in self.menu_items:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.cmd_q.put(cmd)
                    return
            for (x0, y0, x1, y1), cmd in self.buttons:  # clicking the camera button again just closes it
                if cmd == ("camera_menu",) and x0 <= x <= x1 and y0 <= y <= y1:
                    return
        if event == cv2.EVENT_LBUTTONDOWN:
            if top + 10 <= y < top + 45 and SLIDER_X0 - 15 <= x <= SLIDER_X1 + 15:
                self.drag = "range"
            elif top + 50 <= y < top + 85 and SLIDER_X0 - 15 <= x <= SLIDER_X1 + 15:
                self.drag = "smooth"
            else:
                for (x0, y0, x1, y1), cmd in self.buttons:
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        self.press(cmd)
        if self.drag and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            done = event == cv2.EVENT_LBUTTONUP
            if self.drag == "range":  # both handles move together: distance from the middle sets the range
                half = (SLIDER_X1 - SLIDER_X0) / 2
                frac = abs(x - (SLIDER_X0 + half)) / half
                self.range = round(min(RANGE_MAX, max(RANGE_MIN, frac * RANGE_MAX)))
                self.cmd_q.put(("range", self.range, done))
            else:
                frac = (x - SLIDER_X0) / (SLIDER_X1 - SLIDER_X0)
                self.smoothing = round(min(100.0, max(0.0, frac * 100)))
                self.cmd_q.put(("smoothing", self.smoothing, done))
            if done:
                self.drag = None
            self.dirty = True

    @staticmethod
    def rotation_label(st) -> str:
        if st is None:
            return "Rotation: ..."
        if st["cam_rotation_mode"] == "auto":
            return f"Rotation: auto ({st['cam_rotation']})"
        return f"Rotation: {st['cam_rotation_mode']}"

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
        """Adopt the engine's values unless the user is dragging that slider."""
        if self.drag != "range":
            self.range = st["axes"]["yaw"]["output_max"]
        if self.drag != "smooth":
            self.smoothing = st["smoothing"]
        self.rotation_mode = st["cam_rotation_mode"]

    def controls_top(self) -> int:
        return CAM_H + SIM_H if self.preview_on else 30

    # ---- drawing
    def draw(self, st: dict | None, frame) -> np.ndarray:
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        top = self.controls_top()
        img = np.full((top + CTRL_H, UI_W, 3), 30, np.uint8)
        if self.preview_on:
            self.draw_camera(img[:CAM_H], st, frame)
            self.draw_sim(img[CAM_H:CAM_H + SIM_H], st)
        else:
            cv2.putText(img, "Preview off - no video or drawing work while you fly", (10, 20), font, 0.5,
                        (170, 170, 170), 1, cv2.LINE_AA)
        self.draw_controls(img, top, st)
        return img

    def draw_camera(self, area, st, frame):
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        cw = CAM_H * 4 // 3  # the camera area; a portrait image is drawn narrower inside it
        if frame is None:
            cv2.putText(area, "Starting camera...", (20, CAM_H // 2), font, 0.7, (255, 255, 255), 2)
        else:
            fw = min(cw, round(CAM_H * frame.shape[1] / frame.shape[0]))
            cam = cv2.resize(cv2.flip(frame, 1), (fw, CAM_H))  # mirrored so it feels natural
            if st is not None and st["landmarks"] is not None:
                for x, y in st["landmarks"][::3]:
                    cv2.circle(cam, (int((1 - x) * fw), int(y * CAM_H)), 1, (80, 220, 120), -1)
            for x0, y0, x1, y1 in (st or {}).get("hand_boxes") or []:
                color = (0, 140, 255) if st["hand_at_face"] else (200, 200, 200)
                cv2.rectangle(cam, (int((1 - x1) * fw), int(y0 * CAM_H)), (int((1 - x0) * fw), int(y1 * CAM_H)),
                              color, 2)
            area[:, (cw - fw) // 2:(cw - fw) // 2 + fw] = cam
        if st is None:
            return
        x0 = cw + 12
        lines = status_lines(SimpleNamespace(**st), dict(self.cfg, axes=st["axes"]))
        wrapped = []
        for line in lines:
            wrapped += textwrap.wrap(line, 48) or [""]
        for i, line in enumerate(wrapped[:7]):
            cv2.putText(area, line, (x0, 22 + 19 * i), font, 0.45, (80, 80, 255) if st["error"] else (230, 230, 230),
                        1, cv2.LINE_AA)
        units = ["deg", "deg", "deg", "cm", "cm", "cm"]
        for i, a in enumerate(AXES):
            y = 170 + 20 * i
            cv2.putText(area, f"{a:>5} {st['raw'][i]:+6.1f} {units[i]:<3} -> {st['target'][i]:+7.2f}", (x0, y), font,
                        0.42, (200, 200, 200), 1, cv2.LINE_AA)

    def draw_sim(self, area, st):
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.cockpit is None:
            self.cockpit = CockpitView()
        out = st["target"] if st is not None else np.zeros(6)
        fov = (st or {}).get("xp_fov") or 75.0
        zoom = min(2.0, max(0.7, 1 + 2 * out[5]))  # leaning in/out (no parallax in this stand-in)
        area[:] = self.cockpit.render(out[0], out[1], out[2], UI_W, SIM_H, fov, zoom)
        cv2.rectangle(area, (0, 0), (UI_W, 24), (20, 20, 20), -1)
        side = "right" if out[0] >= 0 else "left"
        vert = "up" if out[1] >= 0 else "down"
        cv2.putText(area, f"Simulated view (X-Plane FOV {fov:.0f} deg): looking {abs(out[0]):.0f} {side}, "
                          f"{abs(out[1]):.0f} {vert}", (8, 17), font, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    def draw_controls(self, img, top, st):
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        mid = (SLIDER_X0 + SLIDER_X1) // 2
        half = (SLIDER_X1 - SLIDER_X0) / 2
        # look range: two handles, symmetric around the middle
        y = top + 28
        cv2.putText(img, "Look range", (10, y + 5), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.line(img, (SLIDER_X0, y), (SLIDER_X1, y), (90, 90, 90), 4)
        for deg in (90, 180):
            for sgn in (-1, 1):
                tx = int(mid + sgn * deg / RANGE_MAX * half)
                cv2.line(img, (tx, y - 8), (tx, y + 8), (120, 120, 120), 1)
                cv2.putText(img, str(deg), (tx - 10, y + 22), font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        off = int(self.range / RANGE_MAX * half)
        cv2.line(img, (mid - off, y), (mid + off, y), (0, 170, 255), 4)
        for hx in (mid - off, mid + off):
            cv2.circle(img, (hx, y), 9, (0, 200, 255), -1)
        cv2.putText(img, f"+/-{self.range:.0f} deg", (SLIDER_X1 + 20, y + 5), font, 0.5, (230, 230, 230), 1,
                    cv2.LINE_AA)
        # smoothness: single handle
        y = top + 68
        cv2.putText(img, "Smoothness", (10, y + 5), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.line(img, (SLIDER_X0, y), (SLIDER_X1, y), (90, 90, 90), 4)
        hx = int(SLIDER_X0 + self.smoothing / 100 * (SLIDER_X1 - SLIDER_X0))
        cv2.line(img, (SLIDER_X0, y), (hx, y), (0, 170, 255), 4)
        cv2.circle(img, (hx, y), 9, (0, 200, 255), -1)
        cv2.putText(img, f"{self.smoothing:.0f}%", (SLIDER_X1 + 20, y + 5), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, "snappy", (SLIDER_X0 - 5, y + 22), font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "steady", (SLIDER_X1 - 40, y + 22), font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        # camera row: which camera (click for the next one) and its rotation
        self.buttons = []
        y0 = top + 98
        cv2.putText(img, "Camera", (10, y0 + 21), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        name = (st or {}).get("cam_name") or "..."
        name = name if len(name) <= 30 else name[:29] + "..."
        x = self.button(img, SLIDER_X0, y0, f"{name}    ", ("camera_menu",))
        tri = np.array([[x - 22, y0 + 13], [x - 10, y0 + 13], [x - 16, y0 + 20]], np.int32)
        cv2.fillPoly(img, [tri], (240, 240, 240))  # dropdown arrow
        self.button(img, x + 10, y0, self.rotation_label(st), ("rotation",))
        # main buttons
        paused = bool(st and st["paused"])
        lean = bool(st is None or st["lean"])
        x, y0 = 10, top + 142
        for label, cmd in ((f"Preview: {'ON' if self.preview_on else 'OFF'}", ("preview",)),
                           ("Resume" if paused else "Pause", ("pause",)), ("Recenter", ("recenter",)),
                           (f"Lean: {'ON' if lean else 'OFF'}", ("lean",)), ("Quit", ("quit",))):
            x = self.button(img, x, y0, label, cmd) + 10
        cv2.putText(img, self.hotkey_status, (x + 6, y0 + 21), font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        self.menu_items = []
        if self.menu_open:
            self.draw_camera_menu(img, top + 98, st)

    def draw_camera_menu(self, img, button_y: int, st):
        """The list of cameras, opening upwards from the camera button."""
        import cv2

        names = (st or {}).get("cam_names") or []
        current = (st or {}).get("cam_index")
        if not names:
            names, current = ["(no cameras found)"], None
        font = cv2.FONT_HERSHEY_SIMPLEX
        w = max(cv2.getTextSize(n, font, 0.5, 1)[0][0] for n in names) + 60
        x0, y1 = SLIDER_X0, button_y - 4
        y0 = max(0, y1 - 30 * len(names) - 8)
        cv2.rectangle(img, (x0, y0), (x0 + w, y1), (55, 55, 55), -1)
        cv2.rectangle(img, (x0, y0), (x0 + w, y1), (150, 150, 150), 1)
        for i, n in enumerate(names):
            top = y0 + 4 + 30 * i
            cv2.circle(img, (x0 + 18, top + 15), 7, (200, 200, 200), 1, cv2.LINE_AA)
            if i == current:
                cv2.circle(img, (x0 + 18, top + 15), 4, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(img, n, (x0 + 34, top + 20), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            if current is not None:
                self.menu_items.append(((x0, top, x0 + w, top + 30), ("camera", i)))

    def button(self, img, x: int, y: int, label: str, cmd: tuple) -> int:
        """Draws a button and registers it for clicks; returns its right edge."""
        import cv2

        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        rect = (x, y, x + tw + 24, y + 32)
        cv2.rectangle(img, rect[:2], rect[2:], (70, 70, 70), -1)
        cv2.rectangle(img, rect[:2], rect[2:], (120, 120, 120), 1)
        cv2.putText(img, label, (x + 12, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        self.buttons.append((rect, cmd))
        return rect[2]


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


def preview_process(cfg: dict, status_q, cmd_q, hotkey_status: str):
    """Preview window (separate process): draws engine snapshots, sends user input back."""
    import cv2

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C goes to the engine, which closes us
    engine = multiprocessing.parent_process()
    ui = PreviewUI(cfg, cmd_q, hotkey_status)
    accept_first_click()
    win = "X-Plane Head Tracker"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, ui.on_mouse)
    st, frame = None, None
    last_draw, was_focused = 0.0, None
    while (engine is None or engine.is_alive()) and not ui.quit:
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
        focused = app_focused()
        if focused != was_focused:
            cmd_q.put(("focus", focused))
            was_focused = focused
        # Redraw at ~30 fps while you're looking at the preview, ~5 fps otherwise, and only
        # on input while it's switched off.
        interval = 0.0 if focused else 0.2
        due = ui.dirty if not ui.preview_on else time.monotonic() - last_draw >= interval
        if not (due or ui.clicked or ui.drag):  # clicks always get an immediate redraw
            cv2.waitKey(30)
            continue
        last_draw, ui.dirty, ui.clicked = time.monotonic(), False, False
        cv2.imshow(win, ui.draw(st, frame))
        k = cv2.waitKey(16) & 0xFF
        keys = {ord("c"): ("recenter",), ord(" "): ("recenter",), ord("p"): ("pause",), ord("l"): ("lean",),
                ord("v"): ("preview",)}
        if k in keys:
            ui.press(keys[k])
        elif k in (ord("q"), 27):
            ui.press(("quit",))
    cv2.destroyAllWindows()
