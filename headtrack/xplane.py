"""X-Plane link over its built-in UDP interface (no plugin needed).

DREF packets write sim/graphics/view/pilots_head_* at a fixed rate; RREF subscriptions read
the view type, field of view and flight time. The seat position comes from the loaded
aircraft's .acf file (see EyePoint).
"""
from __future__ import annotations

import math
import re
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

from .config import AXES
from .macos import set_thread_interactive
from .pose import fill_gap, map_pose, smoothing_ease, smoothing_fill
from .state import Shared, diag

DATAREFS = {
    "yaw": "sim/graphics/view/pilots_head_psi",
    "pitch": "sim/graphics/view/pilots_head_the",
    "roll": "sim/graphics/view/pilots_head_phi",
    "x": "sim/graphics/view/pilots_head_x",
    "y": "sim/graphics/view/pilots_head_y",
    "z": "sim/graphics/view/pilots_head_z",
}
VIEW_TYPE_DREF = "sim/graphics/view/view_type"
VIEW_3D_COCKPIT = 1026
FOV_DREF = "sim/graphics/view/field_of_view_deg"
FLIGHT_TIME_DREF = "sim/time/total_flight_time_sec"  # drops when a new flight starts
FEET = 0.3048
MAX_BASE_PITCH = 15.0  # X-Plane's default view looks a little down (e.g. -6 deg); more is a panned view


def xplane_folder() -> Path | None:
    """The folder of the X-Plane that's running on this Mac."""
    try:
        from AppKit import NSWorkspace
    except ImportError:
        return None
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if (app.localizedName() or "").startswith("X-Plane") and app.bundleURL() is not None:
            return Path(app.bundleURL().path()).parent
    return None


def acf_eye_point(acf_text: str) -> np.ndarray | None:
    """The designed pilot eye point from an .acf file, in X-Plane's pilots_head_x/y/z frame
    (meters from the reference CG; the file has feet from the aircraft's reference point)."""
    v = dict(re.findall(r"^P acf/(_pe_xyz/[012]|_cgX_lft|_cgX_rgt|_cgY|_cgZ) (\S+)$", acf_text, re.M))
    try:
        cg_x = (float(v["_cgX_lft"]) + float(v["_cgX_rgt"])) / 2
        return np.array([float(v["_pe_xyz/0"]) - cg_x, float(v["_pe_xyz/1"]) - float(v["_cgY"]),
                         float(v["_pe_xyz/2"]) - float(v["_cgZ"])]) * FEET
    except (KeyError, ValueError):
        return None


class EyePoint:
    """The current aircraft's seat position, from the .acf file X-Plane's log says it loaded.

    X-Plane keeps the eye point and the live head position in the same variable, so the
    tracker's own writes move it and it can't be read back as a reference. The file stays put.
    """

    def __init__(self, folder: Path | None):
        self.folder = folder
        self.log_read = 0  # bytes of Log.txt already looked at (it only grows, until X-Plane restarts)
        self.acf = None
        self.point: np.ndarray | None = None

    def refresh(self) -> np.ndarray | None:
        if self.folder is None:
            return None
        try:
            with open(self.folder / "Log.txt", "rb") as log:
                size = log.seek(0, 2)
                if size < self.log_read:  # a new log: X-Plane restarted
                    self.log_read = 0
                log.seek(max(0, self.log_read - 200))  # a little overlap, for a line split across reads
                text = log.read().decode(errors="replace")
                self.log_read = size
            loads = re.findall(r"Loading airplane number 0 with (.+?\.acf)", text)
            if loads and loads[-1] != self.acf:
                self.acf = loads[-1]
                self.point = acf_eye_point((self.folder / self.acf).read_text(errors="replace"))
                print(f"aircraft: {Path(self.acf).stem}, seat at {np.round(self.point, 3)} m", flush=True)
        except OSError:
            pass
        return self.point


# Our head convention is yaw right+, pitch up+, roll right+, x right+, y up+, z forward+.
# X-Plane's aircraft frame has +z pointing aft, so leaning forward means decreasing z.
XPLANE_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0])


class XPlaneLink(threading.Thread):
    """Talks to X-Plane's native UDP interface.

    The view is written as base + tracked offset at send_rate_hz. The base is the aircraft's
    seat position (from its .acf file when X-Plane runs on this Mac) looking straight ahead;
    RREF subscriptions read the view type, field of view and X-Plane's default view pitch.
    """

    SETTLE_S = 0.3  # wait this long after our last write before trusting X-Plane's values as base

    def __init__(self, cfg: dict, shared: Shared):
        super().__init__(daemon=True, name="xplane")
        self.cfg, self.shared = cfg, shared
        self.addr = (cfg["xplane_host"], int(cfg["xplane_port"]))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.setblocking(False)
        self.names = [DATAREFS[a] for a in AXES] + [VIEW_TYPE_DREF, FOV_DREF, FLIGHT_TIME_DREF]
        self.local = cfg["xplane_host"] in ("127.0.0.1", "localhost")
        self.eye = EyePoint(xplane_folder() if self.local else None)
        self.next_eye_check = 0.0
        self.flight_time = None
        self.values: dict[int, tuple[float, float]] = {}  # index -> (value, rx time)
        self.last_rx = 0.0
        self.last_sent = 0.0
        self.base: np.ndarray | None = None
        self.prev_base: np.ndarray | None = None
        self.prev_out: np.ndarray | None = None
        self.smoothed = np.zeros(6)
        self.lean_gain = 1.0 if cfg["lean_enabled"] else 0.0
        self.was_active = False

    def _send(self, data: bytes):
        try:
            self.sock.sendto(data, self.addr)
        except OSError:
            pass

    def _subscribe(self, freq: int):
        for i, name in enumerate(self.names):
            self._send(b"RREF\x00" + struct.pack("<ii400s", freq, i, name.encode()))

    def _dref(self, name: str, value: float):
        self._send(b"DREF\x00" + struct.pack("<f500s", float(value), name.encode()))

    def _write_pose(self, pose: np.ndarray):
        for i, a in enumerate(AXES):
            if self.cfg["axes"][a]["enabled"]:
                self._dref(DATAREFS[a], pose[i])
        self.last_sent = time.monotonic()

    def _recv(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(8192)
            except BlockingIOError:
                return
            except OSError:  # e.g. ECONNREFUSED echoed back while X-Plane isn't running
                return
            if data[:4] != b"RREF":
                continue
            now = time.monotonic()
            self.last_rx = now
            payload = data[5:]
            for off in range(0, len(payload) - 7, 8):
                idx, val = struct.unpack_from("<if", payload, off)
                if 0 <= idx < len(self.names):
                    self.values[idx] = (val, now)

    def _follow_aircraft(self, now: float, connected: bool):
        """Keep the base on the current aircraft's seat; a new flight takes a new base."""
        if now >= self.next_eye_check:
            self.next_eye_check = now + 1.0
            if self.eye.folder is None and self.local and connected:
                self.eye.folder = xplane_folder()  # X-Plane was started after the tracker
            self.eye.refresh()
        flight = self.values.get(len(AXES) + 2)
        if flight is not None:
            if self.flight_time is not None and flight[0] < self.flight_time - 1.0:
                self.base = self.prev_out = None  # a new flight: take the new seat position
            self.flight_time = flight[0]
        if self.base is not None and self.eye.point is not None:
            self.base[3:] = self.eye.point

    def _capture_base(self):
        """Take the base from X-Plane's own head pose, once our writes have stopped echoing back:
        straight ahead and level at X-Plane's default view pitch, in the seat."""
        head = [self.values.get(i) for i in range(len(AXES))]
        if not all(h and h[1] > self.last_sent + self.SETTLE_S for h in head):
            return
        captured = np.array([h[0] for h in head])
        # If X-Plane is still showing our last written pose (e.g. after a view switch
        # round-trip), keep the old base instead of baking in the offset.
        if self.prev_out is not None and np.allclose(captured, self.prev_out, atol=1e-3):
            self.base = self.prev_base
        else:
            pitch = captured[1] if abs(captured[1]) <= MAX_BASE_PITCH else 0.0
            self.base = np.array([0.0, pitch, 0.0, *captured[3:]])
        if self.eye.point is not None:
            self.base[3:] = self.eye.point  # the seat position from the aircraft file
        self.smoothed = np.zeros(6)

    def run(self):
        cfg, sh = self.cfg, self.shared
        set_thread_interactive()
        period = 1.0 / cfg["send_rate_hz"]
        next_tick = time.monotonic()
        next_sub = 0.0
        last = time.monotonic()
        while True:
            now = time.monotonic()
            dt, last = now - last, now
            if now >= next_sub:
                self._subscribe(20)  # re-sent periodically in case X-Plane was restarted
                next_sub = now + 5.0
            self._recv()

            with sh.lock:
                quitting, paused = sh.quit, sh.paused
                has_ref, face_ok, reacquired, lean = sh.has_ref, sh.face_ok, sh.reacquired_time, sh.lean
                raw, vel, raw_time = sh.raw.copy(), sh.raw_vel.copy(), sh.raw_time
                smoothing = sh.smoothing
            if cfg["fill_between_frames"]:
                raw = fill_gap(raw, vel, now - raw_time, smoothing_fill(smoothing))

            # Leaning fades in/out over ~0.25 s when toggled instead of jumping.
            self.lean_gain += ((1.0 if lean else 0.0) - self.lean_gain) * (1.0 - math.exp(-dt / 0.25))
            target = map_pose(cfg, raw, self.lean_gain)

            connected = now - self.last_rx < 2.0
            view = self.values.get(len(AXES))
            view_type = int(round(view[0])) if view else None
            fov = self.values.get(len(AXES) + 1)
            in_cockpit = (not cfg["only_in_cockpit_view"]) or view_type in (None, VIEW_3D_COCKPIT)
            want_active = connected and in_cockpit and not paused and has_ref and not quitting

            self._follow_aircraft(now, connected)
            if want_active and self.base is None:
                self._capture_base()
            active = want_active and self.base is not None

            if active:
                slow = (not face_ok) or (now - reacquired < 0.5)
                tau = cfg["face_lost_smoothing_s"] if slow else smoothing_ease(cfg["output_smoothing_s"], smoothing)
                self.smoothed += (target - self.smoothed) * (1.0 - math.exp(-dt / max(tau, 1e-3)))
                out = self.base + self.smoothed * XPLANE_SIGN
                self._write_pose(out)
                self.prev_base, self.prev_out = self.base, out
            elif self.was_active:
                if (paused or quitting) and connected:
                    self._write_pose(self.base)  # hand the view back where we found it
                    self.prev_out = self.base
                self.base = None
            self.was_active = active

            with sh.lock:
                sh.xp_connected, sh.xp_view, sh.xp_active = connected, view_type, active
                sh.xp_fov = fov[0] if fov and connected else None
                sh.target = target

            diag("tick", float(active), dt * 1000)
            if quitting:
                break
            # Fixed-rate schedule; if we fell behind, restart from now rather than bursting.
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
        self._subscribe(0)
        self.sock.close()
