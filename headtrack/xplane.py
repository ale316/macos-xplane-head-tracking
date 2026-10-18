"""X-Plane link over its built-in UDP interface (no plugin needed).

RREF subscriptions read X-Plane's own head position, view type and field of view; DREF
packets write sim/graphics/view/pilots_head_* at a fixed rate.
"""
from __future__ import annotations

import math
import socket
import struct
import threading
import time

import numpy as np

from .config import AXES
from .macos import set_thread_interactive
from .pose import fill_gap, map_pose
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
# Our head convention is yaw right+, pitch up+, roll right+, x right+, y up+, z forward+.
# X-Plane's aircraft frame has +z pointing aft, so leaning forward means decreasing z.
XPLANE_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0])


class XPlaneLink(threading.Thread):
    """Talks to X-Plane's native UDP interface.

    RREF subscriptions read X-Plane's own head pose (our "base", e.g. the default eye
    point of the current aircraft) and the current view type; DREF packets write
    base + tracked offset back at send_rate_hz.
    """

    SETTLE_S = 0.3  # wait this long after our last write before trusting X-Plane's values as base

    def __init__(self, cfg: dict, shared: Shared):
        super().__init__(daemon=True, name="xplane")
        self.cfg, self.shared = cfg, shared
        self.addr = (cfg["xplane_host"], int(cfg["xplane_port"]))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.setblocking(False)
        self.names = [DATAREFS[a] for a in AXES] + [VIEW_TYPE_DREF, FOV_DREF]
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
            if cfg["fill_between_frames"]:
                raw = fill_gap(raw, vel, now - raw_time)

            # Leaning fades in/out over ~0.25 s when toggled instead of jumping.
            self.lean_gain += ((1.0 if lean else 0.0) - self.lean_gain) * (1.0 - math.exp(-dt / 0.25))
            target = map_pose(cfg, raw, self.lean_gain)

            connected = now - self.last_rx < 2.0
            view = self.values.get(len(AXES))
            view_type = int(round(view[0])) if view else None
            fov = self.values.get(len(AXES) + 1)
            in_cockpit = (not cfg["only_in_cockpit_view"]) or view_type in (None, VIEW_3D_COCKPIT)
            want_active = connected and in_cockpit and not paused and has_ref and not quitting

            if want_active and self.base is None:
                head = [self.values.get(i) for i in range(len(AXES))]
                if all(h and h[1] > self.last_sent + self.SETTLE_S for h in head):
                    captured = np.array([h[0] for h in head])
                    # If X-Plane is still showing our last written pose (e.g. after a view
                    # switch round-trip), keep the old base instead of baking in the offset.
                    stale = self.prev_out is not None and np.allclose(captured, self.prev_out, atol=1e-3)
                    self.base = self.prev_base if stale else captured
                    self.smoothed = np.zeros(6)
            active = want_active and self.base is not None

            out = None
            if active:
                slow = (not face_ok) or (now - reacquired < 0.5)
                tau = cfg["face_lost_smoothing_s"] if slow else cfg["output_smoothing_s"]
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
