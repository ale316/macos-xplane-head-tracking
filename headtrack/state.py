"""State shared between the tracking threads, and the --diag timing log."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

_diag = None  # (file, lock) when running with --diag


def enable_diag(path: Path):
    global _diag
    _diag = (open(path, "w", buffering=1 << 16), threading.Lock())
    _diag[0].write("kind,t,values...\n")


def close_diag():
    global _diag
    if _diag:
        _diag[0].close()
        _diag = None


def diag(kind: str, *values: float):
    """Append an event (time.monotonic() clock) and its values to the --diag CSV, if enabled."""
    if _diag:
        f, lock = _diag
        with lock:
            f.write(f"{kind},{time.monotonic():.5f}," + ",".join(f"{v:.5f}" for v in values) + "\n")


class Shared:
    """Everything the tracker, X-Plane link, engine loop and command handler share (use `lock`)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.quit = False
        self.paused = False
        self.error: str | None = None
        # written by the tracker
        self.frame = None
        self.landmarks = None
        self.face_ok = False
        self.face_time = 0.0
        self.reacquired_time = 0.0
        self.has_ref = False
        self.recentering = False
        self.hand_at_face = False  # a hand is in front of the face: tracking is steadied / held
        self.hand_boxes: list = []  # hands seen, normalized (x0, y0, x1, y1), for the preview
        self.raw = np.zeros(6)  # filtered head pose relative to neutral
        self.noise = np.zeros(6)  # measured camera noise per axis (deg / cm)
        self.raw_vel = np.zeros(6)  # velocity of `raw` between the last two frames
        self.raw_time = 0.0  # when `raw` became available
        self.cam_fps = 0.0
        self.cam_info = ""
        self.cam_target_fps = 0
        self.cam_id = ""
        self.cam_name = ""
        self.cam_list: list = []  # CameraInfo of the cameras that can be picked
        self.cam_rotation = 0  # rotation currently applied to camera images
        self.cam_rotation_mode = "auto"
        # settings changed from the preview / hotkeys
        self.lean = True
        self.smoothing = 40.0  # 0-100
        self.preview_visible = True
        self.preview_on = True
        # written by the X-Plane link
        self.target = np.zeros(6)  # view offset sent on top of X-Plane's own head position
        self.xp_connected = False
        self.xp_view: int | None = None
        self.xp_active = False
        self.xp_fov: float | None = None  # X-Plane's horizontal field of view
