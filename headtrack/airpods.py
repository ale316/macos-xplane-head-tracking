"""Head rotation from AirPods / Beats head tracking (CoreMotion), instead of the camera's.

Their gyros report a head turn ~70 ms before the camera sees it, and hands or bad light don't
disturb them. The camera keeps doing leaning, corrects the AirPods' slow yaw drift, and takes
over rotation while they're out.

macOS kills a process that reads headphone motion unless the app it was opened as says why in
its Info.plist (NSMotionUsageDescription). X-Plane Head Tracker.app does (see install.sh);
Terminal doesn't, so this needs the tracker opened as the app.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .macos import APP_NAME, running_as_app, set_thread_interactive
from .pose import euler_yxz, rot_x, rot_y, rot_z
from .state import Shared, diag

# No samples for this long means they're out. Sound starting or stopping stalls them ~0.3 s.
STALE_S = 0.5
# The headphones' frame (x right, y forward, z up) as a head frame like MediaPipe's (z back).
TO_HEAD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
STILL_DPS = 5.0  # head turning slower than this counts as still
ANCHOR_DPS = 30.0  # line up with the camera only below this speed (the camera lags)
DRIFT_TAU_S = 10.0  # yaw drift (~2 deg/min) is pulled toward the camera's yaw this slowly
DRIFT_MAX_DPS = 0.3  # and never faster than this, so a camera glitch can't swing the view
DRIFT_WINDOW_DEG = 15.0  # only while looking about straight ahead, where the camera is accurate


def quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def headphone_rotation(R_ref: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Headphone attitude R relative to R_ref as [yaw right+, pitch up+, roll right+] (deg),
    in the head's own frame like pose.head_delta."""
    yaw, pitch, roll = euler_yxz(TO_HEAD @ R_ref.T @ R @ TO_HEAD.T)
    return np.array([-yaw, pitch, -roll])


def relative_attitude(rot: np.ndarray) -> np.ndarray:
    """The inverse of headphone_rotation: R_ref.T @ R for a rotation [yaw, pitch, roll]."""
    return TO_HEAD.T @ rot_y(-rot[0]) @ rot_x(rot[1]) @ rot_z(-rot[2]) @ TO_HEAD


def rate_dps(rate) -> np.ndarray:
    """Headphone rotation rate (rad/s about x, y, z) -> [yaw, pitch, roll] deg/s."""
    return np.degrees([-rate[2], rate[0], rate[1]])


class HeadphoneRotation:
    """Head rotation from headphone samples, lined up with the camera's. It starts out from the
    camera's rotation, and again after a gap or a switch to the other earbud (each has its own
    reference), while the face is tracked and the head about still. Then it slowly pulls its yaw
    toward the camera's while the head is still and near center."""

    def __init__(self):
        self.R = None  # latest attitude
        self.R_ref = None  # attitude at the neutral pose; None until lined up with the camera
        self.rate = np.zeros(3)
        self.sensor = None
        self.t = -1e9  # when the latest sample arrived
        self.rot_t: float | None = None  # when `rot` was last updated
        self.yaw_fix = 0.0  # drift correction (deg)
        self.rot = np.zeros(3)

    def anchor(self, rot: np.ndarray):
        """From now on, the current attitude reads as rotation `rot` (zero: recenter)."""
        self.R_ref = self.R @ relative_attitude(rot).T
        self.yaw_fix = 0.0
        self.rot = np.array(rot, dtype=float)
        self.rot_t = self.t

    def sample(self, now: float, quat, rate, sensor: int, camera: np.ndarray, camera_ok: bool):
        """One sample. `camera` is the camera's rotation, `camera_ok` when it is tracking the face
        undisturbed (no hand at it, not recentering)."""
        dt = now - self.t
        if dt > STALE_S or sensor != self.sensor:
            self.R_ref = None
        self.R, self.rate, self.sensor, self.t = quat_matrix(*quat), rate_dps(rate), sensor, now
        speed = np.abs(self.rate).max()
        if self.R_ref is None:
            if camera_ok and speed < ANCHOR_DPS:
                self.anchor(camera)
            return
        rot = headphone_rotation(self.R_ref, self.R)
        if camera_ok and speed < STILL_DPS and np.abs(camera[:2]).max() < DRIFT_WINDOW_DEG:
            step = (camera[0] - (rot[0] + self.yaw_fix)) * min(dt / DRIFT_TAU_S, 1.0)
            self.yaw_fix += float(np.clip(step, -DRIFT_MAX_DPS * dt, DRIFT_MAX_DPS * dt))
        rot[0] += self.yaw_fix
        self.rot, self.rot_t = rot, now

    def current(self, now: float) -> tuple[np.ndarray, np.ndarray] | None:
        """(rotation in deg, rate in deg/s), or None while there's none to use."""
        if self.R_ref is None or self.rot_t is None or now - self.rot_t > STALE_S:
            return None
        return self.rot.copy(), self.rate.copy()


class HeadphoneMotion(threading.Thread):
    """Reads the AirPods and publishes their rotation in shared.airpods_rot, until stopped. The
    last rotation stays there when they go out; airpods_time tells whether it's still fresh."""

    def __init__(self, shared: Shared):
        super().__init__(daemon=True, name="airpods")
        self.shared = shared
        self.rotation = HeadphoneRotation()
        # ("sample", arrival time, quaternion x/y/z/w, rate x/y/z in rad/s, earbud) or ("status", status, detail)
        self.events: queue.Queue = queue.Queue()
        self.stopping = threading.Event()
        self.manager = None

    def stop(self):
        """Back to the camera's rotation."""
        self.stopping.set()
        self.join(1)

    def _start_updates(self) -> tuple[str, str] | None:
        """Start CoreMotion sending samples to `events`; or the (status, detail) why not."""
        if not running_as_app():  # macOS would kill us
            return "needs the app", f": open {APP_NAME}.app instead of run.sh"
        import CoreMotion
        from Foundation import NSOperationQueue

        manager_class = getattr(CoreMotion, "CMHeadphoneMotionManager", None)
        if manager_class is None:
            return "unsupported", " (needs macOS 14 or newer)"
        if manager_class.authorizationStatus() in (1, 2):  # restricted, denied
            return "not allowed", (f": allow {APP_NAME} under System Settings > Privacy & Security > "
                                   "Motion & Fitness, then restart")
        manager = manager_class.alloc().init()
        if not manager.isDeviceMotionAvailable():
            return "unsupported", " (no headphone motion on this Mac)"

        def on_motion(motion, error):
            if error is not None:
                self.events.put(("status", "error", f": {error.localizedDescription()}"))
                return
            q, r = motion.attitude().quaternion(), motion.rotationRate()
            self.events.put(("sample", time.monotonic(), (q.x, q.y, q.z, q.w), (r.x, r.y, r.z),
                             motion.sensorLocation()))

        manager.startDeviceMotionUpdatesToQueue_withHandler_(NSOperationQueue.alloc().init(), on_motion)
        self.manager = manager
        return None

    def _status(self, status: str, detail: str = ""):
        with self.shared.lock:
            changed, self.shared.airpods_status = self.shared.airpods_status != status, status
        if changed:
            print(f"AirPods: {status}{detail}", flush=True)

    def run(self):
        set_thread_interactive()
        self._status("starting")
        problem = self._start_updates()  # or what went wrong since, until samples arrive
        while not self.shared.quit and not self.stopping.is_set():
            try:
                event = self.events.get(timeout=0.1)
            except queue.Empty:
                event = None
            if event is not None and event[0] == "status":
                problem = event[1:]
            elif event is not None:
                problem = None
                self._sample(*event[1:])
            now = time.monotonic()
            with self.shared.lock:
                recenter = self.shared.airpods_recenter and self.rotation.R is not None
                if recenter:
                    self.shared.airpods_recenter = False
            if recenter:
                self.rotation.anchor(np.zeros(3))
            current = self.rotation.current(now)
            rot = self.rotation
            with self.shared.lock:  # the last rotation stays, to hold the view if they go out
                self.shared.airpods_rot = None if rot.rot_t is None else rot.rot.copy()
                self.shared.airpods_vel = np.zeros(3) if current is None else current[1]
                self.shared.airpods_time = rot.rot_t or 0.0
            self._status(*(problem or ["on" if current is not None else "waiting"]))
        if self.manager is not None:
            self.manager.stopDeviceMotionUpdates()
        with self.shared.lock:
            self.shared.airpods_status, self.shared.airpods_rot = "", None

    def _sample(self, now: float, quat, rate, sensor: int):
        sh = self.shared
        with sh.lock:
            camera = sh.raw[:3].copy()
            camera_ok = sh.face_ok and sh.has_ref and not sh.hand_at_face and not sh.recentering
            recentering = sh.recentering
        self.rotation.sample(now, quat, rate, sensor, camera, camera_ok)
        if recentering:  # the camera's neutral pose is being taken: it's ours too
            self.rotation.anchor(np.zeros(3))
        diag("airpods", *self.rotation.rot)
