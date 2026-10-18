"""Camera capture and face tracking: MediaPipe Face Landmarker -> filtered head pose."""
from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from .camera import Camera, CameraInfo, camera_access
from .config import HAND_MODEL_PATH, MODEL_PATH
from .macos import play_sound, set_thread_interactive
from .pose import AdaptiveFilter, euler_yxz, head_delta, orthonormalize, smoothing_k
from .state import Shared, diag

ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# While a hand is at the face ("follow" mode): filter this much stiffer, and limit the speed.
HAND_FILTER_FACTOR = 4.0
HAND_MAX_SPEED = np.array([60.0, 60.0, 60.0, 10.0, 10.0, 10.0])  # deg/s, cm/s


def rotate(frame, degrees: int):
    """The frame turned clockwise by 0/90/180/270 degrees."""
    return cv2.rotate(frame, ROTATE[degrees]) if degrees else frame


def mp_image(frame):
    """A MediaPipe image from an OpenCV (BGR) frame."""
    import mediapipe as mp  # imported when needed: it takes a while
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def box(landmarks) -> tuple[float, float, float, float]:
    """Bounding box (x0, y0, x1, y1) of MediaPipe landmarks, in normalized image coordinates."""
    pts = np.array([[p.x, p.y] for p in landmarks])
    return (*pts.min(0), *pts.max(0))


def hand_over_face(hand_boxes: list, face_box, margin: float = 0.15) -> bool:
    """Does any hand box overlap the face box grown by `margin` (fraction of its size)?"""
    if face_box is None:
        return False
    fx0, fy0, fx1, fy1 = face_box
    mx, my = (fx1 - fx0) * margin, (fy1 - fy0) * margin
    return any(hx0 < fx1 + mx and hx1 > fx0 - mx and hy0 < fy1 + my and hy1 > fy0 - my
               for hx0, hy0, hx1, hy1 in hand_boxes)


class HandWatcher(threading.Thread):
    """Looks for hands in front of the face, about 15 times a second alongside the tracking.

    A hand over the mouth or chin usually doesn't make MediaPipe lose the face: it reports
    the face with a wrong pose (up to 15 deg off in tests), which throws the view around.
    So while a hand overlaps the face the tracker steadies the view (or holds it).
    """

    HOLD_S = 0.35  # keep holding this long after the hand was last seen at the face

    def __init__(self, model_path, rate_hz: float):
        super().__init__(daemon=True, name="hands")
        self.model_path, self.period = model_path, 1.0 / rate_hz
        self._cond = threading.Condition()
        self._job = None
        self._stopped = False
        self.last_hit = -1e9  # capture time of the latest frame with a hand at the face
        self.onset = 0.0  # capture time of the first such frame in the current streak
        self.boxes: list = []

    def submit(self, frame, t: float, face_box):
        with self._cond:
            self._job = (frame, t, face_box)
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify()

    def at_face(self, now: float) -> bool:
        return now - self.last_hit < self.HOLD_S

    def check(self, boxes: list, face_box, t: float):
        """Record the hands found in the frame captured at time t."""
        self.boxes = boxes
        if hand_over_face(boxes, face_box):
            if t - self.last_hit >= self.HOLD_S:
                self.onset = t
            self.last_hit = t

    def run(self):
        from mediapipe.tasks.python import BaseOptions, vision

        set_thread_interactive()
        opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(self.model_path)),
                                            running_mode=vision.RunningMode.VIDEO, num_hands=2)
        last_ts = -1
        with vision.HandLandmarker.create_from_options(opts) as lm:
            while True:
                with self._cond:
                    while self._job is None and not self._stopped:
                        self._cond.wait(0.5)
                    if self._stopped:
                        return
                    (frame, t, face_box), self._job = self._job, None
                start = time.monotonic()
                last_ts = ts = max(int(t * 1000), last_ts + 1)
                res = lm.detect_for_video(mp_image(frame), ts)
                self.check([box(hand) for hand in res.hand_landmarks], face_box, t)
                time.sleep(max(0.0, self.period - (time.monotonic() - start)))


class Tracker(threading.Thread):
    """Reads the camera and turns each frame into a filtered head pose relative to the neutral
    (recentered) pose, published in `shared.raw`. Also handles camera switching and rotation,
    and steadies the pose while a hand is at the face."""

    def __init__(self, cfg: dict, shared: Shared, camera: CameraInfo | None = None):
        super().__init__(daemon=True, name="tracker")
        self.cfg, self.shared, self.camera = cfg, shared, camera
        self.prev = None  # (filtered pose, capture time) of the previous frame
        self.filter = AdaptiveFilter(smoothing_k(40), floor=np.array([0.05] * 3 + [0.03] * 3))
        self.pivot = np.array(cfg["translation_pivot_cm"], dtype=float)
        self.R_ref = None
        self.p_ref = None
        self._recenter_samples: list | None = []  # recenter on the first frames
        self._lock = threading.Lock()
        mode = cfg["camera_rotation"]
        self.rotation_mode = "auto" if mode == "auto" else int(mode) % 360
        self.rotation = 0 if mode == "auto" else self.rotation_mode
        self._new_rotation = None  # set from other threads, applied by the tracker thread
        self._new_camera = None
        self._probe_hits: dict[int, int] = {}
        self._probe_i, self._probe_next = 0, 0.0
        self._tilted_since = None  # when the tracked face started looking turned on its side
        self.hands: HandWatcher | None = None
        self._held = None  # output pose while a hand is at the face
        self._held_t = 0.0
        self._history = deque(maxlen=90)  # recent (time, pose), to rewind past a hand's first effects

    def request_camera(self, camera: CameraInfo):
        """Switch to another camera while running."""
        self._new_camera = camera

    def _open_camera(self, camera: CameraInfo | None) -> Camera | None:
        if camera is None:
            return None
        try:
            cam = Camera(camera.id, self.cfg["camera_width"], self.cfg["camera_height"], self.cfg["camera_fps"])
        except Exception as e:  # AVFoundation errors arrive as various exception types
            print(f"camera {camera.name}: {e}", flush=True)
            with self.shared.lock:
                self.shared.error = f"camera {camera.name}: {e}"
            return None
        print(f"camera: {cam.name}, {cam.mode}", flush=True)
        with self.shared.lock:
            self.shared.cam_id, self.shared.cam_name = camera.id, cam.name
            self.shared.cam_locked = cam.locked
            self.shared.cam_target_fps = cam.fps
            self.shared.error = None
        return cam

    def _wait_for_camera(self) -> Camera | None:
        """Open the chosen camera, retrying every 2 s (or right away with a newly picked one)
        until it works, e.g. while an iPhone reconnects."""
        while not self.shared.quit:
            if self._new_camera is not None:
                self.camera, self._new_camera = self._new_camera, None
            cam = self._open_camera(self.camera)
            if cam is not None:
                return cam
            with self.shared.lock:
                self.shared.error = self.shared.error or "Could not open the camera. Pick another camera."
            for _ in range(10):
                if self._new_camera is not None or self.shared.quit:
                    break
                time.sleep(0.2)
        return None

    def set_rotation_mode(self, mode):
        """'auto' or a fixed clockwise rotation (0/90/180/270) for the camera image."""
        self.rotation_mode = mode
        if mode != "auto":
            self._new_rotation = mode

    def _probe_rotation(self, raw_frame, now: float, probe) -> int | None:
        """While the face is lost, try the other camera orientations one at a time (~10 per second)
        with a separate image-mode model, so the main video tracker's state isn't disturbed.
        A roughly upright face found twice in the same orientation means the camera was turned."""
        if now < self._probe_next:
            return None
        self._probe_next = now + 0.1
        candidates = [r for r in (0, 90, 270, 180) if r != self.rotation]
        r = candidates[self._probe_i % len(candidates)]
        self._probe_i += 1
        res = probe(rotate(raw_frame, r))
        if not res.facial_transformation_matrixes or abs(self._image_roll(res)) > 45:
            self._probe_hits[r] = 0  # no face, or one still lying on its side in this orientation
            return None
        self._probe_hits[r] = self._probe_hits.get(r, 0) + 1
        return r if self._probe_hits[r] >= 2 else None

    @staticmethod
    def _image_roll(res) -> float:
        """How far the detected face is turned in the image (degrees, clockwise negative)."""
        M = np.asarray(res.facial_transformation_matrixes[0], dtype=float).reshape(4, 4)
        return euler_yxz(orthonormalize(M[:3, :3]))[2]

    def _check_orientation(self, res, raw_frame, now: float, probe):
        """Auto camera rotation. A face lying on its side for a second means the camera was turned
        while tracking continued; a lost face may mean it was turned and tracking dropped.
        `probe(image)` runs face detection on a single image."""
        if res.facial_transformation_matrixes:
            roll = self._image_roll(res)
            if abs(roll) < 60:
                self._tilted_since = None
            elif self._tilted_since is None:
                self._tilted_since = now
            elif now - self._tilted_since > 1.0:
                self._tilted_since = None
                self._apply_rotation((self.rotation + 90 * round(roll / 90)) % 360)
            return
        self._tilted_since = None
        with self.shared.lock:
            lost_for = now - self.shared.face_time
        if lost_for > 0.5:
            r = self._probe_rotation(raw_frame, now, probe)
            if r is not None:
                self._apply_rotation(r)

    def _apply_rotation(self, r: int):
        self.rotation = r
        self._probe_hits.clear()
        self.filter.reset()
        self.prev = None
        with self.shared.lock:
            self.shared.cam_rotation = r
        print(f"camera orientation: rotating image {r} deg", flush=True)
        self.request_recenter()  # a turned camera means a new neutral pose

    def _pose_before(self, t: float) -> np.ndarray:
        """The last pose from before time t (or the oldest one kept)."""
        before = [pose for when, pose in self._history if when <= t]
        if before:
            return before[-1].copy()
        return self._history[0][1].copy() if self._history else np.zeros(6)

    def _hand_pose(self, d: np.ndarray, now: float) -> np.ndarray:
        """The pose while a hand is at the face, given the measured (unreliable) pose d.
        "hold" keeps the pose from just before the hand arrived. "follow" keeps following with a
        much stiffer filter and a speed limit: they smooth out the jumps a hand causes, while
        slower head movement still comes through."""
        mode = self.cfg["hand_at_face"]
        if self._held is None:  # a hand just arrived
            if mode == "hold":  # rewind past the hand's first effects
                self._held = self._pose_before(self.hands.onset - 0.15)
            else:  # carry on from the current pose (rewinding would snap back if you were moving)
                self._held = self._history[-1][1].copy() if self._history else np.zeros(6)
            self._held_t = now
            diag("hand", 1.0)
        if mode == "follow":
            self.filter.k *= HAND_FILTER_FACTOR
            target = self.filter(d)
            step = HAND_MAX_SPEED * max(now - self._held_t, 0.0)
            self._held = self._held + np.clip(target - self._held, -step, step)
            self._held_t = now
        return self._held.copy()

    def request_recenter(self):
        with self._lock:
            self._recenter_samples = []
        play_sound(self.cfg, "Tink")

    def run(self):
        set_thread_interactive()
        try:
            self._run()
        except Exception as e:  # surface errors in the UI instead of dying silently
            with self.shared.lock:
                self.shared.error = f"tracker: {e!r}"
            raise

    def _run(self):
        from mediapipe.tasks.python import BaseOptions, vision

        if not camera_access():
            with self.shared.lock:
                self.shared.error = ("No camera access. Allow your terminal app (e.g. Terminal) under System "
                                     "Settings > Privacy & Security > Camera, then start again.")
            return
        cam = self._wait_for_camera()
        if cam is None:
            return
        if self.cfg["hand_at_face"] != "off" and HAND_MODEL_PATH.exists():
            self.hands = HandWatcher(HAND_MODEL_PATH, self.cfg["hand_check_hz"])
            self.hands.start()

        opts = vision.FaceLandmarkerOptions(
            # CPU only: MediaPipe's Metal (GPU) path fails to detect faces at webcam resolution.
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        probe_opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                                                  output_facial_transformation_matrixes=True)
        t0 = time.monotonic()
        last_ts, misses = -1, 0
        fps_t, fps_n = time.monotonic(), 0
        with self.shared.lock:
            self.shared.cam_rotation, self.shared.cam_rotation_mode = self.rotation, self.rotation_mode
        with vision.FaceLandmarker.create_from_options(opts) as lm, \
                vision.FaceLandmarker.create_from_options(probe_opts) as probe_lm:
            while not self.shared.quit:
                if self._new_camera is not None:
                    cam.close()
                    cam = self._wait_for_camera()
                    if cam is None:
                        break
                    self.filter.reset()
                    self.prev = None
                    self.request_recenter()  # a different camera means a new neutral pose
                raw_frame, captured = cam.read()
                if raw_frame is None:
                    misses += 1
                    if misses >= 2:  # no frames for 2 s: unplugged, out of range or asleep; reopen it
                        print("camera: no frames, reopening", flush=True)
                        cam.close()
                        cam, misses = self._wait_for_camera(), 0
                        if cam is None:
                            break
                    continue
                misses = 0
                now = time.monotonic()
                if self._new_rotation is not None:
                    r, self._new_rotation = self._new_rotation, None
                    self._apply_rotation(r)
                with self.shared.lock:
                    self.shared.cam_rotation_mode = self.rotation_mode
                frame = rotate(raw_frame, self.rotation)
                last_ts = ts = max(int((now - t0) * 1000), last_ts + 1)
                res = lm.detect_for_video(mp_image(frame), ts)
                infer_ms = (time.monotonic() - now) * 1000
                self._process(frame, res, now)
                if self.hands is not None:
                    self.hands.submit(frame, now, box(res.face_landmarks[0]) if res.face_landmarks else None)
                if self.rotation_mode == "auto":
                    self._check_orientation(res, raw_frame, now, lambda img: probe_lm.detect(mp_image(img)))
                # frame: when read returned, the camera's own timestamp, inference time
                diag("frame", now, captured, infer_ms)
                fps_n += 1
                if now - fps_t >= 1.0:
                    with self.shared.lock:
                        self.shared.cam_fps = fps_n / (now - fps_t)
                    fps_t, fps_n = now, 0
        if self.hands is not None:
            self.hands.stop()
        if cam is not None:
            cam.close()

    def _process(self, frame, res, now: float):
        sh = self.shared
        if not res.facial_transformation_matrixes:
            with sh.lock:
                sh.frame, sh.landmarks = frame, None
                if sh.face_ok:
                    sh.face_ok = False
                sh.raw_vel = np.zeros(6)
                if now - sh.face_time > self.cfg["face_lost_timeout_s"]:
                    sh.raw = np.zeros(6)
            return

        M = np.asarray(res.facial_transformation_matrixes[0], dtype=float).reshape(4, 4)
        R = orthonormalize(M[:3, :3])
        p = R @ self.pivot + M[:3, 3]
        lms = np.array([[pt.x, pt.y] for pt in res.face_landmarks[0]], dtype=np.float32)

        recentering = False
        with self._lock:
            if self._recenter_samples is not None:
                self._recenter_samples.append((R, p))
                recentering = True
                if len(self._recenter_samples) >= self.cfg["recenter_frames"]:
                    self.R_ref = orthonormalize(sum(s[0] for s in self._recenter_samples))
                    self.p_ref = np.mean([s[1] for s in self._recenter_samples], axis=0)
                    self._recenter_samples = None
                    self.filter.reset()

        # Recentering eases the view back to center while samples are collected.
        raw, vel, d = np.zeros(6), np.zeros(6), None
        if self.R_ref is not None and not recentering:
            with sh.lock:
                lost_for = now - sh.face_time
                self.filter.k = smoothing_k(sh.smoothing)
            if lost_for > 0.3:  # don't let the filter interpolate across a gap
                self.filter.reset()
            d = head_delta(self.R_ref, self.p_ref, R, p)
            diag("pose", *d[:3])  # unfiltered, for tuning on real recordings
            if self.hands is not None and self.hands.at_face(now):
                raw = self._hand_pose(d, now)
                d = None  # no velocity / gap filling meanwhile
            else:
                if self._held is not None:  # hand gone: back to normal, easing over
                    self._held = None
                    diag("hand", 0.0)
                    self.filter.reset()
                    with sh.lock:
                        sh.reacquired_time = now
                raw = self.filter(d)
                if self.prev is not None and lost_for <= 0.3 and now - self.prev[1] < 0.1:
                    vel = (raw - self.prev[0]) / max(now - self.prev[1], 1e-3)
            self._history.append((now, raw))
            diag("out", *raw[:3])
        self.prev = (raw, now) if d is not None else None

        with sh.lock:
            if not sh.face_ok and now - sh.face_time > 0.3:
                sh.reacquired_time = now
            sh.frame, sh.landmarks = frame, lms
            sh.face_ok, sh.face_time = True, now
            sh.recentering = recentering
            sh.hand_at_face = self._held is not None
            sh.hand_boxes = self.hands.boxes if self.hands is not None else []
            sh.has_ref = self.R_ref is not None
            sh.raw, sh.raw_vel, sh.raw_time = raw, vel, time.monotonic()
            sh.noise = self.filter.noise.copy()
