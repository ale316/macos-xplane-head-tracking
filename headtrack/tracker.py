"""Camera capture and face tracking: MediaPipe Face Landmarker -> filtered head pose."""
from __future__ import annotations

import threading
import time

import numpy as np

from .config import MODEL_PATH
from .macos import lock_camera_fps, play_sound, set_thread_interactive
from .pose import AdaptiveFilter, euler_yxz, head_delta, orthonormalize, smoothing_k
from .state import Shared, diag

ROTATE_CODES = {90: 0, 180: 1, 270: 2}  # cv2.ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE


class Tracker(threading.Thread):
    def __init__(self, cfg: dict, shared: Shared):
        super().__init__(daemon=True, name="tracker")
        self.cfg, self.shared = cfg, shared
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
        self._probe_hits: dict[int, int] = {}
        self._probe_i, self._probe_next = 0, 0.0
        self._tilted_since = None  # when the tracked face started looking turned on its side

    def set_rotation_mode(self, mode):
        """'auto' or a fixed clockwise rotation (0/90/180/270) for the camera image."""
        self.rotation_mode = mode
        if mode != "auto":
            self._new_rotation = mode

    def _probe_rotation(self, raw_frame, now: float, probe_lm, mp, cv2) -> int | None:
        """While the face is lost, try the other camera orientations one at a time (~10 per second)
        with a separate image-mode model, so the main video tracker's state isn't disturbed.
        A roughly upright face found twice in the same orientation means the camera was turned."""
        if now < self._probe_next:
            return None
        self._probe_next = now + 0.1
        candidates = [r for r in (0, 90, 270, 180) if r != self.rotation]
        r = candidates[self._probe_i % len(candidates)]
        self._probe_i += 1
        img = raw_frame if r == 0 else cv2.rotate(raw_frame, ROTATE_CODES[r])
        res = probe_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
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

    def _check_orientation(self, res, raw_frame, now: float, probe_lm, mp, cv2):
        """Auto camera rotation. A face lying on its side for a second means the camera was turned
        while tracking continued; a lost face may mean it was turned and tracking dropped."""
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
            r = self._probe_rotation(raw_frame, now, probe_lm, mp, cv2)
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
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        cap = cv2.VideoCapture(int(self.cfg["camera_index"]), cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["camera_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["camera_height"])
        if not cap.isOpened():
            with self.shared.lock:
                self.shared.error = ("Could not open camera. Grant camera access to your terminal app in "
                                     "System Settings > Privacy & Security > Camera, or try --camera 1.")
            return

        locked_dev, info = None, ""
        if cap.read()[0]:  # the capture session is running now
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            locked_dev, info = lock_camera_fps(int(self.cfg["camera_index"]), w, h, self.cfg["camera_fps"])
        print(info, flush=True)
        with self.shared.lock:
            self.shared.cam_info = info

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
        last_ts = -1
        fps_t, fps_n = time.monotonic(), 0
        with self.shared.lock:
            self.shared.cam_rotation, self.shared.cam_rotation_mode = self.rotation, self.rotation_mode
        with vision.FaceLandmarker.create_from_options(opts) as lm, \
                vision.FaceLandmarker.create_from_options(probe_opts) as probe_lm:
            while not self.shared.quit:
                ok, raw_frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                now = time.monotonic()
                if self._new_rotation is not None:
                    r, self._new_rotation = self._new_rotation, None
                    self._apply_rotation(r)
                with self.shared.lock:
                    self.shared.cam_rotation_mode = self.rotation_mode
                frame = raw_frame if not self.rotation else cv2.rotate(raw_frame, ROTATE_CODES[self.rotation])
                ts = max(int((now - t0) * 1000), last_ts + 1)
                last_ts = ts
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                infer_ms = (time.monotonic() - now) * 1000
                self._process(frame, res, now)
                if self.rotation_mode == "auto":
                    self._check_orientation(res, raw_frame, now, probe_lm, mp, cv2)
                # frame: when read returned, the camera's own timestamp, inference time
                diag("frame", now, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, infer_ms)
                fps_n += 1
                if now - fps_t >= 1.0:
                    with self.shared.lock:
                        self.shared.cam_fps = fps_n / (now - fps_t)
                    fps_t, fps_n = now, 0
        if locked_dev is not None:
            locked_dev.unlockForConfiguration()
        cap.release()

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
        lms = np.array([[l.x, l.y] for l in res.face_landmarks[0]], dtype=np.float32)

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
        raw = np.zeros(6)
        vel = np.zeros(6)
        d = None
        if self.R_ref is not None and not recentering:
            with sh.lock:
                lost_for = now - sh.face_time
                self.filter.k = smoothing_k(sh.smoothing)
            if lost_for > 0.3:  # don't let the filter interpolate across a gap
                self.filter.reset()
            d = head_delta(self.R_ref, self.p_ref, R, p)
            raw = self.filter(d)
            if self.prev is not None and lost_for <= 0.3 and now - self.prev[1] < 0.1:
                vel = (raw - self.prev[0]) / max(now - self.prev[1], 1e-3)
        self.prev = (raw, now) if d is not None else None

        with sh.lock:
            if not sh.face_ok and now - sh.face_time > 0.3:
                sh.reacquired_time = now
            sh.frame, sh.landmarks = frame, lms
            sh.face_ok, sh.face_time = True, now
            sh.recentering = recentering
            sh.has_ref = self.R_ref is not None
            sh.raw, sh.raw_vel, sh.raw_time = raw, vel, time.monotonic()
            sh.noise = self.filter.noise.copy()
