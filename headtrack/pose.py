"""Head-pose maths: pose relative to the neutral pose, response curves, noise-adaptive filter."""
from __future__ import annotations

import math

import numpy as np

from .config import AXES


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose R = Ry(yaw) @ Rx(pitch) @ Rz(roll); returns degrees."""
    pitch = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    yaw = math.atan2(R[0, 2], R[2, 2])
    roll = math.atan2(R[1, 0], R[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def head_delta(R_ref: np.ndarray, p_ref: np.ndarray, R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Head motion relative to the neutral pose, expressed in the neutral head's own frame.

    Because everything is measured relative to the head's reference orientation (not the
    camera's axes), a camera sitting off to the side or below the face still yields clean
    yaw/pitch/roll.

    MediaPipe face frame: +x = subject's left, +y = up, +z = out of the face.
    Returns [yaw right+, pitch up+, roll right+ (deg), x right+, y up+, z forward+ (cm)].
    """
    yaw, pitch, roll = euler_yxz(R_ref.T @ R)
    d = R_ref.T @ (p - p_ref)
    return np.array([-yaw, -pitch, roll, -d[0], d[1], d[2]])


def apply_curve(v: float, ax: dict) -> float:
    """Head movement -> view movement for one axis: nothing inside the deadzone, then rising
    along a power curve to output_max at input_max (and capped there)."""
    if not ax["enabled"]:
        return 0.0
    a = abs(v) - ax["deadzone"]
    if a <= 0:
        return 0.0
    span = max(1e-6, ax["input_max"] - ax["deadzone"])
    out = ax["output_max"] * min(1.0, a / span) ** ax["exponent"]
    sign = -1.0 if v < 0 else 1.0
    return -sign * out if ax["invert"] else sign * out


class AdaptiveFilter:
    """Error-driven smoother with a noise-adaptive threshold (in the spirit of opentrack's Accela).

    Each frame the output moves toward the measurement by the fraction u^3 / (1 + u^3) of the
    error, where u = |error| / (k * noise). Errors at the camera's noise level barely move it,
    so it holds steady; real head motion quickly produces errors far above the noise and is
    followed almost immediately. `noise` is estimated live per axis from the median of second
    differences over the last few seconds, which ignores smooth head motion.
    """

    def __init__(self, k: float, floor: np.ndarray, window: int = 120):
        self.k, self.floor, self.window = k, floor, window
        self.hist: list[np.ndarray] = []
        self.noise = floor.copy()
        self.y = None

    def reset(self):
        """Forget the output (e.g. after a tracking gap) but keep the noise estimate."""
        self.y = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.hist.append(x.copy())
        del self.hist[:-self.window]
        if len(self.hist) >= 10:
            h = np.array(self.hist)
            d2 = h[2:] - 2 * h[1:-1] + h[:-2]
            # median |2nd difference| of white noise with std s is 0.6745 * sqrt(6) * s
            self.noise = np.maximum(np.median(np.abs(d2), axis=0) / (0.6745 * math.sqrt(6)), self.floor)
        if self.y is None:
            self.y = x.copy()
            return x.copy()
        e = x - self.y
        u3 = (np.abs(e) / (self.k * self.noise)) ** 3
        self.y = self.y + e * u3 / (1 + u3)
        return self.y.copy()


def fill_gap(raw: np.ndarray, vel: np.ndarray, age: float, amount: float = 1.0) -> np.ndarray:
    """Advance the latest filtered pose by its velocity to now, at most about one frame."""
    return raw + vel * min(max(age, 0.0), 0.04) * amount


def map_pose(cfg: dict, raw: np.ndarray, lean_gain: float) -> np.ndarray:
    """Head pose (deg / cm) -> in-sim view offset (deg / m) through each axis' curve."""
    out = np.array([apply_curve(raw[i], cfg["axes"][a]) for i, a in enumerate(AXES)])
    out[3:] *= lean_gain
    out[5] = max(out[5], -cfg["lean_back_max_m"])  # leaning back puts the eye inside the seat
    return out


def smoothing_k(pct: float) -> float:
    """Smoothing (0-100) -> AdaptiveFilter threshold, in multiples of the measured camera noise.
    Kept moderate: a higher threshold would swallow small real head movements (the view would
    stop short), so the top of the scale gets its extra smoothness from smoothing_ease()."""
    return 1.5 + 6.5 * min(100.0, max(0.0, pct)) / 100.0


def smoothing_ease(base_s: float, pct: float) -> float:
    """Final output easing time (s) for a smoothing setting: base_s normally, rising to 0.3 s
    at the top of the scale, where most of the extra smoothness comes from (it delays, but
    always arrives)."""
    p = min(100.0, max(0.0, pct)) / 100.0
    return base_s + max(0.0, 0.3 - base_s) * p ** 4


def smoothing_fill(pct: float) -> float:
    """How much of the between-frames motion prediction to use: less as smoothing goes up."""
    p = min(100.0, max(0.0, pct)) / 100.0
    return 1.0 - p ** 4
