import numpy as np
import pytest

from headtrack.config import DEFAULT_CONFIG
from headtrack.pose import (
    AdaptiveFilter,
    apply_curve,
    euler_yxz,
    fill_gap,
    head_delta,
    map_pose,
    rot_x,
    rot_y,
    rot_z,
    smoothing_ease,
    smoothing_fill,
    smoothing_k,
)


def test_euler_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        y, p, r = rng.uniform(-80, 80), rng.uniform(-60, 60), rng.uniform(-60, 60)
        assert np.allclose(euler_yxz(rot_y(y) @ rot_x(p) @ rot_z(r)), [y, p, r], atol=1e-6)


def test_off_axis_camera():
    # Camera 40 deg to the side and 15 deg below; head turns 20 right, looks 10 up, leans 5 cm in.
    R_ref = rot_y(40) @ rot_x(-15)
    p_ref = np.array([3.0, -2.0, -50.0])
    R = R_ref @ rot_y(-20) @ rot_x(-10)  # face frame: right turn / look up are negative rotations
    p = p_ref + R_ref @ np.array([0, 0, 5.0])
    assert np.allclose(head_delta(R_ref, p_ref, R, p), [20, 10, 0, 0, 0, 5], atol=1e-6)


AX = DEFAULT_CONFIG["axes"]["yaw"]


def test_curve_deadzone_and_clamp():
    assert apply_curve(0.5, AX) == 0.0
    assert apply_curve(90, AX) == AX["output_max"]


def test_curve_symmetric_and_monotonic():
    assert apply_curve(-20, AX) == -apply_curve(20, AX)
    vals = [apply_curve(v, AX) for v in np.linspace(0, 40, 50)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_curve_invert_and_disable():
    assert apply_curve(20, dict(AX, invert=True)) == -apply_curve(20, AX)
    assert apply_curve(20, dict(AX, enabled=False)) == 0.0


def test_filter_measures_noise_and_holds_steady():
    rng = np.random.default_rng(0)
    f = AdaptiveFilter(smoothing_k(40), floor=np.array([0.05]))
    xs = [f(np.array([10.0 + rng.normal(0, 0.5)]))[0] for _ in range(150)]
    assert f.noise[0] == pytest.approx(0.5, abs=0.15)
    assert np.mean(np.abs(np.diff(xs[-60:]))) < 0.05  # frame-to-frame motion is what reads as jitter


def test_filter_follows_real_motion():
    rng = np.random.default_rng(1)
    f = AdaptiveFilter(smoothing_k(40), floor=np.array([0.05]))
    for _ in range(150):
        f(np.array([10.0 + rng.normal(0, 0.5)]))
    ys = [f(np.array([30.0 + rng.normal(0, 0.5)]))[0] for _ in range(3)]
    assert ys[-1] == pytest.approx(30, abs=1.0)


def test_smoothing_scale():
    assert smoothing_k(0) < smoothing_k(40) < smoothing_k(100) == smoothing_k(150)
    # the top of the slider adds slower, smoother easing and less prediction; the default barely changes
    assert smoothing_ease(0.03, 0) == 0.03 and smoothing_ease(0.03, 40) < 0.04
    assert smoothing_ease(0.03, 100) == pytest.approx(0.3)
    assert smoothing_fill(0) == 1.0 and smoothing_fill(40) > 0.95 and smoothing_fill(100) == 0.0


def test_leaning_back_is_capped_forward_is_not():
    assert map_pose(DEFAULT_CONFIG, np.array([0, 0, 0, 0, 0, -10.0]), 1.0)[5] == -DEFAULT_CONFIG["lean_back_max_m"]
    assert map_pose(DEFAULT_CONFIG, np.array([0, 0, 0, 0, 0, 10.0]), 1.0)[5] > 0.2


def test_gap_fill_advances_but_not_past_one_frame():
    raw, vel = np.array([10.0, 0, 0, 0, 0, 0]), np.array([60.0, 0, 0, 0, 0, 0])
    assert fill_gap(raw, vel, 0.02)[0] == pytest.approx(11.2)
    assert fill_gap(raw, vel, 0.5)[0] == pytest.approx(12.4)
