"""AirPods head tracking fed with fake motion samples (no AirPods needed)."""
import math
import time

import numpy as np
import pytest

from headtrack import airpods
from headtrack.airpods import HeadphoneMotion, HeadphoneRotation, headphone_rotation, rate_dps, relative_attitude
from headtrack.state import Shared

# The headphones' frame is x right, y forward, z up: turning right is a negative turn about z.
RIGHT, UP, TILT_RIGHT = (0, 0, -1), (1, 0, 0), (0, 1, 0)


def quat(axis=(0, 0, 1), deg=0.0, start=None):
    """Quaternion (x, y, z, w) of a turn by `deg` about `axis`, after the attitude `start`."""
    h = math.radians(deg) / 2
    q = np.array([*(np.array(axis, float) * math.sin(h)), math.cos(h)])
    if start is None:
        return q
    (x1, y1, z1, w1), (x2, y2, z2, w2) = start, q  # start * q: the turn in the head's own frame
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def test_turns_read_as_yaw_pitch_roll():
    ref = airpods.quat_matrix(*quat())
    for axis, i in ((RIGHT, 0), (UP, 1), (TILT_RIGHT, 2)):
        rot = headphone_rotation(ref, airpods.quat_matrix(*quat(axis, 30)))
        assert rot[i] == pytest.approx(30.0) and np.delete(rot, i) == pytest.approx([0, 0], abs=1e-9)


def test_rate_matches_the_turn():
    assert rate_dps(np.radians(np.array(RIGHT, float) * 90)) == pytest.approx([90, 0, 0])
    assert rate_dps(np.radians(np.array(UP, float) * 90)) == pytest.approx([0, 90, 0])
    assert rate_dps(np.radians(np.array(TILT_RIGHT, float) * 90)) == pytest.approx([0, 0, 90])


def test_relative_attitude_undoes_headphone_rotation():
    rot = np.array([50.0, -20.0, 10.0])
    assert headphone_rotation(np.eye(3), relative_attitude(rot)) == pytest.approx(rot)


def feed(rot, seconds, start_t, q=None, camera=(0, 0, 0), camera_ok=True, rate=(0, 0, 0), sensor=2, hz=25,
         drift_dps=0.0):
    """Samples at `hz` of a still head (attitude q), whose yaw drifts by drift_dps."""
    t = start_t
    for i in range(int(seconds * hz)):
        t = start_t + i / hz
        drifted = quat(RIGHT, drift_dps * (t - start_t), q if q is not None else quat())
        rot.sample(t, drifted, rate, sensor, np.array(camera, float), camera_ok)
    return t


def test_starts_from_the_cameras_rotation_once_the_face_is_tracked():
    rot = HeadphoneRotation()
    t = feed(rot, 1, 0.0, camera_ok=False)
    assert rot.current(t) is None  # nothing to line up with yet: the camera keeps rotation
    t = feed(rot, 0.2, t + 0.04, camera=(20, 0, 0))
    assert rot.current(t)[0] == pytest.approx([20, 0, 0], abs=0.01)
    t = feed(rot, 0.2, t + 0.04, q=quat(RIGHT, 30), camera_ok=False)  # turn 30 deg further right
    assert rot.current(t)[0] == pytest.approx([50, 0, 0], abs=0.01)


def test_gap_or_other_earbud_lines_up_again_without_a_jump():
    rot = HeadphoneRotation()
    t = feed(rot, 1, 0.0, camera=(10, 0, 0))
    assert rot.current(t + 0.4) is not None and rot.current(t + 0.6) is None  # out of the ears
    # The other earbud has its own reference: the same head pose reads 26 deg rolled.
    other = quat(TILT_RIGHT, 26)
    t = feed(rot, 1, t + 0.04, q=other, camera=(10, 0, 0), sensor=1)
    assert rot.current(t)[0] == pytest.approx([10, 0, 0], abs=0.05)


def test_yaw_drift_is_pulled_back_to_the_camera():
    rot = HeadphoneRotation()
    t = feed(rot, 300, 0.0, drift_dps=2 / 60)  # 2 deg/min, as measured, for 5 minutes
    assert abs(rot.current(t)[0][0]) < 0.5
    rot = HeadphoneRotation()
    feed(rot, 1, 0.0)
    t = feed(rot, 300, 1.0, drift_dps=2 / 60, camera_ok=False)  # no face: nothing to correct with
    assert rot.current(t)[0][0] == pytest.approx(10, abs=0.5)


def test_no_drift_correction_while_turning_or_looking_away():
    rot = HeadphoneRotation()
    t = feed(rot, 1, 0.0)
    # the camera says still and centered, but the head is turning: the AirPods are right
    t = feed(rot, 2, t + 0.04, camera=(0, 0, 0), rate=(0, 0, -0.5))
    assert rot.yaw_fix == 0.0
    t = feed(rot, 2, t + 0.04, q=quat(RIGHT, 40), camera=(37, 0, 0))  # far right: the camera is inexact
    assert rot.current(t)[0][0] == pytest.approx(40, abs=1e-6)


def test_recenter_makes_the_current_pose_neutral():
    rot = HeadphoneRotation()
    t = feed(rot, 1, 0.0, q=quat(RIGHT, 25), camera=(25, 0, 0))
    rot.anchor(np.zeros(3))
    t = feed(rot, 0.2, t + 0.04, q=quat(RIGHT, 35), camera_ok=False)
    assert rot.current(t)[0] == pytest.approx([10, 0, 0], abs=0.01)


@pytest.fixture
def motion(monkeypatch):
    """The AirPods reader with fake CoreMotion samples: (send(quaternion), shared, reader)."""
    monkeypatch.setattr(HeadphoneMotion, "_start_updates", lambda self: None)
    sh = Shared()
    sh.face_ok = sh.has_ref = True
    reader = HeadphoneMotion(sh)
    reader.start()
    yield (lambda q: reader.events.put(("sample", time.monotonic(), q, (0, 0, 0), 2))), sh, reader
    with sh.lock:
        sh.quit = True
    reader.join(1)


def test_reader_publishes_rotation_and_falls_back_when_they_are_out(motion):
    send, sh, reader = motion
    with sh.lock:
        sh.raw[:3] = [15, 0, 0]
    for _ in range(10):
        send(quat())
        time.sleep(0.04)
    time.sleep(0.05)
    with sh.lock:
        assert sh.airpods_rot == pytest.approx([15, 0, 0], abs=0.01) and sh.airpods_status == "on"
    time.sleep(0.7)
    with sh.lock:  # the last rotation stays (to hold the view), but it's old
        assert sh.airpods_status == "waiting" and time.monotonic() - sh.airpods_time > airpods.STALE_S


def test_reader_shows_what_went_wrong(motion):
    send, sh, reader = motion
    reader.events.put(("status", "not allowed", ": allow it in System Settings"))
    time.sleep(0.3)
    with sh.lock:
        assert sh.airpods_status == "not allowed"


def test_stopping_goes_back_to_the_camera(motion):
    send, sh, reader = motion
    for _ in range(5):
        send(quat())
        time.sleep(0.04)
    reader.stop()
    with sh.lock:
        assert sh.airpods_rot is None and sh.airpods_status == ""


def test_recenter_reaches_the_airpods_directly(motion):
    send, sh, reader = motion
    for i in range(10):
        send(quat(RIGHT, 20))
        time.sleep(0.04)
        if i == 5:
            with sh.lock:
                sh.airpods_recenter = True  # e.g. while the camera is off
    time.sleep(0.05)
    with sh.lock:
        assert sh.airpods_rot == pytest.approx([0, 0, 0], abs=0.01) and not sh.airpods_recenter


def test_from_terminal_asks_for_the_app(monkeypatch):
    monkeypatch.delenv("HEADTRACK_APP", raising=False)
    assert HeadphoneMotion(Shared())._start_updates()[0] == "needs the app"  # CoreMotion would kill the process
