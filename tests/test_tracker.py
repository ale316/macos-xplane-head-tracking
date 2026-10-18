"""Tracker logic fed with fake MediaPipe results (no camera or face model needed)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from headtrack import tracker as tracker_mod
from headtrack.config import load_config
from headtrack.pose import rot_y, rot_z
from headtrack.state import Shared
from headtrack.tracker import HAND_MAX_SPEED, HandWatcher, Tracker, hand_over_face


def face(yaw: float = 0.0, roll: float = 0.0):
    """A fake MediaPipe face result: head turned `yaw` right, or turned `roll` in the image
    (clockwise negative), 50 cm from the camera."""
    M = np.eye(4)
    M[:3, :3] = rot_y(-yaw) @ rot_z(roll)  # in the face frame, turning right is a negative rotation
    M[:3, 3] = [0.0, 0.0, -50.0]
    return SimpleNamespace(facial_transformation_matrixes=[M.ravel()],
                           face_landmarks=[[SimpleNamespace(x=0.5, y=0.5)]])


@pytest.fixture
def tracker(monkeypatch):
    monkeypatch.setattr(tracker_mod, "play_sound", lambda *a: None)
    return Tracker(load_config(Path("/nonexistent")), Shared())


def feed_roll(tr, roll, seconds):
    for i in range(int(seconds * 30)):
        tr._check_orientation(face(roll=roll), None, i / 30, probe=None)


def test_face_on_its_side_for_a_second_turns_the_image(tracker):
    feed_roll(tracker, -88, 1.5)  # camera turned clockwise -> rotate images counter-clockwise (270)
    assert tracker.rotation == 270


def test_face_on_its_other_side_turns_it_the_other_way(tracker):
    feed_roll(tracker, 92, 1.5)
    assert tracker.rotation == 90


def test_head_tilt_or_brief_turn_keeps_orientation(tracker):
    feed_roll(tracker, 40, 3)  # a strongly tilted head is still upright enough
    assert tracker.rotation == 0
    feed_roll(tracker, -85, 0.5)  # too brief
    assert tracker.rotation == 0


def run_frames(tr, t0, seconds, yaw, hand_at_face=False):
    """Feed `seconds` of 30 fps frames of a head turned `yaw` right; returns the end time."""
    t = t0
    for i in range(int(seconds * 30)):
        t = t0 + i / 30
        if hand_at_face:
            tr.hands.check([HAND_ON_MOUTH], FACE, t)
        tr._process(None, face(yaw), t)
    return t


def test_recenters_on_the_first_frames_then_follows_the_head(tracker):
    t = run_frames(tracker, 100.0, 0.5, yaw=10)  # whatever the pose at the start is neutral
    assert tracker.shared.has_ref and not tracker.shared.recentering
    run_frames(tracker, t + 1 / 30, 0.3, yaw=30)
    assert tracker.shared.raw[0] == pytest.approx(20, abs=0.5)


def test_hand_at_face_slows_the_view_and_releases_it_after(tracker):
    tracker.hands = HandWatcher("unused", 15)  # not started: hands are reported by the test
    t = run_frames(tracker, 100.0, 0.5, yaw=0)
    t = run_frames(tracker, t + 1 / 30, 0.2, yaw=30, hand_at_face=True)  # a hand makes the pose jump
    assert tracker.shared.hand_at_face
    assert 0 < tracker.shared.raw[0] <= HAND_MAX_SPEED[0] * 0.2
    run_frames(tracker, t + 1 / 30, 1.0, yaw=30)
    assert not tracker.shared.hand_at_face and tracker.shared.raw[0] == pytest.approx(30, abs=0.5)


FACE = (0.4, 0.3, 0.6, 0.6)
HAND_ON_MOUTH = (0.45, 0.5, 0.55, 0.7)


def test_hand_over_or_touching_the_face_counts():
    assert hand_over_face([HAND_ON_MOUTH], FACE)
    assert hand_over_face([(0.61, 0.35, 0.7, 0.5)], FACE)  # touching the cheek (within the margin)
    assert not hand_over_face([(0.75, 0.1, 0.9, 0.4)], FACE)  # raised beside the head
    assert not hand_over_face([HAND_ON_MOUTH], None)  # no face


def test_hand_watcher_holds_a_little_after_the_hand_leaves_and_marks_the_onset():
    w = HandWatcher("unused", 15)
    w.check([(0.75, 0.1, 0.9, 0.4)], FACE, 1.0)
    assert not w.at_face(1.0)
    w.check([HAND_ON_MOUTH], FACE, 2.0)
    w.check([HAND_ON_MOUTH], FACE, 2.1)
    assert w.at_face(2.15) and w.onset == 2.0
    w.check([], FACE, 2.2)
    assert w.at_face(2.3) and not w.at_face(2.5)
    w.check([HAND_ON_MOUTH], FACE, 3.0)
    assert w.onset == 3.0  # a new streak


def test_holding_rewinds_to_before_the_hand(tracker):
    for i in range(30):
        tracker._history.append((i / 30, np.full(6, float(i))))
    assert tracker._pose_before(0.5)[0] == 15
    assert tracker._pose_before(-1)[0] == 0
