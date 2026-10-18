"""Camera-orientation decisions (no camera or face model needed)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from headtrack import tracker as tracker_mod
from headtrack.tracker import HandWatcher, hand_over_face
from headtrack.config import load_config
from headtrack.pose import rot_z
from headtrack.state import Shared


def face_with_roll(deg: float):
    """A fake MediaPipe result whose face is turned `deg` in the image (clockwise negative)."""
    M = np.eye(4)
    M[:3, :3] = rot_z(deg)
    return SimpleNamespace(facial_transformation_matrixes=[M.ravel()])


def make_tracker():
    tracker_mod.play_sound = lambda *a: None
    return tracker_mod.Tracker(load_config(Path("/nonexistent")), Shared())


def feed(tr, roll, seconds):
    for i in range(int(seconds * 30)):
        tr._check_orientation(face_with_roll(roll), None, i / 30, None, None, None)


def test_face_on_its_side_for_a_second_turns_the_image():
    tr = make_tracker()
    feed(tr, -88, 1.5)  # camera turned clockwise -> rotate images counter-clockwise (270)
    assert tr.rotation == 270
    tr = make_tracker()
    feed(tr, 92, 1.5)
    assert tr.rotation == 90


def test_head_tilt_or_brief_turn_keeps_orientation():
    tr = make_tracker()
    feed(tr, 40, 3)  # a strongly tilted head is still upright enough
    assert tr.rotation == 0
    feed(tr, -85, 0.5)  # too brief
    assert tr.rotation == 0


FACE = (0.4, 0.3, 0.6, 0.6)


def test_hand_over_or_touching_the_face_counts():
    assert hand_over_face([(0.45, 0.5, 0.55, 0.7)], FACE)  # over the mouth
    assert hand_over_face([(0.61, 0.35, 0.7, 0.5)], FACE)  # touching the cheek (within the margin)
    assert not hand_over_face([(0.75, 0.1, 0.9, 0.4)], FACE)  # raised beside the head
    assert not hand_over_face([(0.45, 0.5, 0.55, 0.7)], None)  # no face


def test_hand_watcher_holds_a_little_after_the_hand_leaves_and_marks_the_onset():
    w = HandWatcher("unused", 15)
    w.check([(0.75, 0.1, 0.9, 0.4)], FACE, 1.0)
    assert not w.at_face(1.0)
    w.check([(0.45, 0.5, 0.55, 0.7)], FACE, 2.0)
    w.check([(0.45, 0.5, 0.55, 0.7)], FACE, 2.1)
    assert w.at_face(2.15) and w.onset == 2.0
    w.check([], FACE, 2.2)
    assert w.at_face(2.3) and not w.at_face(2.5)
    w.check([(0.45, 0.5, 0.55, 0.7)], FACE, 3.0)
    assert w.onset == 3.0  # a new streak


def test_holding_rewinds_to_before_the_hand():
    tr = make_tracker()
    for i in range(30):
        tr._history.append((i / 30, np.full(6, float(i))))
    assert tr._pose_before(0.5)[0] == 15
    assert tr._pose_before(-1)[0] == 0
