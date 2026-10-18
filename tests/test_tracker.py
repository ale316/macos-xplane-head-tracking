"""Camera-orientation decisions (no camera or face model needed)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from headtrack import tracker as tracker_mod
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
