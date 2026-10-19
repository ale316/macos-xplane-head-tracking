"""Commands from the preview window, as the engine carries them out."""
import copy
import json
import queue
from types import SimpleNamespace

import pytest

from headtrack import app
from headtrack.app import Controls
from headtrack.camera import CameraInfo
from headtrack.config import DEFAULT_CONFIG
from headtrack.state import Shared

FACETIME, IPHONE = CameraInfo("A1", "FaceTime HD Camera"), CameraInfo("B2", "iPhone Camera")


@pytest.fixture
def controls(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["sounds"] = False
    sh = Shared()
    sh.cam_list, sh.cam_id = [FACETIME, IPHONE], FACETIME.id
    tracker = SimpleNamespace(cameras=[], rotations=[], recentered=0)
    tracker.request_camera = tracker.cameras.append
    tracker.set_rotation_mode = tracker.rotations.append
    tracker.request_recenter = lambda: setattr(tracker, "recentered", tracker.recentered + 1)
    return Controls(cfg, sh, tracker, tmp_path / "config.json")


def saved(controls):
    return json.loads(controls.config_path.read_text()) if controls.config_path.exists() else None


def test_sliders_apply_at_once_and_save_on_release(controls):
    controls.handle("range", 90, False)
    controls.handle("smoothing", 70, False)
    assert controls.cfg["axes"]["yaw"]["output_max"] == 90 and controls.sh.smoothing == 70
    assert saved(controls) is None  # not on every step of a drag
    controls.handle("smoothing", 75, True)
    assert saved(controls)["smoothing"] == 75 and saved(controls)["axes"]["yaw"]["output_max"] == 90


def test_picking_a_camera_switches_and_remembers_it(controls):
    controls.handle("camera", 0)  # already in use
    assert controls.tracker.cameras == []
    controls.handle("camera", 1)
    assert controls.tracker.cameras == [IPHONE] and saved(controls)["camera_id"] == IPHONE.id
    controls.handle("camera", 5)  # gone since the list was drawn
    assert controls.tracker.cameras == [IPHONE]


def test_rotation_is_passed_on_and_saved(controls):
    controls.handle("rotation", 90)
    assert controls.tracker.rotations == [90] and saved(controls)["camera_rotation"] == 90


def test_buttons_toggle_and_unknown_commands_are_ignored(controls):
    controls.handle("pause")
    controls.handle("lean")
    controls.handle("recenter")
    controls.handle("preview", False)
    controls.handle("something new", 1)
    sh = controls.sh
    assert sh.paused and not sh.lean and controls.tracker.recentered == 1 and not sh.preview_on


def test_serves_commands_until_quit(controls):
    q = queue.Queue()
    for cmd in [("pause",), ("quit",), ("lean",)]:
        q.put(cmd)
    controls.serve(q)
    assert controls.sh.quit and controls.sh.paused and controls.sh.lean  # nothing after quit


def test_airpods_switch_on_and_off_and_are_remembered(controls, monkeypatch):
    class FakeMotion:
        def __init__(self, sh):
            self.running = False
        def start(self):
            self.running = True
        def stop(self):
            self.running = False
    monkeypatch.setattr(app, "HeadphoneMotion", FakeMotion)
    controls.handle("airpods", True)
    reader = controls.airpods
    assert reader.running and saved(controls)["airpods"] is True
    controls.handle("airpods", True)  # already on
    assert controls.airpods is reader
    controls.handle("airpods", False)
    assert not reader.running and controls.airpods is None and saved(controls)["airpods"] is False
