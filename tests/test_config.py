import copy
import json

from headtrack.config import DEFAULT_CONFIG, load_config, save_settings, set_look_range
from headtrack.camera import CameraInfo, camera_to_use, pick_camera


def test_look_range_limits_and_pitch_follows():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    set_look_range(cfg, 500)
    assert (cfg["axes"]["yaw"]["output_max"], cfg["axes"]["pitch"]["output_max"]) == (180.0, 72.0)
    set_look_range(cfg, 90)
    assert (cfg["axes"]["yaw"]["output_max"], cfg["axes"]["pitch"]["output_max"]) == (90.0, 36.0)


def test_missing_config_gives_defaults(tmp_path):
    assert load_config(tmp_path / "nope.json") == DEFAULT_CONFIG


def test_old_smoothing_level_converts(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"smoothing_level": 2}))
    assert load_config(path)["smoothing"] == 23


def test_save_keeps_other_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"send_rate_hz": 90, "smoothing_level": 2, "axes": {"yaw": {"exponent": 2.0}}}))
    cfg = load_config(path)
    set_look_range(cfg, 90)
    cfg["smoothing"], cfg["camera_rotation"] = 55, 90
    save_settings(path, cfg)
    saved = json.loads(path.read_text())
    assert saved["send_rate_hz"] == 90 and saved["axes"]["yaw"]["exponent"] == 2.0
    assert "smoothing_level" not in saved
    assert (saved["smoothing"], saved["camera_rotation"], saved["axes"]["yaw"]["output_max"]) == (55, 90, 90.0)


def test_saved_camera_is_found_by_id_then_name():
    facetime, iphone = CameraInfo("A1", "FaceTime HD Camera"), CameraInfo("B2", "iPhone Camera")
    cfg = dict(DEFAULT_CONFIG, camera_id="B2", camera_name="iPhone Camera")
    assert pick_camera(cfg, [facetime, iphone]) == iphone
    assert pick_camera(cfg, [iphone, facetime]) == iphone  # macOS listed them in another order
    assert pick_camera(dict(cfg, camera_id="other"), [facetime, iphone]) == iphone  # same name, new ID
    assert pick_camera(cfg, [facetime]) == facetime  # iPhone not around: the first camera
    assert pick_camera(cfg, []) is None


def test_switches_to_the_chosen_camera_when_it_appears_and_away_from_a_missing_one():
    facetime, iphone = CameraInfo("A1", "FaceTime HD Camera"), CameraInfo("B2", "iPhone Camera")
    cfg = dict(DEFAULT_CONFIG, camera_id="B2", camera_name="iPhone Camera")
    assert camera_to_use(cfg, [facetime], "A1") is None  # iPhone not around yet: stay
    assert camera_to_use(cfg, [facetime, iphone], "A1") == iphone  # it connected: switch to it
    assert camera_to_use(cfg, [facetime, iphone], "B2") is None  # already on it
    assert camera_to_use(cfg, [facetime], "B2") == facetime  # it went away: use what's there
    assert camera_to_use(cfg, [], "B2") is None
