import copy
import json

from headtrack.config import DEFAULT_CONFIG, load_config, save_settings, set_look_range
from headtrack.macos import pick_camera


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
    path.write_text(json.dumps({"camera_index": 1, "smoothing_level": 2, "axes": {"yaw": {"exponent": 2.0}}}))
    cfg = load_config(path)
    set_look_range(cfg, 90)
    cfg["smoothing"], cfg["camera_rotation"] = 55, 90
    save_settings(path, cfg)
    saved = json.loads(path.read_text())
    assert saved["camera_index"] == 1 and saved["axes"]["yaw"]["exponent"] == 2.0
    assert "smoothing_level" not in saved
    assert (saved["smoothing"], saved["camera_rotation"], saved["axes"]["yaw"]["output_max"]) == (55, 90, 90.0)


def test_saved_camera_is_found_by_name():
    cfg = dict(DEFAULT_CONFIG, camera_index=0, camera_name="iPhone Camera")
    assert pick_camera(cfg, ["FaceTime HD Camera", "Desk View Camera", "iPhone Camera"]) == 2
    assert pick_camera(cfg, ["FaceTime HD Camera"]) == 0  # iPhone not around: saved number
    assert pick_camera(dict(cfg, camera_index=3, camera_name=""), ["FaceTime HD Camera"]) == 0
