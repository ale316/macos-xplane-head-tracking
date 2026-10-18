"""Settings: defaults, loading config.json, and saving what the preview's sliders change."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
CONFIG_PATH = ROOT / "config.json"


AXES = ("yaw", "pitch", "roll", "x", "y", "z")


DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_width": 640,
    "camera_height": 480,
    "xplane_host": "127.0.0.1",
    "xplane_port": 49000,
    "send_rate_hz": 60,
    # Highest frame rate to lock the camera at. The built-in camera tops out at 30 (and would
    # otherwise drop to 15 in dim light); an iPhone via Continuity Camera can do 60.
    "camera_fps": 60,
    # Clockwise rotation (0/90/180/270) that makes the camera image upright, e.g. for a phone
    # mounted sideways; "auto" finds it by itself whenever your face is lost for a moment.
    "camera_rotation": "auto",
    # 0 = snappiest / most jitter ... 100 = steadiest / most lag. Also a slider in the preview.
    "smoothing": 40,
    # Final easing (seconds) that turns 30 fps camera steps into smooth 60 Hz motion.
    "output_smoothing_s": 0.03,
    # Between camera frames, continue the filtered motion up to the current moment (never
    # ahead of it). Cuts ~15 ms of lag; the filter holds still at rest, so no added jitter.
    "fill_between_frames": True,
    # Leaning (x/y/z translation) on at startup; toggle at runtime with the lean hotkey / L.
    "lean_enabled": True,
    # When the face is lost, the view eases back to center after this long.
    "face_lost_timeout_s": 0.6,
    "face_lost_smoothing_s": 0.5,
    # Only drive the view while X-Plane is in the 3D cockpit view.
    "only_in_cockpit_view": True,
    "recenter_frames": 8,
    # Head-tracking translation is measured at this point in the face model's frame
    # (cm; +x subject's left, +y up, +z out of the face). Pointing it back toward the
    # neck pivot stops head *rotation* from leaking into x/z *translation*.
    "translation_pivot_cm": [0.0, -5.0, -5.0],
    "sounds": True,
    "hotkeys": {"recenter": "ctrl+shift+c", "pause": "ctrl+shift+p", "lean": "ctrl+shift+l"},
    # input_* in degrees (rotation) or cm (translation) of real head movement;
    # output_max in degrees or meters of in-sim view movement.
    "axes": {
        "yaw": {"enabled": True, "invert": False, "deadzone": 1.0, "input_max": 35.0, "output_max": 150.0, "exponent": 1.4},
        "pitch": {"enabled": True, "invert": False, "deadzone": 1.0, "input_max": 25.0, "output_max": 60.0, "exponent": 1.3},
        "roll": {"enabled": True, "invert": False, "deadzone": 2.0, "input_max": 30.0, "output_max": 20.0, "exponent": 1.0},
        "x": {"enabled": True, "invert": False, "deadzone": 0.5, "input_max": 12.0, "output_max": 0.20, "exponent": 1.0},
        "y": {"enabled": True, "invert": False, "deadzone": 0.5, "input_max": 8.0, "output_max": 0.10, "exponent": 1.0},
        "z": {"enabled": True, "invert": False, "deadzone": 0.5, "input_max": 12.0, "output_max": 0.25, "exponent": 1.0},
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path.exists():
        user = json.loads(path.read_text())
        if "smoothing_level" in user and "smoothing" not in user:  # older config: levels 1-5
            user["smoothing"] = {1: 15, 2: 23, 3: 38, 4: 54, 5: 69}.get(user["smoothing_level"], 40)
        cfg = deep_merge(cfg, user)
    return cfg


def set_look_range(cfg: dict, yaw_deg: float):
    """How far the view turns at full head turn: yaw +/-yaw_deg, pitch proportionally."""
    yaw = round(min(180.0, max(30.0, yaw_deg)), 1)
    cfg["axes"]["yaw"]["output_max"] = yaw
    cfg["axes"]["pitch"]["output_max"] = round(min(90.0, max(15.0, yaw * 0.4)), 1)


def save_settings(path: Path, cfg: dict):
    """Write the slider settings into the config file, leaving everything else as is."""
    data = json.loads(path.read_text()) if path.exists() else {}
    data["smoothing"] = cfg["smoothing"]
    data["camera_rotation"] = cfg["camera_rotation"]
    data.pop("smoothing_level", None)
    for axis in ("yaw", "pitch"):
        data.setdefault("axes", {}).setdefault(axis, {})["output_max"] = cfg["axes"][axis]["output_max"]
    path.write_text(json.dumps(data, indent=2) + "\n")
