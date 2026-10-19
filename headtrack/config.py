"""Settings: defaults, loading config.json, and saving what the preview's sliders change."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
CONFIG_PATH = ROOT / "config.json"


AXES = ("yaw", "pitch", "roll", "x", "y", "z")


def axis(deadzone: float, input_max: float, output_max: float, exponent: float = 1.0) -> dict:
    return {"enabled": True, "invert": False, "deadzone": deadzone, "input_max": input_max,
            "output_max": output_max, "exponent": exponent}


DEFAULT_CONFIG = {
    "camera_id": "",  # chosen in the preview; found again by ID, or by name
    "camera_name": "",
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
    "lean_back_max_m": 0.03,  # leaning back further would put the eye inside the seat
    # When the face is lost, the view eases back to center after this long.
    "face_lost_timeout_s": 1.5,
    "face_lost_smoothing_s": 0.8,
    # A hand in front of the face makes MediaPipe report jumpy, partly wrong poses meanwhile:
    # "follow" keeps tracking but steadied, "hold" freezes the view, "off" ignores hands.
    "hand_at_face": "follow",
    "hand_check_hz": 15,
    # Turn the view with AirPods (Pro, Max, 3rd/4th gen) or Beats head tracking instead of the
    # camera: less lag, and hands or bad light don't disturb it. The camera still does leaning,
    # and rotation while they're out; with lean off, it switches off. Needs the tracker opened
    # as X-Plane Head Tracker.app. Also in the preview's camera menu.
    "airpods": False,
    # Only drive the view while X-Plane is in the 3D cockpit view.
    "only_in_cockpit_view": True,
    "recenter_frames": 8,
    # Head-tracking translation is measured at this point in the face model's frame
    # (cm; +x subject's left, +y up, +z out of the face). Pointing it back toward the
    # neck pivot stops head *rotation* from leaking into x/z *translation*.
    "translation_pivot_cm": [0.0, -5.0, -5.0],
    "sounds": True,
    "hotkeys": {"recenter": "ctrl+shift+c", "pause": "ctrl+shift+p", "lean": "ctrl+shift+l"},
    # Response per axis (see pose.apply_curve): deadzone and input_max in degrees (rotation) or
    # cm (translation) of real head movement; output_max in degrees or meters of view movement.
    "axes": {
        "yaw": axis(deadzone=1.0, input_max=35.0, output_max=150.0, exponent=1.4),
        "pitch": axis(deadzone=1.0, input_max=25.0, output_max=60.0, exponent=1.3),
        "roll": axis(deadzone=2.0, input_max=30.0, output_max=20.0),
        "x": axis(deadzone=0.5, input_max=12.0, output_max=0.20),
        "y": axis(deadzone=0.5, input_max=8.0, output_max=0.10),
        "z": axis(deadzone=0.5, input_max=12.0, output_max=0.25),
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


class ConfigError(Exception):
    pass


def load_config(path: Path) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path.exists():
        try:
            user = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path} isn't valid JSON ({e.msg}, line {e.lineno}). "
                              "Fix it, or delete it to go back to the defaults.") from None
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
    data["camera_id"], data["camera_name"] = cfg["camera_id"], cfg["camera_name"]
    data.pop("camera_index", None)  # older versions picked cameras by number
    data["camera_rotation"] = cfg["camera_rotation"]
    data["airpods"] = cfg["airpods"]
    data.pop("smoothing_level", None)
    for axis in ("yaw", "pitch"):
        data.setdefault("axes", {}).setdefault(axis, {})["output_max"] = cfg["axes"][axis]["output_max"]
    path.write_text(json.dumps(data, indent=2) + "\n")
