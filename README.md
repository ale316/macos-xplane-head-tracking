# X-Plane Head Tracker for Mac

[![Tests](https://github.com/ale316/macos-xplane-head-tracking/actions/workflows/tests.yml/badge.svg)](https://github.com/ale316/macos-xplane-head-tracking/actions/workflows/tests.yml)

Free head tracking for **X-Plane 12 on macOS** using the camera you already have: the MacBook's built-in camera, a USB webcam, or your iPhone via Continuity Camera. Turn your head and the cockpit view follows. No IR clip, no paid app, no X-Plane plugin.

> **macOS only, Apple Silicon only.** There is no Windows, Linux or Intel Mac version. On those, have a look at [opentrack](https://github.com/opentrack/opentrack).

- **Full 6DoF:** look around (yaw, pitch, roll) and lean (side to side, up/down, forward/back).
- **Camera can be off to the side or below you.** Movement is measured relative to how you sit when you recenter.
- **Steady but responsive:** the filter measures your camera's own jitter and ignores movement at that level. A hand at your face doesn't throw the view around.
- **Phone in any orientation:** it detects when the camera is turned sideways or upside down.
- **Preview window:** shows the camera and a simulated cockpit view, with sliders for look range and smoothness. It can be switched off while you fly.

## Requirements

| | |
|---|---|
| **Mac** | Apple Silicon (M1 or newer). Intel Macs aren't supported: the face-tracking library has no Intel build. |
| **macOS** | 14 Sonoma or newer (developed and tested on 14). macOS 13 may work but isn't tested. |
| **X-Plane** | X-Plane 12, running **on the same Mac** (developed with 12.4; X-Plane 11 isn't tested). It uses X-Plane's built-in network interface, which is on by default. |
| **Camera** | The built-in camera (30 fps), a USB webcam, or an iPhone via [Continuity Camera](https://support.apple.com/en-us/102546) (iPhone XR or newer with iOS 16 or newer; up to 60 fps). |
| **Disk and download** | About 550 MB of disk space; the installer downloads about 170 MB. Internet is only needed to install or update. |
| **Permissions** | **Camera** access for your terminal app (required). **Input Monitoring** for your terminal app (optional, for the global hotkeys). |
| **CPU** | About half a CPU core while tracking, plus the preview while it's switched on. |

You don't need to install Python or anything else: the installer brings its own.

## Install

Open **Terminal** (in Applications › Utilities), paste this line and press Return:

```bash
curl -fsSL https://raw.githubusercontent.com/ale316/macos-xplane-head-tracking/main/install.sh | bash
```

It installs everything into a folder called **X-Plane Head Tracker** in your home folder, and opens that folder in Finder when it's done. That folder contains:

- the code;
- a private copy of Python and the packages, set up with [uv](https://docs.astral.sh/uv/);
- the face and hand models.

The downloads it relies on are pinned and checksum-verified. Nothing outside that folder is touched.

- **Update:** run the same line again. Your settings are kept.
- **Uninstall:** delete the folder.

If you'd rather not paste a command, download this repository instead (green **Code** button › **Download ZIP**), unzip it, and run `bash install.sh` inside it.

## Run

1. Start X-Plane, load a flight and switch to the 3D cockpit view (`Shift+9`).
2. Double-click **Start Head Tracker** in the X-Plane Head Tracker folder.
3. **Allow camera access** for Terminal when macOS asks, then start it again.
4. In the preview window, pick your camera from the **Camera** list (e.g. your iPhone).
5. Sit normally, look at the middle of the X-Plane screen and click **Recenter**.

### Controls

| | Preview window | While flying (global hotkey) |
|---|---|---|
| Recenter | **Recenter** button or `C` | `Ctrl+Shift+C` |
| Pause (hands the view back to X-Plane) | **Pause** button or `P` | `Ctrl+Shift+P` |
| Leaning on/off | **Lean** button or `L` | `Ctrl+Shift+L` |
| Preview on/off | **Preview** button or `V` | – |
| Quit | **Quit** button or `Q` | `Ctrl+C` in Terminal |

- **Look range:** how far the view turns when your head is fully turned. Both handles move together; ±90° means a full turn looks straight out the side window.
- **Smoothness:** from snappy to steady. Towards the top it also makes the view glide, rather than snap, to where you look.
- **Camera:** pick a camera from the list. It's remembered: if it isn't there yet when you start (e.g. the iPhone connects a bit later), the tracker switches to it as soon as it shows up. If the camera in use disappears, it falls back to another one. Cameras that can't be used right now (e.g. the built-in one with the lid closed) aren't listed.
- **Rotation:** detects when the camera is turned sideways or upside down; click it to force 0/90/180/270°.

Everything you set in the window is saved to `config.json` in the same folder.

## Tips

- **Light your face evenly from the front.** Dim light makes the camera noisier and slower, which you'll feel as jitter and lag.
- **An iPhone runs at 60 fps** instead of the built-in camera's 30, which noticeably cuts lag.
- **Switch the preview off while you fly** to save CPU. Tracking keeps running at full speed either way.
- **Leaning back** is limited to a few centimetres, so the view never ends up inside the seat. Leaning forward, e.g. to read an instrument, isn't limited.
- **Hands at your face** (touching your face, drinking, rubbing your eyes): while a hand overlaps your face, tracking keeps following you but is steadied, so the sudden jumps a hand causes are smoothed out. The preview marks hands it sees with a box.
- **A new flight or aircraft** starts looking straight ahead from that aircraft's seat.
- The tracker only moves the view in X-Plane's 3D cockpit; outside views are left alone.

## Privacy

- **Camera image:** processed on your Mac only. It's never saved or sent anywhere.
- **Network:** the tracker only talks to X-Plane on your own Mac (UDP port 49000).
- **Files it reads:** X-Plane's `Log.txt` and the loaded aircraft's `.acf` file, to find the seat position.
- **Internet:** only the installer uses it, to download Python, the packages and the models.
- **`--diag`:** the recording contains only timings and head angles (numbers, no images).

## Troubleshooting

| Problem | Fix |
|---|---|
| "X-Plane: waiting for 127.0.0.1:49000" | Make sure X-Plane is running, and check under X-Plane's Settings › Network that its UDP receive port is 49000. |
| "No camera access" / "Could not open the camera" | Allow your terminal app under System Settings › Privacy & Security › Camera, or pick another camera. |
| iPhone not in the Camera list | macOS isn't offering it as a camera right now. For Continuity Camera, the iPhone must be locked, still (on a mount or stand) and near the Mac, with Wi-Fi and Bluetooth on and the same Apple ID on both. Also check that iPhone Settings › General › AirPlay & Continuity › Continuity Camera is on. A USB cable helps. |
| "camera at 24 of 60 fps" in the preview | A macOS video effect (Portrait, Studio Light, Center Stage in Control Center › Video Effects) or very dim light is slowing the camera down. |
| Global hotkeys don't work | Allow your terminal app under System Settings › Privacy & Security › Input Monitoring, then restart the tracker. |
| An axis moves the wrong way | Set `"invert": true` for that axis in `config.json` (see below). |
| "config.json isn't valid JSON" | Fix the typo it points at, or delete `config.json` to go back to the defaults. |
| Laggy or jittery | Improve the lighting, then adjust the Smoothness slider. |

## Reporting problems

Please [open an issue](https://github.com/ale316/macos-xplane-head-tracking/issues) with:

- your macOS version, Mac model and X-Plane version;
- the camera you use;
- what happened.

A short recording helps a lot. Start the tracker from Terminal in the install folder with `./run.sh --diag diag.csv`, reproduce the problem, quit, then attach `diag.csv` and paste what the Terminal window printed.

## Configuration

`config.json` is created when you first change a setting; anything not in it uses the default. Useful keys:

| Key | Default | Meaning |
|---|---|---|
| `camera_id` / `camera_name` | – | Camera, set with the Camera list |
| `camera_rotation` | `"auto"` | `"auto"` or a fixed rotation: `0`, `90`, `180`, `270` |
| `smoothing` | `40` | Smoothness slider, 0–100 |
| `lean_enabled` | `true` | Start with leaning on |
| `lean_back_max_m` | `0.03` | How far the view may move back from the seat (meters) |
| `hand_at_face` | `"follow"` | With a hand at your face: `"follow"` (steadied), `"hold"` (freeze the view) or `"off"` |
| `axes.<axis>.input_max` | yaw 35, pitch 25 | Head movement (degrees or cm) that gives full output |
| `axes.<axis>.output_max` | yaw 150, pitch 60 | Full output in X-Plane (degrees or meters); yaw is the Look range slider |
| `axes.<axis>.exponent` | 1.0–1.4 | Above 1 makes small movements gentler |
| `axes.<axis>.invert` / `enabled` | `false` / `true` | Flip or switch off an axis (`yaw`, `pitch`, `roll`, `x`, `y`, `z`) |
| `hotkeys` | see Controls | e.g. `{"recenter": "ctrl+shift+c"}` |

## How it works

1. **Head pose:** Google's [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) estimates your head's 3D pose from each camera frame. Its Hand Landmarker spots hands in front of your face.
2. **Filtering and curves:** the pose is filtered and run through response curves.
3. **Sending:** the view is sent 60 times a second to X-Plane's built-in UDP interface, as `sim/graphics/view/pilots_head_*` dataref writes. The view is measured from the seat position in the loaded aircraft's `.acf` file. X-Plane stores its eye point in the same variable as the live head position, so the tracker can't read it back.

The tracking runs in a windowless background process, because macOS slows down background apps that have a window. The preview is a separate process.

## Development

```bash
git clone https://github.com/ale316/macos-xplane-head-tracking.git && cd macos-xplane-head-tracking
bash install.sh                                     # .venv, .tools (uv, Python) and the models, all local
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
./run.sh --help                                     # --camera, --no-preview, --diag, --list-cameras ...
```

The code is in `headtrack/`:

| Module | Contents |
|---|---|
| `camera.py` | Camera capture (AVFoundation) |
| `tracker.py` | Face and hand tracking |
| `pose.py` | Pose maths and filter |
| `xplane.py` | UDP link and seat position |
| `preview.py` | Preview window |
| `macos.py` | macOS integration |
| `config.py` / `state.py` | Settings, and state shared between threads |
| `app.py` | Command line and wiring |

## License and credits

MIT, see [LICENSE](LICENSE).

It builds on:

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) and its face and hand models, which are downloaded from Google at install (Apache 2.0);
- [OpenCV](https://opencv.org) (Apache 2.0), [NumPy](https://numpy.org) (BSD) and [PyObjC](https://github.com/ronaldoussoren/pyobjc) (MIT);
- [uv](https://github.com/astral-sh/uv) (MIT / Apache 2.0) and Python, which are downloaded at install.

Not affiliated with Laminar Research (X-Plane), Apple or Google.
