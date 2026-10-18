# X-Plane Head Tracker for Mac

Free head tracking for **X-Plane 12 on macOS** using the camera you already have: the MacBook's built-in camera, or your iPhone via Continuity Camera. Turn your head and the cockpit view follows. No IR clip, no paid app, no X-Plane plugin.

- **Full 6DoF:** look around (yaw, pitch, roll) and lean (side to side, up/down, forward/back).
- **Camera can be off to the side or below you.** Movement is measured relative to how you sit when you recenter.
- **Steady but responsive:** the filter measures your camera's own jitter and ignores movement at that level.
- **Phone in any orientation:** it detects when the camera is turned sideways or upside down.
- **Preview window:** shows the camera and a simulated cockpit view, with sliders for look range and smoothness. It can be switched off while you fly.

## Requirements

- A Mac with **Apple Silicon** (M1 or newer)
- **X-Plane 12** (it uses X-Plane's built-in network interface, which is on by default)

Nothing else: the installer brings its own Python.

## Install

Open **Terminal** (in Applications › Utilities), paste this line and press Return:

```bash
curl -fsSL https://raw.githubusercontent.com/ale316/macos-xplane-head-tracking/main/install.sh | bash
```

It installs everything into a folder called **X-Plane Head Tracker** in your home folder, and opens that folder in Finder when it's done. That folder contains:

- the code;
- a private copy of Python and the packages, set up with [uv](https://docs.astral.sh/uv/);
- the face model.

Nothing outside that folder is touched.

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

The global hotkeys need **Input Monitoring** permission for Terminal (System Settings › Privacy & Security › Input Monitoring). Without it, everything else still works.

## Tips

- **Light your face evenly from the front.** Dim light makes the camera noisier and slower, which you'll feel as jitter and lag.
- **An iPhone runs at 60 fps** instead of the built-in camera's 30, which noticeably cuts lag.
- **Switch the preview off while you fly** to save CPU. Tracking keeps running at full speed either way.
- **Leaning back** is limited to a few centimetres, so the view never ends up inside the seat. Leaning forward, e.g. to read an instrument, isn't limited.
- **Hands at your face** (touching your face, drinking, rubbing your eyes) don't throw the view around. While a hand overlaps your face, tracking keeps following you but is steadied, so the sudden jumps a hand causes are smoothed out. The preview marks hands it sees with a box.
- **A new flight or aircraft** starts looking straight ahead from that aircraft's eye point.
- The tracker only moves the view in X-Plane's 3D cockpit; outside views are left alone.

## Troubleshooting

| Problem | Fix |
|---|---|
| "X-Plane: waiting for 127.0.0.1:49000" | Make sure X-Plane is running, and check under X-Plane's Settings › Network that its UDP receive port is 49000. |
| iPhone not in the Camera list | macOS isn't offering it as a camera right now. For Continuity Camera, the iPhone must be locked, still (on a mount or stand) and near the Mac, with Wi-Fi and Bluetooth on and the same Apple ID on both. Also check that iPhone Settings › General › AirPlay & Continuity › Continuity Camera is on. A USB cable helps. |
| "Could not open the camera" | Allow Terminal under System Settings › Privacy & Security › Camera, or pick another camera. |
| An axis moves the wrong way | Set `"invert": true` for that axis in `config.json` (see below). |
| Laggy or jittery | Improve the lighting, then adjust the Smoothness slider. `./run.sh --diag diag.csv` (in the install folder) records timings you can attach to a bug report. |

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
| `xplane_host` / `xplane_port` | `127.0.0.1` / `49000` | Where X-Plane listens |

## How it works

1. **Head pose:** Google's [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) estimates your head's 3D pose from each camera frame.
2. **Filtering and curves:** the pose is filtered and run through response curves.
3. **Sending:** it's sent 60 times a second to X-Plane's built-in UDP interface as `sim/graphics/view/pilots_head_*` dataref writes.

The tracking runs in a windowless background process, because macOS slows down background apps that have a window. The preview is a separate process.

## Development

```bash
git clone https://github.com/ale316/macos-xplane-head-tracking.git && cd macos-xplane-head-tracking
bash install.sh                                     # .venv, .tools (uv, Python) and the model, all local
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
./run.sh --help                                     # --camera, --no-preview, --diag, --list-cameras ...
```

The code is in `headtrack/`:

| Module | Contents |
|---|---|
| `tracker.py` | Camera and face tracking |
| `pose.py` | Pose maths and filter |
| `xplane.py` | UDP link |
| `preview.py` | Preview window |
| `macos.py` | macOS integration |
| `app.py` | Command line and wiring |

## License

MIT, see [LICENSE](LICENSE). The face model is downloaded from Google at setup and is licensed under Apache 2.0.

Not affiliated with Laminar Research or X-Plane.
