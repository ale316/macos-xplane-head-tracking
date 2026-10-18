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
- **Python 3.10 or newer.** If you have [uv](https://docs.astral.sh/uv/), it fetches Python by itself. Otherwise get Python from [python.org](https://www.python.org/downloads/macos/) or Homebrew (`brew install python@3.12`).

## Install

Download this repository (green **Code** button › **Download ZIP**, then unzip it), or clone it:

```bash
git clone https://github.com/ale316/macos-xplane-head-tracking.git
```

## Run

1. Start X-Plane, load a flight and switch to the 3D cockpit view (`Shift+9`).
2. Open Terminal, then type `bash ` and drag `run.sh` from the downloaded folder into the window. Press Return.

   The first run sets everything up inside that folder, which takes a minute or two. After that it starts straight away.

3. **Allow camera access** when macOS asks, then run it again.
4. Sit normally, look at the middle of the X-Plane screen and click **Recenter**.

To use your iPhone as the camera, run `bash run.sh --list-cameras` to find its number, then `bash run.sh --camera 1` (using that number).

### Controls

| | Preview window | While flying (global hotkey) |
|---|---|---|
| Recenter | **Recenter** button or `C` | `Ctrl+Shift+C` |
| Pause (hands the view back to X-Plane) | **Pause** button or `P` | `Ctrl+Shift+P` |
| Leaning on/off | **Lean** button or `L` | `Ctrl+Shift+L` |
| Preview on/off | **Preview** button or `V` | – |
| Quit | **Quit** button or `Q` | `Ctrl+C` in Terminal |

- **Look range:** how far the view turns when your head is fully turned. Both handles move together; ±90° means a full turn looks straight out the side window.
- **Smoothness:** from snappy to steady.
- **Camera:** auto-detects the camera's orientation; click it to force 0/90/180/270°.

Slider and camera settings are saved to `config.json` in the same folder.

The global hotkeys need **Input Monitoring** permission for Terminal (System Settings › Privacy & Security › Input Monitoring). Without it, everything else still works.

## Tips

- **Light your face evenly from the front.** Dim light makes the camera noisier and slower, which you'll feel as jitter and lag.
- **An iPhone runs at 60 fps** instead of the built-in camera's 30, which noticeably cuts lag.
- **Switch the preview off while you fly** to save CPU. Tracking keeps running at full speed either way.
- The tracker only moves the view in X-Plane's 3D cockpit; outside views are left alone.

## Troubleshooting

| Problem | Fix |
|---|---|
| "X-Plane: waiting for 127.0.0.1:49000" | Make sure X-Plane is running, and check under X-Plane's Settings › Network that its UDP receive port is 49000. |
| "Could not open camera" | Allow Terminal under System Settings › Privacy & Security › Camera, or try another `--camera` number. |
| An axis moves the wrong way | Set `"invert": true` for that axis in `config.json` (see below). |
| Laggy or jittery | Improve the lighting, then adjust the Smoothness slider. `bash run.sh --diag diag.csv` records timings you can attach to a bug report. |

## Configuration

`config.json` is created when you first change a setting; anything not in it uses the default. Useful keys:

| Key | Default | Meaning |
|---|---|---|
| `camera_index` | `0` | Which camera (same as `--camera`) |
| `camera_rotation` | `"auto"` | `"auto"` or a fixed rotation: `0`, `90`, `180`, `270` |
| `smoothing` | `40` | Smoothness slider, 0–100 |
| `lean_enabled` | `true` | Start with leaning on |
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
bash run.sh --help                                  # sets up .venv on first run
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
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
