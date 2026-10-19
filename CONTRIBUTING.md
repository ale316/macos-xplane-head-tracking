# Contributing

## Setup

```bash
bash install.sh                                    # Python, packages and models, into this folder
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest                         # no camera or X-Plane needed
.venv/bin/ruff check .
./run.sh                                           # runs this working copy
open "X-Plane Head Tracker.app"                     # the same, as the app (made by install.sh; AirPods need it)
```

## How it fits together

It runs as two processes, because macOS throttles background processes that have a window, and tracking must keep full speed while X-Plane is in front.

- **Engine** (`app.py`), with no window, runs these threads around a shared `state.Shared` (hold its `lock`):
  - `tracker.py`: camera frames (`camera.py`, AVFoundation) → MediaPipe face landmarks → head pose relative to the recentered pose, filtered by `pose.py`. A second model watches for a hand at the face.
  - `xplane.py`: 60 times a second, maps the pose through the axis curves and writes X-Plane's head datarefs over its UDP interface. The seat position comes from the aircraft's `.acf` file.
  - `airpods.py` (when switched on): head rotation from AirPods head tracking (CoreMotion), lined up with and drift-corrected by the camera. `xplane.py` uses it instead of the camera's rotation while it's there. With lean off as well, the tracker closes the camera until it's needed again.
  - `app.Controls`: commands from the preview window and the global hotkeys (`macos.py`).
- **Preview** (`preview.py`, `cockpit.py`): the window. It draws snapshots of the engine's state and sends commands back over a queue.

Defaults and their explanations are in `config.py`; `config.json` only keeps what the user changed.

macOS asks for camera, motion and Input Monitoring access for the app it was opened as: the app bundle `install.sh` makes (it runs `.venv/bin/python -m headtrack` here), or Terminal. Reading headphone motion kills a process whose app doesn't declare it, so AirPods need the app.

## Debugging

`./run.sh --diag diag.csv` writes timing events (`state.diag`) to a CSV: camera frames, poses, X-Plane ticks and preview redraws.

## Pull requests

- Keep `pytest` and `ruff check` passing (CI runs both), and add a test when behavior changes.
- Changes to tracking or the X-Plane link also need a try with X-Plane running; say how you tested.
