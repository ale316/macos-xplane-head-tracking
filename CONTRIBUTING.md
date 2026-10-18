# Contributing

## Setup

```bash
bash install.sh                                    # Python, packages and models, into this folder
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest                         # no camera or X-Plane needed
.venv/bin/ruff check .
./run.sh                                           # runs this working copy
```

## How it fits together

It runs as two processes, because macOS throttles background processes that have a window, and tracking must keep full speed while X-Plane is in front.

- **Engine** (`app.py`), with no window, runs these threads around a shared `state.Shared` (hold its `lock`):
  - `tracker.py`: camera frames (`camera.py`, AVFoundation) → MediaPipe face landmarks → head pose relative to the recentered pose, filtered by `pose.py`. A second model watches for a hand at the face.
  - `xplane.py`: 60 times a second, maps the pose through the axis curves and writes X-Plane's head datarefs over its UDP interface. The seat position comes from the aircraft's `.acf` file.
  - `app.Controls`: commands from the preview window and the global hotkeys (`macos.py`).
- **Preview** (`preview.py`, `cockpit.py`): the window. It draws snapshots of the engine's state and sends commands back over a queue.

Defaults and their explanations are in `config.py`; `config.json` only keeps what the user changed.

## Debugging

`./run.sh --diag diag.csv` writes timing events (`state.diag`) to a CSV: camera frames, poses, X-Plane ticks and preview redraws.

## Pull requests

- Keep `pytest` and `ruff check` passing (CI runs both), and add a test when behavior changes.
- Changes to tracking or the X-Plane link also need a try with X-Plane running; say how you tested.
