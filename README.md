# X-Plane Head Tracker for Mac

[![Tests](https://github.com/ale316/macos-xplane-head-tracking/actions/workflows/tests.yml/badge.svg)](https://github.com/ale316/macos-xplane-head-tracking/actions/workflows/tests.yml)

Free head tracking for X-Plane 12 using your Mac's camera or your iPhone. Turn your head and the cockpit view follows. No IR clip, no plugin, and the video never leaves your Mac.

Requires a Mac with Apple Silicon (M1 or newer) and macOS 14 or newer.

## Install

Open Terminal, paste this line and press Return:

```bash
curl -fsSL https://raw.githubusercontent.com/ale316/macos-xplane-head-tracking/main/install.sh | bash
```

Everything goes into a folder called **X-Plane Head Tracker** in your home folder.

- **Update:** run the same line again.
- **Uninstall:** delete the folder.

## Use

1. In X-Plane, load a flight and switch to the 3D cockpit view (`Shift+9`).
2. Open **X-Plane Head Tracker** in that folder (drag it to the Dock to keep it handy). The first time, allow camera access.
3. In the preview window, pick your camera, look at the middle of the screen and click **Recenter**.

The preview window also has:

- **Look range:** how far the view turns.
- **Smoothness:** from snappy to steady.
- **Lean:** turn moving the view by leaning on or off.
- **Preview:** switch it off while you fly.
- **Turn with AirPods head tracking** (in the camera menu): AirPods Pro, Max, 3rd or 4th generation, or Beats with head tracking, turn the view with less lag, and hands or dim light don't disturb them. The camera still does leaning; with **Lean** off, it switches off.

Your settings are saved.

While flying, the hotkeys `Ctrl+Shift+C` (recenter), `Ctrl+Shift+P` (pause) and `Ctrl+Shift+L` (lean) work if X-Plane Head Tracker is allowed under System Settings › Privacy & Security › Input Monitoring.

## Tips

- **Even, frontal light** gives the steadiest tracking.
- **An iPhone** (via Continuity Camera) runs at 60 fps and feels snappier than the built-in camera. It needs to be locked, mounted and near the Mac.
- **X-Plane not connecting?** Make sure its UDP receive port is 49000 (Settings › Network).
- **Something wrong?** Run `./run.sh --diag diag.csv` in the install folder, reproduce it, and attach `diag.csv` to an [issue](https://github.com/ale316/macos-xplane-head-tracking/issues).

## Contributing

Issues and pull requests are welcome; [CONTRIBUTING.md](CONTRIBUTING.md) explains how the code fits together.

## License

Free for personal and other non-commercial use under the [PolyForm Noncommercial License 1.0.0](LICENSE). Built on [MediaPipe](https://github.com/google-ai-edge/mediapipe), OpenCV and PyObjC. Not affiliated with Laminar Research.
