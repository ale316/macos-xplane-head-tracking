"""Camera capture straight from AVFoundation.

Cameras are identified by their unique ID rather than a number: the order macOS lists them
in changes as an iPhone connects or the laptop lid closes, so numbers can point at the wrong
camera. Opening the device ourselves also lets us pick its fastest format and pin the frame
rate (otherwise auto-exposure halves it in dim light, doubling the camera's latency).
"""
from __future__ import annotations

import os
import platform
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraInfo:
    id: str
    name: str


_listed_once = False


@contextmanager
def _quiet_stderr():
    """AVFoundation logs a deprecation warning about Continuity Camera the first time devices
    are listed. Silences the process's stderr meanwhile, so only use it for that first time."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def _devices() -> list:
    import AVFoundation as AV

    global _listed_once
    # USB cameras and iPhones are "External" from macOS 14; before that "ExternalUnknown".
    external = AV.AVCaptureDeviceTypeExternal if _macos_major() >= 14 else AV.AVCaptureDeviceTypeExternalUnknown
    with nullcontext() if _listed_once else _quiet_stderr():
        session = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
            [AV.AVCaptureDeviceTypeBuiltInWideAngleCamera, external], AV.AVMediaTypeVideo,
            AV.AVCaptureDevicePositionUnspecified)
        devices = session.devices()
    _listed_once = True
    # skip suspended cameras, e.g. the built-in one with the lid closed (macOS 14+ knows this)
    return [d for d in devices if not (d.respondsToSelector_("isSuspended") and d.isSuspended())]


def _macos_major() -> int:
    try:
        return int(platform.mac_ver()[0].split(".")[0])
    except ValueError:
        return 0


def list_cameras() -> list[CameraInfo]:
    """Cameras that can be used right now."""
    try:
        return [CameraInfo(str(d.uniqueID()), str(d.localizedName())) for d in _devices()]
    except ImportError:
        return []


def pick_camera(cfg: dict, cameras: list[CameraInfo]) -> CameraInfo | None:
    """The saved camera if it's connected (by ID, then name), else the first one."""
    for cam in cameras:
        if cam.id == cfg.get("camera_id"):
            return cam
    for cam in cameras:
        if cam.name == cfg.get("camera_name"):
            return cam
    return cameras[0] if cameras else None


_sink_class = None


def _frame_sink():
    """An Objective-C object that hands AVFoundation's frames to a Camera (class made once)."""
    global _sink_class
    if _sink_class is None:
        import objc
        from Foundation import NSObject

        delegate = objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")

        class HeadtrackFrameSink(NSObject, protocols=[delegate]):
            def captureOutput_didOutputSampleBuffer_fromConnection_(self, _output, sample, _connection):
                self.camera._on_frame(sample)

        _sink_class = HeadtrackFrameSink
    return _sink_class.alloc().init()


def camera_to_use(cfg: dict, cameras: list[CameraInfo], current_id: str) -> CameraInfo | None:
    """The camera to switch to now, or None to stay: the chosen one as soon as it's available
    (e.g. an iPhone that connects late), otherwise any camera if the current one has gone."""
    if not cameras:
        return None
    chosen = next((c for c in cameras if c.id == cfg.get("camera_id")), None) or \
        next((c for c in cameras if cfg.get("camera_name") and c.name == cfg.get("camera_name")), None)
    if chosen is not None and chosen.id != current_id:
        return chosen
    if all(c.id != current_id for c in cameras):
        return cameras[0]
    return None


def camera_access() -> bool:
    """Ask for camera permission if it hasn't been decided yet; True when allowed."""
    import AVFoundation as AV

    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    if status == AV.AVAuthorizationStatusNotDetermined:
        done, result = threading.Event(), {}

        def answered(granted):
            result["ok"] = bool(granted)
            done.set()

        AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeVideo, answered)
        done.wait(120)
        return result.get("ok", False)
    return status == AV.AVAuthorizationStatusAuthorized


class Camera:
    """One open camera delivering BGR frames (numpy) of about width x height."""

    def __init__(self, camera_id: str, width: int, height: int, max_fps: float):
        import AVFoundation as AV
        import CoreMedia as CM

        device = next((d for d in _devices() if str(d.uniqueID()) == camera_id), None)
        if device is None:
            raise RuntimeError("camera not available")
        self.name = str(device.localizedName())
        self._cond = threading.Condition()
        self._frame, self._pts, self._seq = None, 0.0, 0

        session = AV.AVCaptureSession.alloc().init()
        session.beginConfiguration()
        inp, err = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if inp is None or not session.canAddInput_(inp):
            raise RuntimeError(f"can't open camera: {err}")
        session.addInput_(inp)

        # The fastest format of the wanted size (or the smallest larger one), at up to max_fps.
        def score(f):
            dims = CM.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
            top = max((r.maxFrameRate() for r in f.videoSupportedFrameRateRanges()), default=0.0)
            exact = (dims.width, dims.height) == (width, height)
            big_enough = dims.width >= width and dims.height >= height
            return exact, big_enough, min(top, max_fps), -dims.width * dims.height
        best = max(device.formats(), key=score)
        self.fps = int(min(max_fps, max(r.maxFrameRate() for r in best.videoSupportedFrameRateRanges())))
        self._device = device if device.lockForConfiguration_(None)[0] else None
        self.locked = self._device is not None  # we hold the format and frame rate
        self._format = best
        try:
            self._start(session, best, width)
        except Exception:
            if self._device is not None:  # don't leave the camera locked to us
                self._device.unlockForConfiguration()
            raise
        active = CM.CMVideoFormatDescriptionGetDimensions(device.activeFormat().formatDescription())
        frame = device.activeVideoMinFrameDuration()
        self.mode = (f"{active.width}x{active.height} at {frame.timescale / max(frame.value, 1):.0f} fps"
                     + ("" if self.locked else " (couldn't lock the camera's settings)"))

    def _start(self, session, best, width: int):
        import AVFoundation as AV
        import CoreMedia as CM
        from libdispatch import dispatch_queue_create
        from Quartz import (
            kCVPixelBufferHeightKey,
            kCVPixelBufferPixelFormatTypeKey,
            kCVPixelBufferWidthKey,
            kCVPixelFormatType_32BGRA,
        )

        self._apply_format()  # with the device in the session, this also stops presets overriding the format

        dims = CM.CMVideoFormatDescriptionGetDimensions(best.formatDescription())
        out_w = width
        out_h = round(width * dims.height / dims.width / 2) * 2  # keep the camera's aspect ratio
        output = AV.AVCaptureVideoDataOutput.alloc().init()
        output.setVideoSettings_({kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                                  kCVPixelBufferWidthKey: out_w, kCVPixelBufferHeightKey: out_h})
        output.setAlwaysDiscardsLateVideoFrames_(True)
        self._sink = _frame_sink()
        self._sink.camera = self
        self._queue = dispatch_queue_create(b"headtrack.camera", None)
        output.setSampleBufferDelegate_queue_(self._sink, self._queue)
        if not session.canAddOutput_(output):
            raise RuntimeError("can't read from camera")
        session.addOutput_(output)
        session.commitConfiguration()
        session.startRunning()
        self._apply_format()  # starting the session can switch formats; set it again
        self._session = session

    def _apply_format(self):
        """Fastest format and a fixed frame rate. The device stays locked while open, so
        nothing renegotiates them (e.g. auto-exposure halving the frame rate in dim light)."""
        import CoreMedia as CM

        if self._device is None:
            return
        if self._device.activeFormat() != self._format:
            self._device.setActiveFormat_(self._format)
        frame_time = CM.CMTimeMake(1, self.fps)
        self._device.setActiveVideoMinFrameDuration_(frame_time)
        self._device.setActiveVideoMaxFrameDuration_(frame_time)

    def _on_frame(self, sample):
        import CoreMedia as CM
        import Quartz as Q

        pixels = CM.CMSampleBufferGetImageBuffer(sample)
        Q.CVPixelBufferLockBaseAddress(pixels, Q.kCVPixelBufferLock_ReadOnly)
        try:
            h, w = Q.CVPixelBufferGetHeight(pixels), Q.CVPixelBufferGetWidth(pixels)
            row = Q.CVPixelBufferGetBytesPerRow(pixels)
            data = Q.CVPixelBufferGetBaseAddress(pixels).as_buffer(row * h)
            bgr = np.frombuffer(data, np.uint8).reshape(h, row // 4, 4)[:, :w, :3].copy()
        finally:
            Q.CVPixelBufferUnlockBaseAddress(pixels, Q.kCVPixelBufferLock_ReadOnly)
        pts = CM.CMSampleBufferGetPresentationTimeStamp(sample)
        with self._cond:
            self._frame, self._pts = bgr, pts.value / max(pts.timescale, 1)
            self._seq += 1
            self._cond.notify_all()

    def read(self, timeout: float = 1.0):
        """Waits for the next frame: (frame or None, the camera's capture timestamp in seconds)."""
        with self._cond:
            seq = self._seq
            self._cond.wait_for(lambda: self._seq != seq, timeout)
            return (self._frame, self._pts) if self._seq != seq else (None, 0.0)

    def close(self):
        self._session.stopRunning()
        if self._device is not None:
            self._device.unlockForConfiguration()
