"""X-Plane link against a fake X-Plane that answers RREF subscriptions and records DREF writes."""
import copy
import socket
import struct
import threading
import time

import numpy as np
import pytest

from headtrack.config import AXES, DEFAULT_CONFIG
from headtrack.state import Shared
from headtrack.xplane import DATAREFS, FOV_DREF, VIEW_3D_COCKPIT, VIEW_TYPE_DREF, XPlaneLink


class FakeXPlane:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.02)
        self.port = self.sock.getsockname()[1]
        self.values = {DATAREFS[a]: v for a, v in zip(AXES, [0.0, -5.0, 0.0, -0.3, 0.9, -1.2])}
        self.values[VIEW_TYPE_DREF] = float(VIEW_3D_COCKPIT)
        self.values[FOV_DREF] = 72.0
        self.subs, self.writes = {}, []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        last = 0.0
        while not self.stop.is_set():
            try:
                data, addr = self.sock.recvfrom(2048)
                if data[:5] == b"RREF\x00":
                    freq, idx, name = struct.unpack("<ii400s", data[5:])
                    self.subs[idx] = (name.rstrip(b"\x00").decode(), addr) if freq else None
                elif data[:5] == b"DREF\x00":
                    val, name = struct.unpack("<f500s", data[5:])
                    name = name.split(b"\x00")[0].decode()
                    self.values[name] = val
                    self.writes.append((name, val))
            except socket.timeout:
                pass
            if time.monotonic() - last > 0.05:
                last = time.monotonic()
                for idx, sub in list(self.subs.items()):
                    if sub:
                        self.sock.sendto(b"RREF," + struct.pack("<if", idx, self.values[sub[0]]), sub[1])

    def __getitem__(self, axis):
        return self.values[DATAREFS[axis]]


@pytest.fixture
def setup():
    xp = FakeXPlane()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(xplane_port=xp.port, sounds=False)
    for a in AXES:  # identity curves: degrees in = degrees out, 1 cm in = 0.01 m out
        cfg["axes"][a].update(deadzone=0.0, exponent=1.0, input_max=1000.0,
                              output_max=1000.0 if a in ("yaw", "pitch", "roll") else 10.0)
    sh = Shared()
    sh.has_ref = sh.face_ok = True
    sh.raw = np.array([30.0, 10.0, 0.0, 0.0, 0.0, 10.0])  # 30 right, 10 up, 10 cm forward
    link = XPlaneLink(cfg, sh)
    link.start()
    time.sleep(1.5)
    yield xp, sh, link
    with sh.lock:
        sh.quit = True
    link.join(2)
    time.sleep(0.3)  # let the fake X-Plane read the unsubscribe packets
    xp.stop.set()
    xp.thread.join(1)


def test_drives_view_relative_to_xplane_base(setup):
    xp, sh, _ = setup
    assert sh.xp_connected and sh.xp_active
    assert sh.xp_fov == 72.0
    assert xp["yaw"] == pytest.approx(30.0, abs=0.5)
    assert xp["pitch"] == pytest.approx(5.0, abs=0.5)  # base -5 + 10
    assert xp["z"] == pytest.approx(-1.3, abs=0.02)  # leaning forward = towards -z in X-Plane


def test_pause_restores_view_and_resume_continues(setup):
    xp, sh, _ = setup
    with sh.lock:
        sh.paused = True
    time.sleep(0.3)
    assert xp["yaw"] == pytest.approx(0.0, abs=1e-4) and xp["pitch"] == pytest.approx(-5.0, abs=1e-4)
    with sh.lock:
        sh.paused = False
    time.sleep(1.0)
    assert sh.xp_active and xp["yaw"] == pytest.approx(30.0, abs=0.5)


def test_lean_off_eases_translation_back(setup):
    xp, sh, _ = setup
    with sh.lock:
        sh.lean = False
    time.sleep(1.5)
    assert xp["z"] == pytest.approx(-1.2, abs=0.01) and xp["yaw"] == pytest.approx(30.0, abs=0.5)


def test_idle_outside_cockpit_and_unsubscribes(setup):
    xp, sh, link = setup
    xp.values[VIEW_TYPE_DREF] = 1000.0  # an outside view
    time.sleep(0.4)
    n = len(xp.writes)
    time.sleep(0.3)
    assert not sh.xp_active and len(xp.writes) == n
    with sh.lock:
        sh.quit = True
    link.join(2)
    time.sleep(0.3)
    assert all(v is None for v in xp.subs.values())
