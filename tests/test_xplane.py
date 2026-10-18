"""X-Plane link against a fake X-Plane that answers RREF subscriptions and records DREF writes."""
import copy
import socket
import struct
import threading
import time

import numpy as np
import pytest

from headtrack import xplane
from headtrack.config import AXES, DEFAULT_CONFIG
from headtrack.state import Shared
from headtrack.xplane import (DATAREFS, FLIGHT_TIME_DREF, FOV_DREF, VIEW_3D_COCKPIT, VIEW_TYPE_DREF, XPlaneLink,
                              acf_eye_point)

AQUILA_ACF = """P acf/_cgX_lft -0.010000000
P acf/_cgX_rgt 0.010000000
P acf/_cgY -0.560000002
P acf/_cgZ 7.150000095
P acf/_pe_xyz/0 -0.720000029
P acf/_pe_xyz/1 1.480000019
P acf/_pe_xyz/2 8.689999580
"""
AQUILA_EYE = (-0.219, 0.622, 0.469)  # what X-Plane itself reports for this aircraft


def test_acf_eye_point_matches_xplane():
    assert acf_eye_point(AQUILA_ACF) == pytest.approx(AQUILA_EYE, abs=0.001)
    assert acf_eye_point("P acf/_cgY 1") is None


class FakeXPlane:
    """Like X-Plane, the head position is shared with the eye point: writing it moves both."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.02)
        self.port = self.sock.getsockname()[1]
        # X-Plane's head: a bit off the seat and panned left (e.g. left over from the hat switch)
        self.values = {DATAREFS[a]: v for a, v in zip(AXES, [-40.0, -5.0, 0.0, -0.2, 0.6, 0.5])}
        self.values[VIEW_TYPE_DREF] = float(VIEW_3D_COCKPIT)
        self.values[FOV_DREF] = 72.0
        self.values[FLIGHT_TIME_DREF] = 100.0
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
                self.values[FLIGHT_TIME_DREF] += 0.05
                for idx, sub in list(self.subs.items()):
                    if sub:
                        self.sock.sendto(b"RREF," + struct.pack("<if", idx, self.values.get(sub[0], 0.0)), sub[1])

    def __getitem__(self, axis):
        return self.values[DATAREFS[axis]]


def xplane_folder(tmp_path, acf_text):
    """A fake X-Plane folder whose log says the Aquila was loaded."""
    (tmp_path / "Aircraft").mkdir(exist_ok=True)
    (tmp_path / "Aircraft" / "Aquila.acf").write_text(acf_text)
    (tmp_path / "Log.txt").write_text("0:00:19.111 I/ACF: Loading airplane number 0 with Aircraft/Aquila.acf\n")
    return tmp_path


def start(monkeypatch, folder, raw=(30.0, 10.0, 0.0, 0.0, 0.0, 10.0)):
    monkeypatch.setattr(xplane, "xplane_folder", lambda: folder)  # never look at a real X-Plane
    xp = FakeXPlane()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(xplane_port=xp.port, sounds=False)
    for a in AXES:  # identity curves: degrees in = degrees out, 1 cm in = 0.01 m out
        cfg["axes"][a].update(deadzone=0.0, exponent=1.0, input_max=1000.0,
                              output_max=1000.0 if a in ("yaw", "pitch", "roll") else 10.0)
    sh = Shared()
    sh.has_ref = sh.face_ok = True
    sh.raw = np.array(raw)
    link = XPlaneLink(cfg, sh)
    link.start()
    time.sleep(1.5)
    return xp, sh, link


def stop(xp, sh, link):
    with sh.lock:
        sh.quit = True
    link.join(2)
    time.sleep(0.3)  # let the fake X-Plane read the unsubscribe packets
    xp.stop.set()
    xp.thread.join(1)


@pytest.fixture(params=["aircraft file", "no aircraft file"])
def setup(request, monkeypatch, tmp_path):
    folder = xplane_folder(tmp_path, AQUILA_ACF) if request.param == "aircraft file" else None
    xp, sh, link = start(monkeypatch, folder)
    yield xp, sh, link, (AQUILA_EYE if folder else (-0.2, 0.6, 0.5))
    stop(xp, sh, link)


def test_drives_view_from_the_seat_looking_straight_ahead(setup):
    xp, sh, _, seat = setup
    assert sh.xp_connected and sh.xp_active and sh.xp_fov == 72.0
    assert xp["yaw"] == pytest.approx(30.0, abs=0.5)  # X-Plane's panned -40 isn't the reference
    assert xp["pitch"] == pytest.approx(-5.0 + 10, abs=0.5)  # but its default view pitch is
    assert xp["x"] == pytest.approx(seat[0], abs=0.01)
    assert xp["z"] == pytest.approx(seat[2] - 0.1, abs=0.02)  # leaning forward = towards -z in X-Plane


def test_own_writes_never_move_the_reference(setup):
    xp, _, link, seat = setup
    time.sleep(1.0)  # X-Plane's head (= its eye point) has been showing our offset view all along
    assert link.base[3:] == pytest.approx(seat, abs=0.01)
    assert xp["z"] == pytest.approx(seat[2] - 0.1, abs=0.02)  # no creeping


def test_pause_restores_the_seat_and_resume_continues(setup):
    xp, sh, _, seat = setup
    with sh.lock:
        sh.paused = True
    time.sleep(0.3)
    assert xp["yaw"] == pytest.approx(0.0, abs=1e-4) and xp["z"] == pytest.approx(seat[2], abs=1e-3)
    with sh.lock:
        sh.paused = False
    time.sleep(1.0)
    assert sh.xp_active and xp["yaw"] == pytest.approx(30.0, abs=0.5)


def test_lean_off_eases_translation_back(setup):
    xp, sh, _, seat = setup
    with sh.lock:
        sh.lean = False
    time.sleep(1.5)
    assert xp["z"] == pytest.approx(seat[2], abs=0.01) and xp["yaw"] == pytest.approx(30.0, abs=0.5)


def test_idle_outside_cockpit_and_unsubscribes(setup):
    xp, sh, link, _ = setup
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


def test_new_aircraft_moves_the_seat(monkeypatch, tmp_path):
    folder = xplane_folder(tmp_path, AQUILA_ACF)
    xp, sh, link = start(monkeypatch, folder, raw=(0, 0, 0, 0, 0, 0))
    assert xp["x"] == pytest.approx(AQUILA_EYE[0], abs=0.01)
    (folder / "Aircraft" / "Other.acf").write_text(AQUILA_ACF.replace("_pe_xyz/0 -0.72", "_pe_xyz/0 1.28"))
    with open(folder / "Log.txt", "a") as log:
        log.write("0:10:00.000 I/ACF: Loading airplane number 0 with Aircraft/Other.acf\n")
    time.sleep(1.5)
    assert xp["x"] == pytest.approx(AQUILA_EYE[0] + 2 * 0.3048, abs=0.01)
    stop(xp, sh, link)


def test_new_flight_retakes_the_seat_without_aircraft_file(monkeypatch):
    xp, sh, link = start(monkeypatch, None, raw=(0, 0, 0, 0, 0, 0))
    assert xp["x"] == pytest.approx(-0.2, abs=0.01)
    with sh.lock:
        sh.paused = True  # e.g. X-Plane busy loading; the new flight puts the head in the new seat
    time.sleep(0.3)
    xp.values[DATAREFS["x"]] = 0.7
    xp.values[FLIGHT_TIME_DREF] = 0.0
    time.sleep(0.3)
    with sh.lock:
        sh.paused = False
    time.sleep(1.2)
    assert xp["x"] == pytest.approx(0.7, abs=0.01)
    stop(xp, sh, link)
