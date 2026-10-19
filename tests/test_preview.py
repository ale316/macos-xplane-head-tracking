import copy
import pickle

import cv2
import numpy as np

from headtrack.cockpit import CockpitView
from headtrack.config import DEFAULT_CONFIG
from headtrack.preview import (
    CTRL_H,
    RANGE_ROW,
    SLIDER_X0,
    SLIDER_X1,
    SMOOTH_ROW,
    UI_W,
    PreviewUI,
    snapshot,
    status_lines,
)
from headtrack.state import Shared


def test_sim_view_turns_the_right_way():
    view = CockpitView(512)
    mh, mw = view.map.shape[:2]
    right = view.render(90, 0, 0, 64, 32, 60)[16, 32].astype(int)
    up = view.render(0, 45, 0, 64, 32, 60)[16, 32].astype(int)
    assert np.abs(right - view.map[mh // 2, int(mw * 0.75)].astype(int)).max() < 30
    assert np.abs(up - view.map[int(mh * 0.25), mw // 2].astype(int)).max() < 30


class Commands(list):
    put = list.append


def make_ui():
    ui = PreviewUI(copy.deepcopy(DEFAULT_CONFIG), Commands(), "")
    ui.draw(None, None)  # lays out the buttons
    return ui


def click(ui, x, y):
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, int(x), int(y), 0, None)
    ui.on_mouse(cv2.EVENT_LBUTTONUP, int(x), int(y), 0, None)


def press(ui, cmd):
    (x0, y0, x1, y1), _ = next(b for b in ui.buttons if b[1] == cmd)
    click(ui, (x0 + x1) / 2, (y0 + y1) / 2)


MID, HALF = (SLIDER_X0 + SLIDER_X1) / 2, (SLIDER_X1 - SLIDER_X0) / 2


def test_range_slider_handles_are_symmetric_and_save_on_release():
    ui = make_ui()
    y = ui.controls_top() + RANGE_ROW
    click(ui, MID + HALF / 2, y)
    right = ui.range
    click(ui, MID - HALF / 2, y)
    assert right == ui.range and abs(right - 90) <= 1
    assert ui.cmd_q[-1] == ("range", ui.range, True)


def test_smoothness_slider_follows_a_drag_and_saves_on_release():
    ui = make_ui()
    y = ui.controls_top() + SMOOTH_ROW
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, SLIDER_X0, y, 0, None)
    ui.on_mouse(cv2.EVENT_MOUSEMOVE, SLIDER_X1 + 50, y + 40, 0, None)  # the mouse can wander off the track
    assert ui.smoothing == 100 and ui.cmd_q[-1] == ("smoothing", 100, False)
    ui.on_mouse(cv2.EVENT_LBUTTONUP, MID, y, 0, None)
    assert ui.smoothing == 50 and ui.cmd_q[-1] == ("smoothing", 50, True) and ui.drag is None


def test_released_slider_does_not_jump_back_to_an_older_value():
    ui = make_ui()
    click(ui, MID + HALF / 2, ui.controls_top() + RANGE_ROW)  # set +/-90
    st = {"axes": {"yaw": {"output_max": 150.0}}, "smoothing": 40.0, "cam_rotation_mode": "auto"}
    ui.sync(st)  # an update sent before the engine got the new value
    assert ui.range == 90
    st["axes"]["yaw"]["output_max"] = 90.0
    ui.sync(st)  # confirmed
    st["axes"]["yaw"]["output_max"] = 120.0
    ui.sync(st)  # changed elsewhere later: follow it again
    assert ui.range == 120


def test_clicks_ask_for_an_immediate_redraw():
    ui = make_ui()
    ui.clicked = False
    press(ui, ("recenter",))
    assert ui.clicked and ui.cmd_q[-1] == ("recenter",)


def test_preview_button_switches_off_and_shrinks_window():
    ui = make_ui()
    press(ui, ("preview",))
    assert not ui.preview_on and ui.cmd_q[-1] == ("preview", False)
    assert ui.draw(None, None).shape[0] == ui.controls_top() + CTRL_H


def test_rotation_button_cycles_through_the_orientations():
    ui = make_ui()
    seen = []
    for _ in range(5):
        press(ui, ("rotation",))
        seen.append(ui.cmd_q[-1][1])
    assert seen == [0, 90, 180, 270, "auto"]


def test_camera_dropdown_lists_cameras_and_picks_one():
    ui = make_ui()
    st = {"cam_names": ["FaceTime HD Camera", "iPhone Camera"], "cam_index": 0, "cam_name": "FaceTime HD Camera",
          "cam_rotation": 0, "cam_rotation_mode": "auto", "paused": False, "lean": True}
    ui.draw_controls(np.zeros((ui.controls_top() + CTRL_H, UI_W, 3), np.uint8), ui.controls_top(), st)
    press(ui, ("camera_menu",))
    assert ui.menu_open and ui.cmd_q[-1] == ("cameras",)  # asks the engine to refresh the list
    ui.draw_controls(np.zeros((ui.controls_top() + CTRL_H, UI_W, 3), np.uint8), ui.controls_top(), st)
    (x0, y0, x1, y1), cmd = ui.menu_items[1]
    click(ui, (x0 + x1) / 2, (y0 + y1) / 2)
    assert cmd == ("camera", 1) and ui.cmd_q[-1] == ("camera", 1) and not ui.menu_open


def test_buttons_fit_the_window():
    ui = make_ui()
    assert max(rect[2] for rect, _ in ui.buttons) < UI_W


def test_full_window_draws_from_a_real_snapshot():
    sh = Shared()
    sh.frame = np.zeros((480, 640, 3), np.uint8)
    st = pickle.loads(pickle.dumps(snapshot(sh, DEFAULT_CONFIG, with_frame=True)))  # as sent to the preview
    frame = cv2.imdecode(np.frombuffer(st["jpeg"], np.uint8), cv2.IMREAD_COLOR)
    ui = make_ui()
    ui._build_cockpit()
    assert ui.draw(st, frame).shape == (ui.controls_top() + CTRL_H, UI_W, 3)


def test_status_explains_what_is_going_on():
    sh = Shared()
    assert status_lines(sh, DEFAULT_CONFIG)[0] == "X-Plane: waiting for 127.0.0.1:49000"
    sh.xp_connected, sh.xp_view, sh.face_ok = True, 1000, True
    assert "not in 3D cockpit view" in status_lines(sh, DEFAULT_CONFIG)[0]
    sh.cam_fps, sh.cam_target_fps = 15.0, 30
    assert "Video Effects" in status_lines(sh, DEFAULT_CONFIG)[0]  # a slow camera gets a hint
    sh.hand_at_face = True
    assert "hand at face - steadied" in status_lines(sh, DEFAULT_CONFIG)[1]
    sh.error = "x" * 500
    lines = status_lines(sh, DEFAULT_CONFIG)
    assert len(lines) == 3 and lines[-1].endswith("(see terminal)")


def test_camera_menu_switches_airpods_on_and_off():
    ui = make_ui()
    st = {"cam_names": ["FaceTime HD Camera"], "cam_index": 0, "cam_name": "FaceTime HD Camera",
          "cam_rotation": 0, "cam_rotation_mode": "auto", "paused": False, "lean": True, "airpods_status": ""}
    for status, sent in (("", ("airpods", True)), ("on", ("airpods", False))):
        st["airpods_status"] = status
        press(ui, ("camera_menu",))
        ui.draw_controls(np.zeros((ui.controls_top() + CTRL_H, UI_W, 3), np.uint8), ui.controls_top(), st)
        (x0, y0, x1, y1), cmd = ui.menu_items[-1]
        click(ui, (x0 + x1) / 2, (y0 + y1) / 2)
        assert ui.cmd_q[-1] == sent and not ui.menu_open
