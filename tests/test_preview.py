import copy

import cv2
import numpy as np

from headtrack.config import DEFAULT_CONFIG
from headtrack.preview import CTRL_H, SLIDER_X0, SLIDER_X1, CockpitView, PreviewUI


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


def test_range_slider_handles_are_symmetric_and_save_on_release():
    ui = make_ui()
    y = ui.controls_top() + 28
    mid, half = (SLIDER_X0 + SLIDER_X1) / 2, (SLIDER_X1 - SLIDER_X0) / 2
    click(ui, mid + half / 2, y)
    right = ui.range
    click(ui, mid - half / 2, y)
    assert right == ui.range and abs(right - 90) <= 1
    assert ui.cmd_q[-1] == ("range", ui.range, True)


def test_clicks_ask_for_an_immediate_redraw():
    ui = make_ui()
    ui.clicked = False
    press(ui, ("recenter",))
    assert ui.clicked and ui.cmd_q[-1] == ("recenter",)


def test_smoothness_slider():
    ui = make_ui()
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, SLIDER_X1, ui.controls_top() + 68, 0, None)
    assert ui.smoothing == 100 and ui.cmd_q[-1] == ("smoothing", 100, False)


def press(ui, cmd):
    (x0, y0, x1, y1), _ = next(b for b in ui.buttons if b[1] == cmd)
    click(ui, (x0 + x1) / 2, (y0 + y1) / 2)


def test_preview_button_switches_off_and_shrinks_window():
    ui = make_ui()
    press(ui, ("preview",))
    assert not ui.preview_on and ui.cmd_q[-1] == ("preview", False)
    assert ui.draw(None, None).shape[0] == ui.controls_top() + CTRL_H


def test_camera_button_cycles_rotation():
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
    ui.draw_controls(np.zeros((ui.controls_top() + CTRL_H, 800, 3), np.uint8), ui.controls_top(), st)
    press(ui, ("camera_menu",))
    assert ui.menu_open and ui.cmd_q[-1] == ("cameras",)  # asks the engine to refresh the list
    ui.draw_controls(np.zeros((ui.controls_top() + CTRL_H, 800, 3), np.uint8), ui.controls_top(), st)
    (x0, y0, x1, y1), cmd = ui.menu_items[1]
    click(ui, (x0 + x1) / 2, (y0 + y1) / 2)
    assert cmd == ("camera", 1) and ui.cmd_q[-1] == ("camera", 1) and not ui.menu_open


def test_buttons_fit_the_window():
    ui = make_ui()
    assert max(rect[2] for rect, _ in ui.buttons) < 800
