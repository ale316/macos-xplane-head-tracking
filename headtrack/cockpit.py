"""The preview's simulated view: a stand-in for X-Plane's 3D cockpit, to show what the head pose does."""
from __future__ import annotations

import math

import numpy as np

from .pose import rot_x, rot_y, rot_z


class CockpitView:
    """A procedural 360-degree cockpit (window posts, glareshield and gauges, cabin walls, sky,
    ground grid, runway), rendered for a head direction and field of view."""

    def __init__(self, width: int = 2048):
        h = width // 2
        lon = (np.arange(width) + 0.5) / width * 360 - 180  # 0 = straight ahead, + = right
        lat = 90 - (np.arange(h) + 0.5) / h * 180  # + = up
        LON, LAT = np.meshgrid(lon, lat)
        a = np.abs(LON)
        img = np.zeros((h, width, 3), np.float32)
        sky = np.clip(LAT / 60, 0, 1)[..., None]
        img[:] = (1 - sky) * np.array([235, 215, 190]) + sky * np.array([200, 140, 70])  # BGR
        below = LAT < 0
        d = 2.0 / np.tan(np.radians(np.maximum(-LAT, 0.3)))  # distance on the ground, eye 2 m up
        gx, gz = d * np.sin(np.radians(LON)), d * np.cos(np.radians(LON))
        # 25 m grid, offset half a cell so no line runs through the viewer (it would show as a wedge)
        grid = (np.abs((gx / 25 + 0.5) % 1 - 0.5) > 0.47) | (np.abs((gz / 25 + 0.5) % 1 - 0.5) > 0.47)
        ground = np.where(grid[..., None], [70, 110, 80], [80, 140, 95]).astype(np.float32)
        haze = np.clip(-LAT / 4, 0, 1)[..., None]
        ground = haze * ground + (1 - haze) * np.array([200, 200, 190])
        runway = (np.abs(gx) < 12) & (gz > 0)
        ground[runway] = [110, 110, 110]
        ground[runway & (np.abs(gx) < 0.6) & ((gz / 12) % 1 < 0.5)] = [235, 235, 235]
        img[below] = ground[below]
        cabin, frame = np.array([60, 60, 66], np.float32), np.array([35, 35, 38], np.float32)
        img[LAT > np.where(a < 70, 28, 22)] = cabin * 0.8  # roof
        img[(a > 55) & (LAT < -8)] = cabin  # side walls below the side windows
        rear = a > 125
        img[rear] = cabin * 0.9  # back of the cabin
        for c, half in ((0, 0.8), (27, 2.5), (60, 3.5), (110, 4.0)):  # windscreen divider, A/B/C posts
            img[(np.abs(a - c) < half) & ~rear] = frame
        img[(np.abs(LAT - 28) < 1.2) & (a < 70)] = frame
        edge = -10 - 6 * (a / 55) ** 2  # curved glareshield
        img[(a < 55) & (LAT < edge)] = [45, 45, 48]
        img[(np.abs(LAT - edge) < 0.8) & (a < 55)] = [25, 25, 25]
        for gl in (-30, -18, -6):  # six-pack
            for gb in (-19, -30):
                r = np.hypot((LON - gl) * np.cos(np.radians(gb)), LAT - gb)
                img[r < 5] = [20, 20, 20]
                img[(r > 4.6) & (r < 5)] = [200, 200, 200]
                img[r < 0.6] = [230, 230, 230]
        for gl in (10, 22, 34):  # radio stack / screens
            img[(np.abs(LON - gl) < 5) & (np.abs(LAT + 24) < 9)] = [70, 50, 30]
        self.map = img.astype(np.uint8)
        self._rays_key, self._rays = None, None

    def render(self, yaw: float, pitch: float, roll: float, w: int, h: int, fov_deg: float, zoom: float = 1.0):
        """yaw right+, pitch up+, roll right+ (degrees); horizontal FOV; zoom > 1 = leaning in."""
        import cv2

        fov = 2 * math.degrees(math.atan(math.tan(math.radians(fov_deg) / 2) / zoom))
        if self._rays_key != (w, h, round(fov, 1)):
            f = (w / 2) / math.tan(math.radians(fov) / 2)
            u, v = np.meshgrid(np.arange(w) - w / 2 + 0.5, h / 2 - np.arange(h) - 0.5)
            r = np.stack([u, v, np.full_like(u, f)], -1).reshape(-1, 3)
            self._rays = (r / np.linalg.norm(r, axis=1, keepdims=True)).astype(np.float32)
            self._rays_key = (w, h, round(fov, 1))
        R = rot_y(yaw) @ rot_x(-pitch) @ rot_z(-roll)
        d = self._rays @ R.T.astype(np.float32)
        mh, mw = self.map.shape[:2]
        mx = np.arctan2(d[:, 0], d[:, 2]) * (mw / (2 * math.pi)) + mw / 2
        my = mh / 2 - np.arcsin(np.clip(d[:, 1], -1, 1)) * (mh / math.pi)
        return cv2.remap(self.map, mx.reshape(h, w).astype(np.float32), my.reshape(h, w).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
