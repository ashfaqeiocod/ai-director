"""Virtual cameras: the close-up operator must be causal, and crops must map points both ways."""
import numpy as np

from studio.cameras import VirtualOperator, group_center
from studio.video import crop_to, to_camera, to_source

CFG = {"zoom_width": 960, "smoothing": 0.1, "max_speed_px": 14, "dead_zone_px": 40}


def run(aims):
    op = VirtualOperator(CFG, 4096, 1080)
    return [op.update(a) for a in aims]


def test_operator_never_uses_the_future():
    rng = np.random.default_rng(1)
    aims = [tuple(rng.uniform([0, 0], [4096, 1080])) if rng.random() > 0.2 else None for _ in range(300)]
    full = run(aims)
    for cut in (1, 50, 299):
        assert run(aims[:cut]) == full[:cut]          # frame t's rect is the same whatever comes later


def test_operator_limits_speed_and_stays_inside():
    rects = run([(100, 100)] + [(4000, 1000)] * 50)
    xs = [r[0] for r in rects]
    assert max(abs(b - a) for a, b in zip(xs, xs[1:])) <= CFG["max_speed_px"] + 1e-6
    for x, y, w, h in rects:
        assert x >= 0 and y >= 0 and x + w <= 4096 + 1e-6 and y + h <= 1080 + 1e-6


def test_group_center_picks_the_crowd():
    pitch = np.array([[0, 0], [1, 1], [2, 0], [40, 30]], dtype=float)   # 3 together, 1 alone
    image = np.array([[100, 100], [110, 100], [120, 100], [3000, 900]], dtype=float)
    x, y = group_center(pitch, image, radius=5)
    assert abs(x - 110) < 1e-6 and abs(y - 100) < 1e-6


def test_crop_point_mapping_round_trip():
    rect = (1328, 150, 1440, 810)
    for p in [(1328, 150), (2000, 500), (2767, 959)]:
        cam = to_camera(rect, 1280, 720, p)
        back = to_source(rect, 1280, 720, cam)
        assert np.allclose(back, p)
    image = np.zeros((1080, 4096, 3), np.uint8)
    assert crop_to(image, rect, 1280, 720).shape == (720, 1280, 3)
    assert crop_to(image, (0, 0, 4096, 1080), 1280, 720).shape == (720, 1280, 3)   # letterboxed wide
