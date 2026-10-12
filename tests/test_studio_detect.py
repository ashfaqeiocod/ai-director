"""Detector on the panorama: a spare ball parked off the pitch must not be taken for the match ball."""
from studio.detect import PanoramaDetections

STEP = 5


def pano(balls_per_frame: dict[int, list[tuple[float, float, float]]]) -> PanoramaDetections:
    """A PanoramaDetections without files: _balls() returns the given panorama balls per frame."""
    p = PanoramaDetections.__new__(PanoramaDetections)
    p.step, p.parked_window, p.parked_radius, p.parked_share, p.parked_history = STEP, 20, 20, 0.4, 3
    p.dets = {c: {"frames": {}} for c in (1, 2, 3)}       # no players
    p._balls = lambda f: balls_per_frame.get(f, [])
    return p


def test_parked_ball_is_ignored_and_moving_ball_kept():
    frames = range(0, 200 * STEP, STEP)
    # spare ball always at (3660, 660) with high confidence; the match ball moves 50 px per detection
    balls = {f: [(3660.0, 660.0, 0.9), (500.0 + 10 * f, 400.0, 0.5)] for f in frames}
    p = pano(balls)
    for f in list(frames)[10:]:
        assert p.at(f)["ball"][:2] == (500.0 + 10 * f, 400.0)


def test_parked_check_reads_only_the_past():
    balls = {f: [(3660.0, 660.0, 0.9)] for f in range(0, 100 * STEP, STEP)}
    later = dict(balls)
    later.update({f: [] for f in range(50 * STEP, 100 * STEP, STEP)})   # the future changes ...
    for f in range(0, 50 * STEP, STEP):
        assert pano(balls).at(f) == pano(later).at(f)                      # ... frame f's answer doesn't
