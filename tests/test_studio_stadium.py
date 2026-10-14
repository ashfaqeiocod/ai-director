"""MVP 4 (virtual stadium): the panorama geometry, on a synthetic 3-camera rig whose true answers we know.

The rig: 3 cameras at one spot 12 m high behind the near touchline, turned left / straight / right, like Alfheim.
"""
import math

import cv2
import numpy as np
import pytest

from studio.panorama import CAMS, SRC_H, SRC_W, Panorama, Stitcher, fit_rig

CENTRE = np.array([-3.0, 42.0, -12.0])          # z points down: 12 m above the grass
FOCAL = 800.0


def rotation(yaw_deg: float, down_deg: float = 25.0) -> np.ndarray:
    """World -> camera for a camera turned `yaw` around the vertical, looking `down` below the horizon."""
    forward = np.array([0.0, -1.0, 0.0])          # towards the far touchline
    yaw, down = np.radians(yaw_deg), np.radians(down_deg)
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    f = turn @ forward
    f = np.cos(down) * f + np.sin(down) * np.array([0, 0, 1.0])
    right = np.cross(np.array([0, 0, 1.0]), f)
    right /= np.linalg.norm(right)
    return np.vstack([right, np.cross(f, right), f])


K = np.array([[FOCAL, 0, SRC_W / 2], [0, FOCAL, SRC_H / 2], [0, 0, 1.0]])
R = {1: rotation(-55), 2: rotation(0), 3: rotation(55)}   # z points down: a positive turn is to the right
RAY_TO_CAM = {c: K @ R[c] for c in CAMS}
PITCH_TO_CAM = {c: RAY_TO_CAM[c] @ np.column_stack([np.eye(3)[:, :2], -CENTRE]) for c in CAMS}


def synthetic_panorama(focal: float = 150.0) -> Panorama:
    forward = np.array([-CENTRE[0], -CENTRE[1], 0.0])
    forward /= np.linalg.norm(forward)
    down = np.array([0, 0, 1.0])
    data = {"size": [int(np.pi * focal) + 2, int(1.3 * focal)], "focal_px": focal,
            "origin_px": [np.pi * focal / 2, 0.15 * focal], "centre_m": CENTRE.tolist(),
            "world_to_pano": np.vstack([np.cross(down, forward), down, forward]).tolist(), "feather_px": 40,
            "ray_to_cam": {str(c): RAY_TO_CAM[c].tolist() for c in CAMS},
            "pitch_to_cam": {str(c): PITCH_TO_CAM[c].tolist() for c in CAMS}}
    return Panorama(data)


def test_fit_rig_finds_the_shared_spot():
    rig = fit_rig(PITCH_TO_CAM)
    assert np.allclose(rig.centre, CENTRE, atol=0.05)
    assert all(abs(f - FOCAL) < 2 for f in rig.focal.values())
    assert rig.error_px < 0.5


def test_pitch_and_panorama_round_trip():
    pano = synthetic_panorama()
    pts = np.array([[0, 0], [-52.5, -34], [52.5, -34], [-40, 20], [30, 10], [0, 34]], float)
    back = pano.pano_to_pitch(pano.pitch_to_pano(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_ground_points_agree_with_3d_points_at_height_zero():
    pano = synthetic_panorama()
    pts = np.array([[10, -20], [-30, 5], [45, -30]], float)
    flat = pano.pitch_to_pano(pts)
    as_3d = pano.world_to_pano(np.column_stack([pts, np.zeros(len(pts))]))
    assert np.allclose(flat, as_3d, atol=1e-6)


def test_taller_things_are_higher_in_the_panorama():
    pano = synthetic_panorama()
    feet = pano.world_to_pano([[0, -20, 0]])[0]
    head = pano.world_to_pano([[0, -20, 1.8]])[0]
    assert abs(head[0] - feet[0]) < 1e-6 and head[1] < feet[1]          # straight up = same column, higher row


def test_camera_pixels_land_where_the_pitch_point_is():
    pano = synthetic_panorama()
    for c in CAMS:
        P = np.array([[-20.0 * (2 - c), -15.0]])
        cam_px = cv2.perspectiveTransform(P[None], PITCH_TO_CAM[c])[0]
        assert np.allclose(pano.cam_to_pano(c, cam_px), pano.pitch_to_pano(P), atol=1e-6)


def test_above_the_horizon_is_not_on_the_pitch():
    pano = synthetic_panorama()
    sky = np.array([[pano.u0, pano.v0 - 10]])      # 10 px above the horizon line
    assert np.isnan(pano.pano_to_pitch(sky)).all()


def test_blend_weights_share_out_to_one():
    pano = synthetic_panorama()
    edges = {1: np.array([100.0, 30, -np.inf]), 2: np.array([20.0, 40, 50]), 3: np.array([-np.inf, -np.inf, 10])}
    w = pano.blend_weights(edges)
    total = sum(w.values())
    assert np.allclose(total, 1)
    assert w[1][0] == pytest.approx(1.0)                 # far more central in cam1: cam1 alone
    assert 0 < w[1][1] < 1 and 0 < w[2][1] < 1          # close call: a blend


def test_stitch_fills_the_pitch_from_the_right_camera():
    pano = synthetic_panorama()
    st = Stitcher(pano)
    colours = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
    frames = [np.full((SRC_H, SRC_W, 3), colours[c], np.uint8) for c in CAMS]
    out = st.stitch(frames)
    for c, P in ((1, [-45, -25]), (2, [0, -10]), (3, [45, -25])):
        u, v = np.round(pano.pitch_to_pano(np.array([P], float))[0]).astype(int)
        assert tuple(out[v, u]) == colours[c]
    # every covered pixel is painted (weights sum to 1: no dark seams)
    assert out[st.coverage].astype(int).sum(axis=1).min() >= 250     # a blend of two colours still adds up


# ------------------------------------------------------------------ Step 2: camera models, robotic heads, cards
from studio.pitchmap import MapState  # noqa: E402
from studio.stadium import (Camera, Pinhole, PTZHead, Stadium, View, field_width_of, look_at_of,  # noqa: E402
                            view_towards)

MAIN = dict(name="MAIN", kind="ptz", built_from="panorama", position="rig", look_at=[0, 0], field_width_m=40,
            min_field_width_m=15, max_field_width_m=75, max_pan_deg_s=30, max_zoom_ratio_s=1.5, dead_zone_deg=1.0,
            smoothing_s=0.7)


def stadium_cfg() -> dict:
    return {"output_size": [1280, 720], "shot_types": {"wide_above_m": 50, "close_below_m": 25},
            "cameras": {"main": MAIN,
                        "goall": dict(name="GOAL L", kind="fixed", built_from="3d", position=[-60.5, 0, 6],
                                      look_at=[-36, 0], field_width_m=48)}}


def head(**over) -> PTZHead:
    c = {k: v for k, v in MAIN.items() if k not in ("position", "look_at")} | over
    return PTZHead(Camera(id="main", position_m=(-3, 42, 12), look_at_m=(0, 0), **c))


def test_view_round_trip():
    centre = np.array([-3.0, 42, -12])
    for x, y, fw in ((0, 0, 40), (-45, -20, 25), (40, 25, 60)):
        v = view_towards(centre, x, y, fw)
        assert np.allclose(look_at_of(centre, v), (x, y), atol=1e-6)
        assert field_width_of(centre, v) == pytest.approx(fw, rel=1e-6)


def test_pinhole_round_trip_and_centre():
    pin = Pinhole(View(pan=-30, tilt=12, hfov=25), 1280, 720)
    px = np.array([[0.5, 0.5], [640, 360], [1279.5, 719.5], [100, 600]], float)
    back, ok = pin.project(pin.rays(px))
    assert np.allclose(back, px, atol=1e-6) and ok.all()
    centre_ray = View(-30, 12, 25).forward()
    assert np.allclose(pin.project(centre_ray[None])[0], [[640, 360]], atol=1e-6)


def test_ptz_never_faster_than_its_motor():
    h = head()
    dt = 1 / 30
    h.aim(-50, -30)                                 # a long pan to the far left corner ...
    h.zoom(15)                                      # ... and a big zoom in at the same time
    prev = h.view
    for k in range(600):
        if k == 200:
            h.aim(50, -30)                          # then all the way to the other side
            h.zoom(75)
        v = h.step(dt)
        assert math.hypot(v.pan - prev.pan, v.tilt - prev.tilt) <= 30 * dt + 1e-9
        assert abs(math.log(v.hfov / prev.hfov)) <= math.log(1.5) * dt + 1e-9
        prev = v
    # it got there in the end: within the dead zone (as an angle: at this grazing angle 1 degree of tilt is ~12 m)
    target = np.array([50, -30, 0.0]) - h.centre
    angle = math.degrees(math.acos(h.view.forward() @ target / np.linalg.norm(target)))
    assert angle <= 1.0 * math.sqrt(2) + 1e-6


def test_ptz_dead_zone_ignores_tiny_moves():
    h = head()
    start = h.view
    h.aim(0.3, 0)                                   # ~0.4 degrees away: inside the 1 degree dead zone
    for _ in range(60):
        h.step(1 / 30)
    assert h.view.pan == start.pan and h.view.tilt == start.tilt


def test_ptz_respects_lens_limits():
    h = head()
    h.zoom(2)                                       # tighter than the lens can go
    assert h.target_width == 15
    h.zoom(500)
    assert h.target_width == 75


def test_cards_from_projection():
    pano = synthetic_panorama()
    st = Stadium(stadium_cfg(), pano)
    shots = st.shots()
    # MAIN looks at the centre spot, 40 m wide: the ball on the spot is in the middle of its picture
    pitch = MapState(t=1.0, ball=(0.0, 0.0), ball_status="seen", zone="middle",
                     players=[(1.0, 1.0), (-5.0, 3.0), (-48.0, 0.0), (-50.0, 33.0)])
    cards = {c.name: c for c in st.cards(1.0, shots, pitch, None)}
    main = cards["MAIN"]
    assert main.ball is not None and main.ball.status == "detected"
    assert main.ball.x == pytest.approx(0.5, abs=0.01) and main.ball.y == pytest.approx(0.5, abs=0.01)
    assert main.players == 2 and main.shot_type == "medium" and main.action_in_view
    # the behind-goal camera on the left looks down the pitch: it sees the centre spot (far away, high in its
    # picture) and 3 players, but not the one at the corner beside its goal
    goal = cards["GOAL L"]
    assert goal.ball is not None and goal.ball.y < 0.45 and goal.players == 3
    # a ball near the left goal: GOAL L sees it, MAIN (aimed at the centre) does not
    pitch.ball = (-45.0, 2.0)
    cards = {c.name: c for c in st.cards(1.2, shots, pitch, None)}
    assert cards["GOAL L"].ball is not None and cards["MAIN"].ball is None


def test_shot_type_follows_the_zoom():
    pano = synthetic_panorama()
    st = Stadium(stadium_cfg(), pano)
    pitch = MapState(t=0, ball=(0.0, 0.0), ball_status="seen")
    for width, kind in ((70, "wide"), (40, "medium"), (16, "close_up")):
        st.heads["main"].view = view_towards(st.heads["main"].centre, 0, 0, width)
        card = st.cards(0, st.shots(), pitch, None)[0]
        assert card.shot_type == kind


# ------------------------------------------------------------------ Step 5: the TV director (DirectorV2 mode "tv")
from studio.cards import BallInfo, CamCard, Quality  # noqa: E402
from studio.director import DirectorV2  # noqa: E402

TV = {"mode": "tv", "start_cam": 1, "min_shot_seconds": 2.0, "hysteresis_seconds": 0.6, "attack_max_s": 6, "attack_cooldown_s": 10,
      "roles": {"main": 1, "close": 2, "wide": 3, "boxl": 4, "boxr": 5, "goall": 6, "goalr": 7},
      "attack_min_intensity": 0.45, "attack_exit_s": 1.0, "stoppage_lost_s": 2.5, "close_min_s": 2.0,
      "close_max_s": 4.0, "restart_move_ms": 6, "restart_max_s": 8, "settle_deg_s": 12}
NAMES = ["MAIN", "CLOSE", "WIDE", "BOX L", "BOX R", "GOAL L", "GOAL R"]


def tv_cards(t: float, goal_sees: bool = False, moving: dict | None = None) -> list[CamCard]:
    cards = []
    for k, name in enumerate(NAMES):
        cam = k + 1
        sees = cam == 6 and goal_sees
        cards.append(CamCard(cam=cam, name=name, time=t, logged_at=t, shot_type="medium",
                             ball=BallInfo(0.5, 0.5, "detected") if sees else None, players=0, play_speed=0,
                             quality=Quality(), action_in_view=sees, moving_deg_s=(moving or {}).get(cam, 0.0)))
    return cards


def run_tv(states: dict[float, MapState], seconds: float, goal_sees: bool = True, moving=None, fps: int = 30):
    """Feed the director pitch-map states; like the real map, a new state comes every 0.2 s (the latest given one,
    re-dated, its events only once). Returns the cams on air and the cuts."""
    from dataclasses import replace

    d = DirectorV2(TV, fps, 7)
    d.start_at(0)
    times = sorted(states)
    on_air, cuts = [], []
    pitch = None
    for i in range(int(seconds * fps)):
        t = i / fps
        if i % 6 == 0:                                  # the map's news, 5 times a second
            key = max((s for s in times if s <= t), default=None)
            if key is not None:
                pitch = replace(states[key], t=t, events=states[key].events if t - key < 0.2 else [])
        dec = d.decide(i, t, tv_cards(t, goal_sees, moving), [], [], None, pitch=pitch)
        on_air.append(dec.cam)
        cuts += [e for e in dec.events if e["type"] == "CUT"]
    return on_air, cuts


def map_state(t, ball=(0.0, 0.0), zone="middle", intensity=0.2, status="seen", speed=8.0, events=None, age=0.0):
    return MapState(t=t, ball=ball, ball_status=status, zone=zone, intensity=intensity, speed_ms=speed,
                    events=events or [], ball_age_s=age)


def test_tv_main_by_default_and_goal_cam_in_an_attack():
    states = {0.0: map_state(0.0), 3.0: map_state(3.0, (-45, 2), "left_box", 0.7)}
    on_air, cuts = run_tv(states, 6)
    assert on_air[int(2.5 * 30)] == 1                   # MAIN in open play
    assert on_air[-1] == 6                              # GOAL L: its card sees the action
    on_air, _ = run_tv(states, 6, goal_sees=False)
    assert on_air[-1] == 4                              # ... else BOX L


def test_tv_close_after_ball_out_then_main():
    states = {0.0: map_state(0.0), 3.0: map_state(3.0, (10, 34.5), "out", events=[{"type": "BALL_OUT", "line": "touchline"}]),
              3.2: map_state(3.2, (10, 34.5), "out", speed=0.5)}
    on_air, cuts = run_tv(states, 10)
    assert on_air[int(4 * 30)] == 2                     # CLOSE after the stoppage
    assert on_air[-1] == 1                              # back on MAIN after at most 4 s
    close = sum(1 for c in on_air if c == 2) / 30
    assert 2.0 <= close <= 4.5


def test_tv_goal_kick_goes_wide():
    states = {0.0: map_state(0.0), 3.0: map_state(3.0, (-53, 10), "out", events=[{"type": "BALL_OUT", "line": "goal line"}]),
              3.2: map_state(3.2, (-48, 5), "left_box", speed=0.2, intensity=0.1)}
    on_air, _ = run_tv(states, 10)
    assert 3 in on_air[int(7.5 * 30):]                  # WIDE for the goal kick (after the close-up)


def test_tv_never_a_shot_under_two_seconds():
    rng = np.random.default_rng(3)
    states = {}
    for k in range(60):                                 # a nervous map: zones and events change every 0.5 s
        t = k * 0.5
        zone = rng.choice(["middle", "left_box", "right_box", "out"])
        ev = [{"type": "BALL_OUT", "line": "touchline"}] if zone == "out" else []
        states[t] = map_state(t, (-45, 0) if zone == "left_box" else (45, 0) if zone == "right_box" else (0, 0),
                              zone, 0.8, events=ev)
    on_air, cuts = run_tv(states, 30)
    assert cuts and min(c["shot_seconds"] for c in cuts) >= 2.0


def test_tv_waits_for_a_swinging_robotic_camera():
    states = {0.0: map_state(0.0), 3.0: map_state(3.0, (10, 34.5), "out", events=[{"type": "BALL_OUT", "line": "touchline"}])}
    on_air, _ = run_tv(states, 5, moving={2: 40.0})     # CLOSE is still turning fast
    assert 2 not in on_air


# ------------------------------------------------------------------ Step 3: the twin (ZXY + detections)
from studio.zxy import PitchTracker, apply_correction, ball_track, claim, to_centre  # noqa: E402


def test_zxy_corner_metres_to_centre_metres():
    corners = np.array([[0, 0], [105, 68], [52.5, 34]])
    assert np.allclose(to_centre(corners, False, False), [[-52.5, -34], [52.5, 34], [0, 0]])
    # Alfheim: the sensors' y runs the other way (measured): flip_y
    assert np.allclose(to_centre(corners, False, True), [[-52.5, 34], [52.5, -34], [0, 0]])
    assert np.allclose(apply_correction(corners, [1, 0, 0, 0, 1, 0]), corners)
    assert np.allclose(apply_correction([[1, 2]], [1, 0, -0.5, 0, 1, 0.5]), [[0.5, 2.5]])


def test_each_home_player_claims_only_one_detection():
    home = np.array([[0.0, 0.0]])
    people = [(0.5, 0.0, "player"), (1.5, 0.0, "player"), (20.0, 0.0, "referee")]   # a marker 1.5 m away
    rest = claim(home, people, radius=4.0)
    assert rest == [(1.5, 0.0, "player"), (20.0, 0.0, "referee")]       # the marker is NOT swallowed


def test_tracker_holds_a_lost_player_for_one_second():
    tr = PitchTracker(gate_m=3, hold_s=1.0, smooth=0.5)
    for k in range(5):
        tr.update(k * 0.2, [(k * 1.0, 0.0)])                          # running at 5 m/s
    assert len(tr.at(1.0, min_hits=2)) == 1
    tr.update(1.4, [])
    (_, x, _), = tr.at(1.4, min_hits=2)
    assert x > 4.0                                                     # predicted forward, not frozen
    tr.update(2.5, [])
    assert tr.at(2.5, min_hits=2) == []                                # gone after the hold time


def test_ball_gaps_are_filled_and_long_balls_fly():
    cfg = {"ball_gap_s": 2.0, "lofted_speed_ms": 15, "lofted_min_m": 20, "lofted_max_m": 15}
    ball = ball_track([(0.0, (0.0, 0.0)), (1.0, (10.0, 0.0)), (3.0, (50.0, 0.0)), (6.0, (55.0, 0.0))], cfg)
    assert ball(0.5) == (5.0, 0.0, 0.0)                               # short gap, slow: on the ground
    x, y, z = ball(2.0)
    assert x == pytest.approx(30.0) and z > 3                          # 40 m in 2 s: a lofted ball, high mid-way
    assert ball(4.5) is None                                           # a 3 s gap is not filled


# ------------------------------------------------------------------ Virtual Broadcast: replays, program summary
from studio.broadcast import add_replays, crowd_envelope, program_summary  # noqa: E402

IDS = ["main", "tight", "wide", "high18l", "high18r", "goall", "goalr"]
REPLAY = {"after_attack_s": 8, "before_s": 1.5, "after_s": 0.5, "max_source_s": 6, "speed": 0.6, "delay_s": 1.0,
          "gap_s": 20}


def program_with_attack(fps=30):
    """60 s: MAIN, an attack on the left goal (GOAL L on air) 10-16 s, a stoppage with CLOSE from 16 s, MAIN again."""
    n = 60 * fps
    states = ["PLAY"] * n
    on_air = [0] * n
    for k in range(10 * fps, 16 * fps):
        states[k], on_air[k] = "ATTACK", 5
    for k in range(16 * fps, 19 * fps):
        states[k], on_air[k] = "STOPPAGE", 1
    return on_air, states


def test_a_replay_follows_an_attack_that_ends_in_a_stoppage():
    on_air, states = program_with_attack()
    src, replays = add_replays(on_air, states, IDS, REPLAY, 30, 2.5)
    assert len(replays) == 1
    r = replays[0]
    assert IDS[r["cam"]] == "high18l"             # GOAL L showed it live: the replay comes from HIGH 18 L
    assert r["until"] >= 16 * 30 and r["until"] - r["from"] <= 6 * 30   # the end of the attack (the shot), <= 6 s
    assert src[r["at"] + 10] < r["at"]                     # replay frames show an earlier moment ...
    assert abs((src[r["at"] + 11] - src[r["at"] + 10]) - 0.6) < 1e-6   # ... in slow motion
    assert r["at"] >= 16 * 30 + 75                         # the CLOSE shot keeps its 2.5 s first


def test_no_shot_shorter_than_the_minimum_with_replays():
    on_air, states = program_with_attack()
    src, _ = add_replays(on_air, states, IDS, REPLAY, 30, 2.5)
    s = program_summary(on_air, src, IDS, 30, 2.5)
    assert s["shots_under_min"] == 0


def test_crowd_roars_after_an_attack_and_is_quiet_in_replays():
    on_air, states = program_with_attack()
    cfg = {"play": 0.25, "attack": 0.9, "stoppage": 0.35, "restart": 0.3, "replay": 0.25, "roar": 1.0, "roar_s": 3,
           "smooth_s": 1.2}
    env = crowd_envelope({"fps": 30, "states": states, "replays": []}, cfg)
    assert env[16 * 30 + 15] > env[5 * 30] + 0.4          # louder right after the attack than in open play
    assert env.max() <= 1.0
