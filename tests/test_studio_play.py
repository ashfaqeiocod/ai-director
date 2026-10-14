"""studio.play: the game reconstruction (fixed figures, who has the ball, passes) on small made-up data."""
import numpy as np

from studio import play

CFG = {"run_max_ms": 8.5, "join_slack_m": 3, "sighting_sigma_m": 2.0, "sighting_floor": 0.02, "switch_cost": 5.0,
       "loose_r_m": 4.5, "loose_cost": 2.5, "loose_per_s": 1.5, "pass_speed": {"base_ms": 8, "per_m": 0.35, "max_ms": 24},
       "pass_time_s": [0.25, 3.0], "release_near_m": 2.0, "lofted_m": 28, "lofted_max_m": 12, "foot_m": 0.4,
       "touch_every_m": 3.0, "touch_push": {"per_ms": 0.12, "max_m": 1.0}, "pass_speed_tolerance": 1.3}
STEP = 0.1


def two_players(n: int = 100) -> np.ndarray:
    """A stands at (0, 0), B at (15, 0); the last 'player' is the loose ball (here: far away)."""
    pos = np.zeros((3, n, 2))
    pos[1, :, 0] = 15.0
    pos[2] = (40.0, 30.0)
    return pos


def test_stitch_joins_pieces_of_one_runner_and_drops_impossible_jumps():
    dots = [[] for _ in range(30)]
    for k in range(10):
        dots[k].append((1, k * 0.5, 0.0))            # piece 1: runs along x
    for k in range(15, 30):
        dots[k].append((2, 7.0 + (k - 15) * 0.5, 0.0))   # piece 2: the same runner after a gap
        dots[k].append((3, 0.0, 30.0))               # someone else, at the same time
    out = play.stitch(dots, 2, STEP, CFG, np.zeros((30, 2)))
    together = [s for s in range(2) if not np.isnan(out[s, 0, 0]) and not np.isnan(out[s, 20, 0])]
    assert together and out[together[0], 20, 1] == 0.0      # pieces 1 and 2 are one figure


def test_owner_changes_only_when_the_sightings_say_so():
    pos = two_players()
    sight = np.array([[k * STEP, 0.3, 0.2] for k in range(0, 40, 2)] + [[k * STEP, 15.2, -0.3] for k in range(60, 100, 2)])
    sight = np.vstack([sight, [[5.0, 15.0, 0.0]]])        # one stray sighting near B while A has it ...
    sight = sight[np.argsort(sight[:, 0])]
    own = play.owners(pos, sight, STEP, CFG)
    assert own[10] == 0 and own[90] == 1
    assert len(play.segments(own)) == 2                    # ... is not a pass (one change of owner only)


def test_ball_never_teleports_and_passes_take_real_time():
    pos = two_players()
    sight = np.array([[k * STEP, 0.3, 0.0] for k in range(0, 40, 2)] + [[k * STEP, 15.0, 0.0] for k in range(60, 100, 2)])
    own = play.owners(pos, sight, STEP, CFG)
    pas = play.clean_touches(own, pos, sight, STEP, CFG)
    assert len(pas) == 1 and pas[0]["from"] == 0 and pas[0]["to"] == 1
    head, speed = play.headings(pos, STEP, 1.0)
    ball = play.ball_path(own, pas, pos, head, speed, STEP, CFG)
    jumps = np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1) / STEP
    assert jumps.max() <= CFG["pass_speed"]["max_ms"] * CFG["pass_speed_tolerance"]
    assert np.linalg.norm(ball[5, :2] - pos[0, 5]) < 1.5 and np.linalg.norm(ball[95, :2] - pos[1, 95]) < 1.5


def test_limit_speed_turns_a_jump_into_a_run():
    track = np.zeros((40, 2))
    track[20:, 0] = 10.0
    out = play.limit_speed(track, 0.85)
    assert np.abs(np.diff(out[:, 0])).max() <= 0.85 + 1e-9
    assert out[0, 0] == 0.0 and out[-1, 0] == 10.0


def test_assistant_stays_on_his_line_in_his_half():
    defenders = np.zeros((11, 50, 2))
    defenders[:, :, 0] = np.linspace(10, 50, 11)[:, None]
    out = play.assistant(34.8, 1, defenders, np.zeros((50, 2)), 1, 0.85)
    assert np.allclose(out[:, 1], 34.8) and (out[:, 0] >= 0).all() and abs(out[25, 0] - 46.0) < 0.5
