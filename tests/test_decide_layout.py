import pickle

import numpy as np
import pytest

from talknet_speaker_detection import (
    MIN_MOTION,
    _decide_layout,
    analyze_talknet_per_scene,
    track_motion,
)


def stats(x, s, speaking_pct=0.0, motion=0.0, num_frames=100, y=400.0, **extra):
    spk = int(round(num_frames * speaking_pct / 100))
    d = {
        "track_id": int(x),
        "avg_x": float(x), "avg_y": float(y), "avg_s": float(s),
        "speaking_frames": spk,
        "global_speaking_frames": spk,
        "speaking_pct": speaking_pct,
        "num_frames": num_frames,
        "motion": motion,
    }
    d.update(extra)
    return d


SPEAKER = dict(x=500, s=90, speaking_pct=60.0, motion=0.05)


def test_silent_listener_who_moves_gives_split():
    # Listener never speaks in this shot (it spoke in another shot, which a
    # per-shot TalkNet track cannot know). It moves, so it is a person.
    listener = stats(1400, 85, speaking_pct=0.0, motion=0.04)
    mode, coords = _decide_layout([stats(**SPEAKER), listener], 1920, 1080, 10)
    assert mode == "split"
    assert [c["x"] for c in coords] == [500, 1400]


def test_static_prop_is_ignored():
    poster = stats(1600, 80, speaking_pct=0.0, motion=0.0)
    mode, coords = _decide_layout([stats(**SPEAKER), poster], 1920, 1080, 10)
    assert mode == "single"
    assert coords[0]["x"] == 500


def test_listener_wins_over_static_prop_on_same_side():
    # The prop has more frames, but the moving face is the listener
    poster = stats(1700, 80, motion=0.0, num_frames=150)
    listener = stats(1300, 85, motion=0.03, num_frames=100)
    mode, coords = _decide_layout([stats(**SPEAKER), poster, listener], 1920, 1080, 10)
    assert mode == "split"
    assert coords[1]["x"] == 1300


def test_motion_threshold_boundary():
    below = stats(1400, 85, motion=MIN_MOTION * 0.5)
    above = stats(1400, 85, motion=MIN_MOTION * 2)
    assert _decide_layout([stats(**SPEAKER), below], 1920, 1080, 10)[0] == "single"
    assert _decide_layout([stats(**SPEAKER), above], 1920, 1080, 10)[0] == "split"


def test_small_moving_face_still_filtered():
    # A background face far smaller than the speaker is not a split partner
    tiny = stats(1400, 30, motion=0.08)
    assert _decide_layout([stats(**SPEAKER), tiny], 1920, 1080, 10)[0] == "single"


def test_both_speaking_gives_split():
    other = stats(1400, 88, speaking_pct=40.0, motion=0.0)
    assert _decide_layout([stats(**SPEAKER), other], 1920, 1080, 10)[0] == "split"


def test_missing_motion_is_treated_as_static():
    listener = stats(1400, 85)
    del listener["motion"]
    assert _decide_layout([stats(**SPEAKER), listener], 1920, 1080, 10)[0] == "single"


def test_480p_listener():
    # Same shot at 854x480: sides split at width/2, motion is scale-free
    k = 480 / 1080
    speaker = stats(500 * k, 90 * k, speaking_pct=60.0, motion=0.05)
    listener = stats(1400 * k, 85 * k, motion=0.04)
    mode, _ = _decide_layout([speaker, listener], 854, 480, 10)
    assert mode == "split"


def test_single_and_empty():
    assert _decide_layout([stats(**SPEAKER)], 1920, 1080, 10)[0] == "single"
    assert _decide_layout([], 1920, 1080, 10) == ("pad", [])


# --- track_motion -----------------------------------------------------------

def wobble(n, center, amp, phase=0.0):
    return center + amp * np.sin(np.linspace(0, 6 * np.pi, n) + phase)


def test_track_motion_static_is_zero():
    n = 100
    assert track_motion(np.full(n, 1600.0), np.full(n, 300.0), np.full(n, 80.0)) == 0.0


def test_track_motion_is_resolution_independent():
    n = 100
    hd = track_motion(wobble(n, 1400, 4), wobble(n, 420, 3, 1.0), wobble(n, 85, 1))
    k = 480 / 1080
    sd = track_motion(wobble(n, 1400 * k, 4 * k), wobble(n, 420 * k, 3 * k, 1.0), wobble(n, 85 * k, 1 * k))
    assert hd > MIN_MOTION
    assert sd == pytest.approx(hd, rel=1e-6)


def test_track_motion_without_sizes():
    assert track_motion([1, 2, 3], [1, 2, 3], []) == 0.0
    assert track_motion([1, 2, 3], [1, 2, 3], [0, 0, 0]) == 0.0


# --- analyze_talknet_per_scene on synthetic TalkNet output -----------------

def make_track(frames, x, y, s, amp):
    """TalkNet-style track: 'track' (frames, bbox) + 'proc_track' (x, y, s)."""
    n = len(frames)
    xs, ys, ss = wobble(n, x, amp), wobble(n, y, amp * 0.7, 1.0), wobble(n, s, amp * 0.2, 2.0)
    bbox = np.stack([xs - ss, ys - ss, xs + ss, ys + ss], axis=1)
    return {"track": {"frame": np.array(frames), "bbox": bbox},
            "proc_track": {"x": xs, "y": ys, "s": ss}}


def write_pywork(tmp_path, tracks, scores):
    tracks_path, scores_path = tmp_path / "tracks.pckl", tmp_path / "scores.pckl"
    tracks_path.write_bytes(pickle.dumps(tracks))
    scores_path.write_bytes(pickle.dumps(scores))
    return str(tracks_path), str(scores_path)


def speaking(n, pct):
    """Per-frame TalkNet scores: positive = speaking."""
    out = np.full(n, -1.5)
    out[: int(n * pct / 100)] = 1.2
    return out


@pytest.mark.parametrize("width,height", [(1920, 1080), (854, 480)])
def test_per_scene_two_shots_with_alternating_speakers(tmp_path, width, height):
    # Shot 1 (frames 0-99): A speaks, B listens. Shot 2 (100-199): B speaks,
    # A listens. A static poster sits in both shots. Before the fix, each
    # listener had 0 speaking history on its own per-shot track, so both shots
    # became single; on 480p the listener's pixel variance was also under
    # the old 5 px^2 prop floor.
    k = height / 1080
    shot1, shot2 = list(range(0, 100)), list(range(100, 200))
    tracks = [
        make_track(shot1, 500 * k, 400 * k, 90 * k, 4 * k),   # A speaking
        make_track(shot1, 1400 * k, 420 * k, 85 * k, 3 * k),  # B listening
        make_track(shot1, 1750 * k, 250 * k, 70 * k, 0),      # poster
        make_track(shot2, 1400 * k, 420 * k, 85 * k, 4 * k),  # B speaking
        make_track(shot2, 500 * k, 400 * k, 90 * k, 3 * k),   # A listening
        make_track(shot2, 1750 * k, 250 * k, 70 * k, 0),      # poster
    ]
    scores = [speaking(100, 70), speaking(100, 0), speaking(100, 0),
              speaking(100, 65), speaking(100, 0), speaking(100, 0)]
    tracks_path, scores_path = write_pywork(tmp_path, tracks, scores)

    layouts = analyze_talknet_per_scene(tracks_path, scores_path, fps=25.0,
                                        video_width=width, video_height=height,
                                        video_duration=8.0)
    assert [l["mode"] for l in layouts] == ["split", "split"]
    for layout in layouts:
        xs = sorted(c["x"] for c in layout["target_coordinates"])
        # Speaker + listener, never the poster at x=1750
        assert xs == [pytest.approx(500 * k, abs=10 * k), pytest.approx(1400 * k, abs=10 * k)]


def test_per_scene_speaker_with_only_a_poster_is_single(tmp_path):
    frames = list(range(0, 100))
    tracks = [make_track(frames, 500, 400, 90, 4), make_track(frames, 1500, 350, 85, 0)]
    scores = [speaking(100, 70), speaking(100, 0)]
    tracks_path, scores_path = write_pywork(tmp_path, tracks, scores)
    layouts = analyze_talknet_per_scene(tracks_path, scores_path, fps=25.0,
                                        video_width=1920, video_height=1080,
                                        video_duration=4.0)
    assert [l["mode"] for l in layouts] == ["single"]
    assert layouts[0]["target_coordinates"][0]["x"] == pytest.approx(500, abs=10)


def test_threshold_between_measured_props_and_listeners():
    # GPU run 2026-10-06: props 0.001-0.003, listeners 0.089-0.224 (1080p50 + 480p clips)
    assert 0.003 < MIN_MOTION < 0.089
