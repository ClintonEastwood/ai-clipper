from audio_cues import parse_silencedetect, snap_all, snap_to_pauses

LOG = ("[silencedetect @ 0x1] silence_start: 98.5\n"
       "[silencedetect @ 0x1] silence_end: 99.9 | silence_duration: 1.4\n"
       "[silencedetect @ 0x1] silence_start: 161.2\n"
       "[silencedetect @ 0x1] silence_end: 161.8 | silence_duration: 0.6\n"
       "[silencedetect @ 0x1] silence_start: 300.1\n")  # file ends silent: no silence_end
PAUSES = [(98.5, 99.9), (161.2, 161.8)]


def test_parse_silencedetect_drops_unclosed_pause_and_empty_log():
    assert parse_silencedetect(LOG) == PAUSES
    assert parse_silencedetect("") == []  # no audio stream: nothing to snap to


def test_snap_moves_edges_out_into_the_nearest_pause():
    assert snap_to_pauses(100.0, 160.0, PAUSES) == (99.75, 161.35)


def test_snap_keeps_edges_in_a_pause_or_with_no_pause_near():
    assert snap_to_pauses(99.0, 161.5, PAUSES) == (99.0, 161.5)
    assert snap_to_pauses(110.0, 150.0, PAUSES) == (110.0, 150.0)


def test_snap_all_uses_the_cache_and_never_exceeds_max_len(tmp_path):
    (tmp_path / "pauses.json").write_text("[[98.5, 99.9], [161.2, 161.8]]", encoding="utf-8")
    ok, too_long = {"start_time": 100.0, "end_time": 160.0}, {"start_time": 100.0, "end_time": 160.0}
    snap_all([ok], "unused.mp4", str(tmp_path), max_len=120)
    snap_all([too_long], "unused.mp4", str(tmp_path), max_len=60)
    assert (ok["start_time"], ok["end_time"], ok["duration"]) == (99.75, 161.35, 61.6)
    assert (too_long["start_time"], too_long["end_time"]) == (100.0, 160.0)  # 61.6 s > 60: the cutter would drop it


import numpy as np

from audio_cues import clip_audio_score, laughter_and_loudness


def test_laughter_and_loudness_per_second():
    sr = 100  # a tiny sample rate keeps the test fast; real runs use 16 kHz
    samples = np.zeros(sr * 20, dtype=np.float32)
    samples[sr * 10:] = 0.5  # loud second half
    cues = laughter_and_loudness(samples, sr=sr, classify=lambda x: float(x.mean() > 0.25))
    assert len(cues["laugh"]) == len(cues["db"]) == 20
    assert cues["db"][15] > cues["db"][5] + 40
    assert (cues["laugh"][15], cues["laugh"][0]) == (1.0, 0.0)


def test_clip_audio_score_prefers_the_loud_laughing_clip():
    cues = {"laugh": [0.0] * 50 + [0.9] * 10, "db": [-40.0] * 50 + [-10.0] * 10}
    assert clip_audio_score(50, 60, cues) > clip_audio_score(0, 30, cues)


def test_clip_audio_score_past_the_end_and_with_no_audio():
    cues = {"laugh": [0.0] * 50 + [0.9] * 10, "db": [-40.0] * 50 + [-10.0] * 10}
    assert 0.0 <= clip_audio_score(55, 90, cues) <= 1.0  # the episode's last clip
    assert clip_audio_score(0, 30, {"laugh": [], "db": []}) == 0.0


import subprocess

from audio_cues import cached, decode_audio


def test_no_audio_stream_and_silence_give_no_cues_and_score_0(tmp_path):
    video = tmp_path / "mute.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=1:s=32x32", "-an", str(video)], check=True)
    assert len(decode_audio(str(video))) == 0
    assert laughter_and_loudness(np.zeros(50, dtype=np.float32), sr=100, classify=lambda x: 1 / 0) == {"laugh": [], "db": []}
    assert clip_audio_score(0, 10, {"laugh": [0.0] * 30, "db": [-180.0] * 30}) == 0.0  # silent video


def test_cache_is_redone_for_another_video(tmp_path):
    a, b, out = tmp_path / "a.mp4", tmp_path / "b.mp4", str(tmp_path / "cues.json")
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbbbbb")
    assert cached(out, lambda: "A", str(a)) == "A"
    assert cached(out, lambda: "again", str(a)) == "A"
    assert cached(out, lambda: "B", str(b)) == "B"  # same output dir, new video


def test_unreadable_video_raises_and_caches_nothing(tmp_path):
    import pytest
    from audio_cues import detect_pauses
    out = tmp_path / "pauses.json"
    with pytest.raises(RuntimeError, match="missing.mp4"):
        cached(str(out), lambda: detect_pauses(str(tmp_path / "missing.mp4")))
    assert not out.exists()


def test_snap_within_keeps_length_in_range():
    from audio_cues import snap_within
    P = [(9.0, 9.5), (11.0, 11.6), (68.4, 69.0), (70.5, 71.0)]
    assert snap_within(10.0, 70.0, P, 30, 60) == (9.35, 68.55)
    assert snap_within(10.0, 60.0, P, 30, 60) == (9.35, 60.0)
    assert snap_within(9.2, 69.0, P, 30, 60) == (9.2, 69.0)
    Q = [(7.2, 7.4), (11.0, 11.6), (69.0, 69.5), (70.5, 71.0)]
    assert snap_within(10.0, 70.4, Q, 30, 60) == (11.45, 70.65)
    assert snap_within(0.0, 100.0, [], 30, 60) == (0.0, 100.0)
