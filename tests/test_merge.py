from merge_rulebased import MergeConfig, SegmentMerger


def seg(start, end, topic="funny", confidence=0.8):
    return {"start_time": f"{start}s", "end_time": f"{end}s", "topic": topic, "confidence": confidence}


def test_nested_pick_does_not_cut_the_ending():
    # A 60 s story with a 10 s pick inside it: the merged clip must keep the story's ending.
    merged = SegmentMerger(MergeConfig())._merge_two_segments(seg(100, 160), seg(120, 130), "overlap")
    assert (merged["start_time"], merged["end_time"]) == (100, 160)


def merge(segs, strategy="story", **config):
    return [(s["start_time"], s["end_time"]) for s in SegmentMerger(MergeConfig(strategy=strategy, apply_context_expansion=False,
                                                                                **config)).merge_segments(segs)]


def test_story_clips_stay_as_cut():
    # Touching stories, and two sharing a boundary line, are separate clips: no gluing into 90-120 s.
    assert merge([seg(0, 40), seg(40, 80), seg(75, 120)]) == [(0, 40), (40, 80), (75, 120)]


def test_same_moment_is_posted_once():
    # The same moment found twice (overlap over half the shorter clip): one clip covering both.
    assert merge([seg(100, 140), seg(110, 150)]) == [(100, 150)]
    # Too long to join: the stronger clip wins.
    assert merge([seg(0, 60, confidence=0.7), seg(20, 100, confidence=0.9)], max_merged_duration=90) == [(20, 100)]


def test_smart_strategy_still_merges_close_picks():
    assert merge([seg(0, 10), seg(15, 25)], strategy="smart") == [(0, 25)]


def test_stretched_clips_are_never_under_the_minimum():
    # 29.88-59.88 came out as 29.999999999999996 s and the cutter (min 30) dropped the top clip;
    # a clip at the video start can't grow backwards, so it grows forwards.
    clips = SegmentMerger(MergeConfig(min_segment_duration=30, max_merged_duration=120))._apply_safety_bounds(
        [seg(34.88, 54.88), seg(0.0, 9.32), seg(100.0, 250.37)])
    assert [(c["start_time"], c["end_time"], c["duration"]) for c in clips] == [
        (29.88, 59.88, 30.0), (0.0, 30.0, 30.0), (115.19, 235.19, 120.0)]


def test_tier2_gap_flag_caps_every_tier2_merge():
    # Condition 1 had a fixed 30 s limit, so --tier2-gap 15 still merged a 20 s gap.
    picks = [seg(0, 10), seg(30, 40)]  # 20 s gap, same topic, confidence 0.8
    assert merge(picks, strategy="smart") == [(0, 40)]
    assert merge(picks, strategy="smart", tier2_max_gap=15) == [(0, 10), (30, 40)]
