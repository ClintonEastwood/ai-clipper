from rank_clips import rank, ranking_table

SEGS = [{"start_time": 0.0, "end_time": 30.0, "confidence": 0.70, "title": "A"},
        {"start_time": "50.00s", "end_time": "60.00s", "confidence": 0.68, "title": "B"}]
CUES = {"laugh": [0.0] * 50 + [0.9] * 10, "db": [-40.0] * 50 + [-10.0] * 10}


def test_rank_by_confidence_then_with_audio():
    assert [s["rank"] for s in rank([dict(s) for s in SEGS])] == [1, 2]
    ranked = rank([dict(s) for s in SEGS], CUES)  # B laughs and is loud: 0.744 vs 0.727
    assert [s["rank"] for s in ranked] == [2, 1]
    assert ranking_table(ranked).splitlines()[2].startswith("| 1 | 0.74 |")
