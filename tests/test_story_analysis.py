import json
import os

import pytest

import analyze_content
import analyze_content_cloud
from test_analyze_validation import FakeResponse, make_analyzer, reply

# Four 10 s lines; the model answers with line numbers, times and text come from here.
LINES = ["Jadi gini ceritanya waktu itu gue lagi nunggu kereta pagi",
         "Terus temen gue bilang keretanya bakal telat satu jam lagi",
         "Gue ketawa aja kan padahal jadwalnya udah jelas di layar bro",
         "Ya udah habis itu kita pulang makan bakso di depan stasiun"]
TRANSCRIPT = "\n".join(f"{t}: {i * 10}.00s - {i * 10 + 10}.00s" for i, t in enumerate(LINES))


def clip(first, last, **scores):
    return dict({"start_line": first, "end_line": last, "topic": "funny", "title": "Kereta telat sejam",
                 "hook": 9, "flow": 9, "value": 9}, **scores)


def test_story_spans_lines_and_bad_line_numbers_are_dropped(monkeypatch):
    clips = [clip(0, 2), clip(2, 1), clip(3, 9), {"topic": "funny"}, "junk"]
    analyzer, _ = make_analyzer(monkeypatch, reply(json.dumps(clips)), reply('{"hook": 90, "coherence": 90, "connection": 90, "trend": 90}'))
    segs = analyzer.analyze(TRANSCRIPT, ["funny"])
    assert [(s["start_time"], s["end_time"]) for s in segs] == [("0.00s", "30.00s")]
    assert segs[0]["text"] == " ".join(LINES[:3])
    assert segs[0]["confidence"] == 0.9


def test_judge_failure_falls_back_to_own_scores_and_threshold_applies(monkeypatch):
    clips = [clip(0, 1), clip(2, 3, hook=3, flow=3, value=3)]
    analyzer, _ = make_analyzer(monkeypatch, reply(json.dumps(clips)), reply("no json"))
    segs = analyzer.analyze(TRANSCRIPT, ["funny"])
    assert [(s["start_time"], s["confidence"]) for s in segs] == [("0.00s", 0.9)]  # 0.3 is under 0.68
    assert (segs[0]["badge_label"], segs[0]["badge_emoji"]) == ("Kereta telat sejam", "🔥")  # badge call failed too


def test_cloud_analyzer_sends_key_and_reads_thinking(monkeypatch):
    sent = []

    def fake_post(url, json=None, headers=None, **kw):
        sent.append(headers)
        if len(sent) == 1:  # a thinking model that left the clip list in "thinking"
            return FakeResponse({"message": {"content": "", "thinking": json_clips}, "done_reason": "stop"})
        return FakeResponse({"error": "rate limited"})  # judge call fails: own scores are used

    json_clips = json.dumps([clip(0, 1)])
    monkeypatch.setattr(analyze_content_cloud.requests, "get", lambda *a, **kw: FakeResponse({"models": []}))
    monkeypatch.setattr(analyze_content_cloud.requests, "post", fake_post)
    analyzer = analyze_content_cloud.ContentAnalyzer(model_name="kimi-k2.5:cloud", api_key="test-key")
    segs = analyzer.analyze(TRANSCRIPT, ["funny"])
    assert sent[0] == {"Authorization": "Bearer test-key"}
    assert [(s["end_time"], s["confidence"]) for s in segs] == [("20.00s", 0.9)]


def test_long_badge_label_is_shortened_once_then_falls_back_to_title(monkeypatch):
    seg = {"title": "Kereta Telat", "text": LINES[1]}
    long_label = reply('{"badge_label": "Kereta telat sejam, kita pulang naik ojek aja", "badge_emoji": "🚆"}')
    analyzer, sent = make_analyzer(monkeypatch, long_label, reply('{"badge_label": "Kereta Telat Sejam"}'))
    assert analyzer.badge(dict(seg)) == {"badge_label": "Kereta Telat Sejam", "badge_emoji": "🚆"}
    assert "Too long for the badge (45 of 28 characters)" in sent[1]["messages"][-1]["content"]
    analyzer, _ = make_analyzer(monkeypatch, long_label)  # still too long: the cutting call's title
    assert analyzer.badge(dict(seg)) == {"badge_label": "Kereta Telat", "badge_emoji": "🚆"}


JUDGE90 = reply('{"hook": 90, "coherence": 90, "connection": 90, "trend": 90}')


def test_edge_check_moves_the_start_back_to_the_setup(monkeypatch):
    analyzer, sent = make_analyzer(monkeypatch, reply(json.dumps([clip(1, 2)])),
                                   reply('{"start_line": 0, "end_line": 2}'), JUDGE90)
    analyzer.edge_check = True
    segs = analyzer.analyze(TRANSCRIPT, ["funny"])
    assert (segs[0]["start_time"], segs[0]["end_time"]) == ("0.00s", "30.00s")
    assert "[1] > " in sent[1]["messages"][1]["content"]  # the clip's own lines are marked


def test_edge_check_keeps_the_clip_on_unusable_replies(monkeypatch):
    for bad in ('{"start_line": 2, "end_line": 1}', '{"start_line": 0, "end_line": 9}', "no json"):
        analyzer, _ = make_analyzer(monkeypatch, reply(json.dumps([clip(1, 2)])), reply(bad), JUDGE90)
        analyzer.edge_check = True
        segs = analyzer.analyze(TRANSCRIPT, ["funny"])
        assert (segs[0]["start_time"], segs[0]["end_time"]) == ("10.00s", "30.00s"), bad


def test_edge_check_is_off_by_default():
    assert analyze_content.ContentAnalyzer.edge_check is False


def _dirs(tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    (inp / "v_merged_0.txt").write_text("a: 0.00s - 9.00s\nb: 290.00s - 299.00s\n", encoding="utf-8")
    (inp / "v_merged_1.txt").write_text("c: 300.00s - 310.00s\n", encoding="utf-8")
    (out / "old_analysis.json").write_text("{}", encoding="utf-8")  # an earlier run's result
    return inp, out


def _fake_analyzer(seen):
    an = object.__new__(analyze_content.ContentAnalyzer)  # no Ollama needed
    an.analyze = lambda transcript, *a, **k: seen.append(transcript) or []
    return an


def test_windows_see_a_story_across_the_file_boundary_and_stale_results_go(tmp_path):
    inp, out = _dirs(tmp_path)
    seen = []
    _fake_analyzer(seen).analyze_directory(str(inp), str(out), ["funny"], window_overlap=60)
    assert sorted(os.listdir(out)) == ["window_0_analysis.json", "window_1_analysis.json"]
    assert "b: 290.00s - 299.00s" in seen[1] and "c: 300.00s - 310.00s" in seen[1]  # window 240-540 s


def test_no_overlap_keeps_one_result_per_file(tmp_path):
    inp, out = _dirs(tmp_path)
    _fake_analyzer([]).analyze_directory(str(inp), str(out), ["funny"])
    assert sorted(os.listdir(out)) == ["v_merged_0_analysis.json", "v_merged_1_analysis.json"]


def test_window_overlap_must_be_under_the_window(tmp_path):
    inp, _ = _dirs(tmp_path)
    with pytest.raises(ValueError):  # overlap >= window never advances: endless loop
        analyze_content.overlapping_windows(str(inp), ["v_merged_0.txt"], overlap=300)
