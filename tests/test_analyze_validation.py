import analyze_content
from analyze_content import validate_segment
from analyze_content_cloud import validate_segment as cloud_validate_segment
from validate_results import extract_quoted_terms

TEXT = "Pagi itu gue telat lagi ke kantor Nih gue bilang ke bos macet parah Eh dia ketawa kayak anak kecil gue bilang"
TRANSCRIPT = f"{TEXT}: 810.00s - 819.60s\n"


def test_apostrophes_are_not_quotes():
    # Two possessive apostrophes must not be read as quoting "s reaction and the office"
    seg = {"text": TEXT, "reason": "Discusses the boss's reaction and the office's running joke"}
    assert validate_segment(seg) == (True, [])
    assert cloud_validate_segment(seg) == (True, [])


def test_real_quotes_are_still_checked():
    assert validate_segment({"text": TEXT, "reason": "Calls him 'anak kecil' at the office"})[0]
    assert validate_segment({"text": TEXT, "reason": "The boss's friends said “anak kecil”"})[0]
    ok, issues = validate_segment({"text": TEXT, "reason": "The host's 'pemalas' joke"})
    assert not ok and "pemalas" in issues[0]


def test_validate_results_ignores_apostrophes():
    assert extract_quoted_terms("The boss's claim 'anak kecil' and the hosts' view") == ["anak kecil"]


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def make_analyzer(monkeypatch, *replies):
    """Fake Ollama: answers each /api/chat call with the next reply (the last one repeats)."""
    sent = []

    def fake_post(url, json=None, **kw):
        sent.append(json)
        return FakeResponse(replies[min(len(sent), len(replies)) - 1])

    monkeypatch.setattr(analyze_content.requests, "get", lambda *a, **kw: FakeResponse({}))
    monkeypatch.setattr(analyze_content.requests, "post", fake_post)
    return analyze_content.ContentAnalyzer(model_name="qwen3:14b"), sent


def reply(content, done="stop"):
    return {"message": {"content": content}, "done_reason": done}


STORY = reply('[{"start_line": 0, "end_line": 0, "topic": "controversial", "title": "Telat gara-gara macet", '
              '"hook": 9, "flow": 8, "value": 7}]')
JUDGE = reply('{"hook": 80, "coherence": 70, "connection": 60, "trend": 50}')
BADGE = reply('{"badge_label": "Telat Ngantor Kena Macet", "badge_emoji": "🚗"}')


def test_thinking_off_and_story_kept(monkeypatch):
    analyzer, sent = make_analyzer(monkeypatch, STORY, JUDGE, BADGE)
    segs = analyzer.analyze(TRANSCRIPT, ["controversial"])
    assert (segs[0]["badge_label"], segs[0]["badge_emoji"]) == ("Telat Ngantor Kena Macet", "🚗")
    assert all(call["think"] is False and call["options"]["num_ctx"] == 8192 for call in sent)
    assert len(segs) == 1 and segs[0]["text"] == TEXT
    assert (segs[0]["start_time"], segs[0]["end_time"]) == ("810.00s", "819.60s")
    assert segs[0]["confidence"] == 0.725  # (24/30 own + 260/400 judge) / 2


def test_cut_off_reply_is_reported(monkeypatch, capsys):
    analyzer, _ = make_analyzer(monkeypatch, reply('[{"start_line": 0, "end_li', done="length"))
    assert analyzer.analyze(TRANSCRIPT, ["controversial"]) == []
    assert "cut off at 4096 tokens" in capsys.readouterr().out
