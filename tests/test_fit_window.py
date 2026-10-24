import json

from fit_window import drop_openings, fit_clip, fit_long
from run_pipeline import build_parser, merge_argv

LINES = [(f"w{k}", 10.0 * k, 10.0 * k + 10) for k in range(10)]


def fake_chat(*replies):
    queue, calls = list(replies), []

    def chat(messages):
        calls.append([dict(m) for m in messages])
        return queue.pop(0)
    chat.calls = calls
    return chat


def pick(i, j):
    return json.dumps({"first_line": i, "last_line": j})


def test_fit_clip_picks_window():
    chat = fake_chat(pick(2, 6))
    assert fit_clip(LINES, 0, 90, [], chat) == (20.0, 70.0, "w2 w3 w4 w5 w6")
    assert len(chat.calls) == 1


def test_fit_clip_retries_with_feedback():
    chat = fake_chat(pick(0, 8), pick(1, 5))
    r = fit_clip(LINES, 0, 90, [], chat)
    assert r[:2] == (10.0, 60.0)
    assert len(chat.calls) == 2
    last = chat.calls[1][-1]
    assert last["role"] == "user" and "Lines 0-8 last 90 s" in last["content"] and "Pick again" in last["content"]


def test_fit_clip_unusable_replies():
    assert fit_clip(LINES, 0, 90, [], fake_chat("not json")) is None
    chat = fake_chat(pick(5, 2), pick(5, 2))
    assert fit_clip(LINES, 0, 90, [], chat) is None
    assert len(chat.calls) == 2
    assert "Lines must be 0 to 8" in chat.calls[1][-1]["content"]


def seg(start, end, conf, title="t"):
    return {"start_time": f"{start}s", "end_time": f"{end}s", "duration": end - start,
            "confidence": conf, "title": title, "text": "old"}


def test_fit_long():
    chat = fake_chat()
    short = seg(0, 45, 0.5)
    assert fit_long([short], LINES, [], chat, 30, 60) == [short]
    assert chat.calls == []

    out = fit_long([seg(0, 80, 0.7)], LINES, [], fake_chat(pick(1, 5)), 30, 60)[0]
    assert (out["start_time"], out["end_time"], out["duration"]) == (10.0, 60.0, 50.0)
    assert out["text"] == "w1 w2 w3 w4 w5" and out["fitted_from"] == [0.0, 80.0]
    assert out["confidence"] == 0.7 and out["title"] == "t"

    assert fit_long([seg(0, 80, 0.7)], LINES, [], fake_chat("junk"), 30, 60) == []



def test_fit_long_dedups_same_window():
    a, b = seg(0, 80, 0.6, "a"), seg(5, 85, 0.9, "b")
    out = fit_long([a, b], LINES, [], fake_chat(pick(1, 5), pick(1, 5)), 30, 60)
    assert [s["title"] for s in out] == ["b"]


def _paths():
    return {"analysis": "a", "merged": "m", "transcripts": "tr"}


def test_run_pipeline_max_duration_and_merge_argv():
    p = build_parser()
    assert p.parse_args([]).max_duration == 60
    argv = merge_argv(p.parse_args([]), _paths())
    assert float(argv[argv.index("--fit-to") + 1]) == 60
    assert "--transcripts" in argv and "--max-duration" not in argv
    argv = merge_argv(p.parse_args(["--strategy", "smart"]), _paths())
    assert "--max-duration" in argv and "--fit-to" not in argv
    # Cloud runs fit too (the even trim scored worst), with the model the analysis used
    from analyze_content_cloud import DEFAULT_MODEL
    argv = merge_argv(p.parse_args(["--cloud"]), _paths())
    assert "--fit-to" in argv and "--max-duration" not in argv
    assert argv[argv.index("--model") + 1] == DEFAULT_MODEL
    argv = merge_argv(p.parse_args(["--cloud", "--model", "kimi-k2.5:cloud"]), _paths())
    assert argv[argv.index("--model") + 1] == "kimi-k2.5:cloud"
    assert "--model" not in merge_argv(p.parse_args([]), _paths())  # local: merge's own default (gemma4)


def test_exact_limit_float_bounds():
    from audio_cues import snap_within
    assert snap_within(4.01, 64.01, [], 30, 60) == (4.01, 64.01)
    assert 64.01 - 4.01 > 60  # float noise
    lines = [("a", 4.01, 20.0), ("b", 20.0, 64.01)]
    r = fit_clip(lines, 4.01, 64.01, [], fake_chat(pick(0, 1)), 30, 60)
    assert r is not None and r[:2] == (4.01, 64.01)


def test_fit_long_keeps_exact_limit_clip():
    s = seg(4.01, 64.01, 0.5)
    chat = fake_chat()
    assert fit_long([s], LINES, [], chat, 30, 60) == [s] and chat.calls == []


def test_merge_fit_to_needs_transcripts(monkeypatch, tmp_path):
    import pytest
    import merge_rulebased
    monkeypatch.setattr("sys.argv", ["merge_rulebased.py", "--input", str(tmp_path), "--output", str(tmp_path),
                                     "--fit-to", "60"])
    with pytest.raises(SystemExit):
        merge_rulebased.main()


def test_merge_main_fits_long_story(monkeypatch, tmp_path):
    import merge_rulebased
    ana, tr, out = tmp_path / "ana", tmp_path / "tr", tmp_path / "out"
    ana.mkdir(), tr.mkdir()
    (ana / "a.json").write_text(json.dumps({"segments": [
        {"start_time": "0s", "end_time": "80s", "duration": 80, "confidence": 0.9, "title": "t", "text": "x"}]}))
    (tr / "a.txt").write_text("\n".join(f"w{k}: {10 * k:.2f}s - {10 * k + 10:.2f}s" for k in range(8)))
    monkeypatch.setattr("fit_window.ollama_chat", lambda model, url: fake_chat(pick(1, 5)))
    monkeypatch.setattr("sys.argv", ["merge_rulebased.py", "--input", str(ana), "--output", str(out),
                                     "--fit-to", "60", "--transcripts", str(tr), "--strategy", "story"])
    merge_rulebased.main()
    segs = json.loads((out / "merged_segments.json").read_text())["segments"]
    assert len(segs) == 1 and segs[0]["fitted_from"] == [0.0, 80.0] and segs[0]["duration"] <= 60


def test_warn_old_merge(tmp_path, capsys):
    from run_pipeline import warn_long_clips
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"segments": [{"duration": 80}, {"duration": 40}]}))
    warn_long_clips(f, 60)
    out = capsys.readouterr().out
    assert "1 clips" in out and "--force" in out


def test_warn_long_clips_never_raises(tmp_path):
    from run_pipeline import warn_long_clips
    for body in ("{truncated", "[1, 2]", "null", '{"segments": [{"duration": "x"}]}'):
        f = tmp_path / "m.json"
        f.write_text(body)
        warn_long_clips(f, 60)
    warn_long_clips(tmp_path / "missing.json", 60)


OPEN_LINES = [("hi", 30.0, 40.0), ("guest", 40.0, 60.0), ("story", 100.0, 120.0), ("more", 120.0, 140.0),
              ("later", 400.0, 440.0), ("end", 2700.0, 2767.0)]


def reply(opening, reason="r"):
    return json.dumps({"opening": opening, "reason": reason})


def test_drop_openings_drops_only_flagged_clips_in_the_first_5_min():
    chat = fake_chat(reply(True), reply(False))
    a, b, c = seg(29.88, 59.88, 0.9, "a"), seg(99.76, 141.09, 0.8, "b"), seg(400, 440, 0.7, "c")
    out = drop_openings([a, b, c], OPEN_LINES, chat)
    assert out == [b, c] and len(chat.calls) == 2
    prompt = chat.calls[0][0]["content"]
    assert "It starts 0:29 into an episode that is 46:07 long" in prompt and "hi\nguest" in prompt
    assert drop_openings([a], [], fake_chat()) == [a]


def test_drop_openings_keeps_clips_on_unusable_replies():
    a, b = seg(29.88, 59.88, 0.9, "a"), seg(99.76, 141.09, 0.8, "b")
    assert drop_openings([a, b], OPEN_LINES, fake_chat("nope", json.dumps({"opening": "yes"}))) == [a, b]
    assert drop_openings([a], OPEN_LINES, fake_chat("[1, 2]")) == [a]

    def boom(messages):
        raise OSError("down")
    assert drop_openings([a], OPEN_LINES, boom) == [a]


def test_merge_argv_drop_openings_for_story_only():
    p = build_parser()
    argv = merge_argv(p.parse_args([]), _paths())
    assert "--drop-openings" in argv and argv.count("--transcripts") == 1
    argv = merge_argv(p.parse_args(["--max-duration", "0"]), _paths())
    assert "--drop-openings" in argv and argv.count("--transcripts") == 1 and "--fit-to" not in argv
    assert "--drop-openings" not in merge_argv(p.parse_args(["--strategy", "smart"]), _paths())


def test_merge_drop_openings_needs_transcripts(monkeypatch, tmp_path, capsys):
    import pytest
    import merge_rulebased
    monkeypatch.setattr("sys.argv", ["merge_rulebased.py", "--input", str(tmp_path), "--output", str(tmp_path),
                                     "--drop-openings"])
    with pytest.raises(SystemExit):
        merge_rulebased.main()
    assert "--drop-openings needs --transcripts" in capsys.readouterr().err


def test_drop_openings_keeps_a_clip_with_no_lines_without_asking():
    empty = seg(200, 240, 0.5, "empty")  # first 5 min, but no transcript line inside it
    chat = fake_chat()
    assert drop_openings([empty], OPEN_LINES, chat) == [empty] and chat.calls == []
