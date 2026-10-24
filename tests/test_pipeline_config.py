from pathlib import Path

import pytest

import run_pipeline
from run_pipeline import apply_config, build_parser, build_paths, print_dry_run


def test_yaml_applies_when_flag_not_given():
    # These three were silently ignored when the hand-kept defaults drifted from the parser's
    args = build_parser().parse_args([])
    apply_config(args, {"min_confidence": 0.7, "min_duration": 45, "max_duration": 90})
    assert (args.min_confidence, args.min_duration, args.max_duration) == (0.7, 45, 90)


def test_cli_flag_beats_yaml():
    args = build_parser().parse_args(["--min-confidence", "0.6"])
    apply_config(args, {"min_confidence": 0.7})
    assert args.min_confidence == 0.6


def test_nested_config_is_an_error(capsys):
    # The old config.yaml layout put the whole model: section into args.model and dropped topics/input.
    args = build_parser().parse_args([])
    with pytest.raises(SystemExit) as e:
        apply_config(args, {"model": {"name": "gemma4"}, "analysis": {"topics": ["funny"]}})
    assert e.value.code == 1
    assert "model, analysis" in capsys.readouterr().out


def test_empty_config_file_is_fine():
    args = build_parser().parse_args([])
    apply_config(args, None)
    assert args.topics == build_parser().parse_args([]).topics


def test_dry_run_hides_api_key(capsys):
    args = build_parser().parse_args(["--input", "v.mp4", "--topics", "funny", "--cloud", "--api-key", "sk-secret"])
    print_dry_run(args, build_paths(args))
    out = capsys.readouterr().out
    assert "sk-secret" not in out and "--api-key ***" in out


def parse(*flags):
    return build_parser().parse_args(["--input", "v.mp4", "--topics", "funny", "--output-dir", "o", *flags])


def test_default_is_reframe_then_captions_then_badge():
    # One command, video in, captioned + badged vertical clips out: TalkNet needs the landscape cut
    p = build_paths(parse())
    assert Path(p["cut"]).name == "clips"
    assert Path(p["clips_metadata"]).parent.name == "clips"
    assert Path(p["badge_input"]).name == "clips_subtitled"


def test_pad_mode_cuts_vertical_and_still_captions():
    p = build_paths(parse("--vertical-mode", "pad"))
    assert Path(p["cut"]).name == "clips_vertical"
    assert Path(p["badge_input"]).name == "clips_subtitled"


def test_skip_subtitles_badges_the_vertical_clips():
    assert Path(build_paths(parse("--skip-subtitles"))["badge_input"]).name == "clips_vertical"


def test_dry_run_lists_all_8_steps(capsys):
    args = parse()
    print_dry_run(args, build_paths(args))
    out = capsys.readouterr().out
    assert "auto_reframe.py" in out and "add_subtitles.py" in out and "[8/8] python" in out
    assert "--vertical-mode" not in out  # the cut is landscape; TalkNet makes it vertical

    args = parse("--vertical-mode", "pad")
    print_dry_run(args, build_paths(args))
    out = capsys.readouterr().out
    assert "[6/8] (skipped)" in out and "--vertical-mode pad" in out and "add_subtitles.py" in out


def test_free_gpu_never_stops_the_run(monkeypatch):
    import requests

    def down(*a, **k):
        raise requests.ConnectionError("ollama not running")

    monkeypatch.setattr(requests, "get", down)
    run_pipeline.free_gpu()


def test_free_gpu_unloads_only_models_in_vram(monkeypatch, capsys):
    import requests

    class R:
        def json(self):
            return {"models": [{"name": "gemma4:12b-it-q4_K_M", "size_vram": 8_000_000_000},
                               {"name": "kimi-k2.5:cloud", "size_vram": 0}]}

    posted = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(requests, "post", lambda url, json, **k: posted.append(json))
    run_pipeline.free_gpu()
    assert posted == [{"model": "gemma4:12b-it-q4_K_M", "keep_alive": 0}]
