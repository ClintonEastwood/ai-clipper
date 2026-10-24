import subprocess
import sys
from pathlib import Path

import pytest

import add_badge
import add_subtitles
import auto_reframe
import chunk_audio
import video_cutter


def test_reframe_exits_1_when_clips_fail(tmp_path, monkeypatch):
    # A broken cuDNN made TalkNet fail on all 34 clips, and the step still exited 0.
    (tmp_path / "a.mp4").write_bytes(b"")
    monkeypatch.setattr(auto_reframe, "setup_talknet", lambda d: None)
    monkeypatch.setattr(auto_reframe, "run_talknet", lambda clip, d: None)
    monkeypatch.setattr(sys, "argv", ["auto_reframe.py", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as e:
        auto_reframe.main()
    assert e.value.code == 1


@pytest.mark.parametrize("module", [auto_reframe, add_subtitles])
def test_no_input_clips_is_an_error(tmp_path, monkeypatch, module):
    monkeypatch.setattr(sys, "argv", ["x.py", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as e:
        module.main()
    assert e.value.code == 1


def test_reframe_leaves_no_broken_clip_when_ffmpeg_fails(tmp_path, monkeypatch):
    # FFmpeg wrote straight to the final path, so a failed or killed run left a broken .mp4
    # that every rerun skipped as "output already exists".
    (tmp_path / "a.mp4").write_bytes(b"")
    out = tmp_path / "out"
    monkeypatch.setattr(auto_reframe, "setup_talknet", lambda d: None)
    monkeypatch.setattr(auto_reframe, "run_talknet", lambda clip, d: str(tmp_path))
    monkeypatch.setattr(auto_reframe, "get_video_fps", lambda v: 25.0)
    monkeypatch.setattr(auto_reframe, "get_video_duration", lambda v: 10.0)
    monkeypatch.setattr(auto_reframe, "get_video_size", lambda v: (1920, 1080))
    monkeypatch.setattr(auto_reframe, "analyze_talknet_per_scene", lambda *a, **k: [
        {"start_time": 0.0, "end_time": 10.0, "mode": "pad", "target_coordinates": []}])

    def half_written(clip, output_video, *a):
        Path(output_video).write_bytes(b"half")
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(auto_reframe, "apply_ffmpeg_per_scene", half_written)
    monkeypatch.setattr(sys, "argv", ["auto_reframe.py", "--input-dir", str(tmp_path), "--output-dir", str(out)])
    with pytest.raises(SystemExit) as e:
        auto_reframe.main()
    assert e.value.code == 1
    assert sorted(p.name for p in out.iterdir()) == []


def test_cutter_exits_1_when_clips_fail(tmp_path, monkeypatch):
    # The cutter exited 0 with every clip failed, and run_pipeline marked the step done.
    monkeypatch.setattr(video_cutter, "validate_video_file", lambda p: True)
    monkeypatch.setattr(video_cutter, "validate_segments_file", lambda p: True)
    monkeypatch.setattr(video_cutter, "cut_all_clips", lambda **k: 2)
    monkeypatch.setattr(sys, "argv", ["video_cutter.py", "--input", "v.mp4", "--segments", "s.json", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        video_cutter.main()
    assert e.value.code == 1


def test_badge_exits_1_when_clips_fail(tmp_path, monkeypatch):
    (tmp_path / "a.mp4").write_bytes(b"")
    meta = tmp_path / "clips_metadata.json"
    meta.write_text("{}")
    monkeypatch.setattr(add_badge, "process_clip", lambda *a, **k: False)
    monkeypatch.setattr(sys, "argv", ["add_badge.py", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                                      "--metadata", str(meta)])
    with pytest.raises(SystemExit) as e:
        add_badge.main()
    assert e.value.code == 1


def test_chunker_raises_when_ffmpeg_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(chunk_audio, "get_audio_duration", lambda f: 50.0)
    monkeypatch.setattr(chunk_audio.subprocess, "run",
                        lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    with pytest.raises(RuntimeError, match="2/2 audio chunks failed"):
        chunk_audio.chunk_audio("a.wav", str(tmp_path), chunk_length=30)
