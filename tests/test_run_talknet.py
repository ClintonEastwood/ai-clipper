import os
import subprocess

import auto_reframe


def test_run_talknet_uses_arg_list_without_shell(tmp_path, monkeypatch):
    talknet_dir = tmp_path / "TalkNet-ASD"
    talknet_dir.mkdir()
    (talknet_dir / "pretrain_TalkSet.model").write_bytes(b"x")
    clip = tmp_path / 'my clip"; touch pwned; ".mp4'
    clip.write_bytes(b"fake video")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(auto_reframe.subprocess, "run", fake_run)
    pywork = auto_reframe.run_talknet(str(clip), str(talknet_dir))

    (cmd, kwargs), = calls
    assert isinstance(cmd, list)
    assert not kwargs.get("shell")
    assert kwargs["cwd"] == str(talknet_dir)
    name = cmd[cmd.index("--videoName") + 1]
    # TalkNet pastes the name into its own shell commands: keep it inert
    assert name == "my_clip___touch_pwned___"
    assert os.path.exists(talknet_dir / "demo_videos" / f"{name}.mp4")
    assert pywork == os.path.join(str(talknet_dir), "demo_videos", name, "pywork")


def test_run_talknet_reports_failure(tmp_path, monkeypatch):
    talknet_dir = tmp_path / "TalkNet-ASD"
    talknet_dir.mkdir()
    (talknet_dir / "pretrain_TalkSet.model").write_bytes(b"x")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video")
    monkeypatch.setattr(auto_reframe.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1))
    assert auto_reframe.run_talknet(str(clip), str(talknet_dir)) is None
