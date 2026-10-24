import pytest

from run_pipeline import run_step, step_complete


def test_partial_outputs_are_not_a_finished_step(tmp_path):
    # An interrupted run leaves some clips behind; that must not skip the step
    (tmp_path / "clip_1.mp4").write_bytes(b"x")
    assert not step_complete(tmp_path, "*.mp4")

    run_step(5, 6, "", "Cutting", lambda: None, done_dir=tmp_path)
    assert step_complete(tmp_path, "*.mp4")


def test_failed_step_leaves_no_marker(tmp_path):
    def boom():
        raise ValueError("ffmpeg died")

    with pytest.raises(RuntimeError):
        run_step(5, 6, "", "Cutting", boom, done_dir=tmp_path)
    assert not step_complete(tmp_path, "*.mp4")
