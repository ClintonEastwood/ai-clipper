import os
import subprocess
import sys

import pytest

import setup_talknet
from setup_talknet import (
    TALKNET_COMMIT,
    WEIGHTS,
    clone_talknet,
    download_weights,
    patch_numpy_aliases,
)


# --- NumPy alias patch ------------------------------------------------------

def test_patch_numpy_aliases(tmp_path):
    src = tmp_path / "box_utils.py"
    src.write_text(
        "import numpy as np\n"
        "a = np.array(keep).astype(np.int)\n"
        "b = x.dtype != np.float or y.dtype != numpy.float\n"
        "c = np.zeros(3, dtype=np.bool)\n"
        "d = np.NAN, np.NaN\n"
        "keep = np.int32, np.int64, np.float32, np.bool_, np.int_, np.nan, np.integer\n"
    )
    patched = patch_numpy_aliases(tmp_path)
    assert patched == [src]
    assert src.read_text() == (
        "import numpy as np\n"
        "a = np.array(keep).astype(int)\n"
        "b = x.dtype != float or y.dtype != float\n"
        "c = np.zeros(3, dtype=bool)\n"
        "d = np.nan, np.nan\n"
        "keep = np.int32, np.int64, np.float32, np.bool_, np.int_, np.nan, np.integer\n"
    )
    # Idempotent: second run changes nothing
    assert patch_numpy_aliases(tmp_path) == []


def test_patch_keeps_crlf_line_endings(tmp_path):
    src = tmp_path / "eval.py"
    src.write_bytes(b"import numpy as np\r\nx = np.float\r\n")
    patch_numpy_aliases(tmp_path)
    assert src.read_bytes() == b"import numpy as np\r\nx = float\r\n"


# --- Weight download --------------------------------------------------------

@pytest.fixture
def gdown_installed(monkeypatch):
    real_find_spec = setup_talknet.importlib.util.find_spec
    monkeypatch.setattr(setup_talknet.importlib.util, "find_spec",
                        lambda name: object() if name == "gdown" else real_find_spec(name))


def test_download_weights_uses_python_m_gdown(tmp_path, monkeypatch, gdown_installed):
    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(cmd)
        out = cmd[cmd.index("-O") + 1]
        with open(out, "wb") as f:
            f.write(b"weights")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(setup_talknet.subprocess, "run", fake_run)
    download_weights(tmp_path)

    assert len(calls) == len(WEIGHTS)
    for cmd, (file_id, rel_path) in zip(calls, WEIGHTS):
        # Not `gdown --id`, which gdown 5.x removed
        assert cmd == [sys.executable, "-m", "gdown", file_id, "-O", str(tmp_path / rel_path) + ".part"]
        assert (tmp_path / rel_path).read_bytes() == b"weights"
        assert not (tmp_path / (rel_path + ".part")).exists()

    # Idempotent: files exist, nothing is downloaded again
    calls.clear()
    download_weights(tmp_path)
    assert calls == []


def test_failed_download_leaves_no_weight_file(tmp_path, monkeypatch, gdown_installed):
    def failing_run(cmd, check=False, **kwargs):
        open(cmd[cmd.index("-O") + 1], "wb").close()  # partial/empty file
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(setup_talknet.subprocess, "run", failing_run)
    with pytest.raises(subprocess.CalledProcessError):
        download_weights(tmp_path)
    # Next run must retry, so no (empty) weight file may be left behind
    assert not (tmp_path / WEIGHTS[0][1]).exists()


def test_download_without_gdown_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_talknet.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="pip install gdown"):
        download_weights(tmp_path)


# --- Clone ------------------------------------------------------------------

def test_clone_skipped_when_present(tmp_path, monkeypatch):
    (tmp_path / "demoTalkNet.py").write_text("")
    monkeypatch.setattr(setup_talknet.subprocess, "run",
                        lambda *a, **k: pytest.fail("should not run git"))
    clone_talknet(tmp_path)


def test_clone_refuses_unknown_non_empty_dir(tmp_path):
    (tmp_path / "something.txt").write_text("")
    with pytest.raises(RuntimeError, match="no demoTalkNet.py"):
        clone_talknet(tmp_path)


def test_clone_pins_commit_and_renames_into_place(tmp_path, monkeypatch):
    target = tmp_path / "temp_work" / "TalkNet-ASD"
    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append(cmd)
        if cmd[1] == "clone":
            dest = cmd[-1]
            os.makedirs(dest)
            open(os.path.join(dest, "demoTalkNet.py"), "w").close()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(setup_talknet.subprocess, "run", fake_run)
    clone_talknet(target)

    partial = str(target) + ".partial"
    assert calls[0][:2] == ["git", "clone"] and calls[0][-1] == partial
    assert calls[1] == ["git", "-C", partial, "checkout", "--quiet", TALKNET_COMMIT]
    assert (target / "demoTalkNet.py").exists()
    assert not (tmp_path / "temp_work" / "TalkNet-ASD.partial").exists()
