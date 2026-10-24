#!/usr/bin/env python3
"""
setup_talknet.py - one-time TalkNet-ASD setup for auto_reframe.py

Safe to run again: each step skips what is already done.
  1. Clone TalkNet-ASD into temp_work/TalkNet-ASD (pinned commit)
  2. Download the TalkNet and S3FD face-detector weights with
     `python -m gdown <id> -O <path>`. TalkNet's own code calls `gdown --id`,
     which gdown 5.x removed; with the weights in place it never runs.
  3. Patch NumPy aliases removed in NumPy 1.24/2.0 (np.int, np.float, np.bool,
     np.NAN, ...)

auto_reframe.py runs this automatically before TalkNet.

Usage:
    python scripts/setup_talknet.py
    python scripts/setup_talknet.py --talknet-dir /path/to/TalkNet-ASD
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TALKNET_DIR = PROJECT_ROOT / "temp_work" / "TalkNet-ASD"

TALKNET_REPO = "https://github.com/TaoRuijie/TalkNet-ASD"
# Latest upstream commit; the alias patch below was checked against it
TALKNET_COMMIT = "6d6821479af485e251c4991487e40573b42181b4"

# (Google Drive file id, path inside the TalkNet dir) - the ids TalkNet uses.
# demoTalkNet.py runs from the TalkNet dir and loads both from these paths.
WEIGHTS = [
    ("1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea", "pretrain_TalkSet.model"),
    ("1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt", "model/faceDetector/s3fd/sfd_face.pth"),
]

# Aliases removed in NumPy 1.24/2.0. \b stops the match before np.int32,
# np.float64, np.bool_, ...
NUMPY_ALIASES = [
    (re.compile(rf"\b(?:np|numpy)\.{name}\b"), name)
    for name in ("int", "float", "bool", "complex", "object", "str")
] + [(re.compile(r"\b(np|numpy)\.(?:NaN|NAN)\b"), r"\1.nan")]


def clone_talknet(talknet_dir):
    """Clone TalkNet-ASD at TALKNET_COMMIT unless it is already there."""
    talknet_dir = Path(talknet_dir)
    if (talknet_dir / "demoTalkNet.py").is_file():
        print(f"✓ TalkNet already cloned: {talknet_dir}")
        return
    if talknet_dir.exists() and any(talknet_dir.iterdir()):
        raise RuntimeError(
            f"{talknet_dir} exists but has no demoTalkNet.py. Delete it and run again."
        )

    # Clone next to the target, then rename, so an interrupted clone never
    # leaves a half-filled TalkNet dir that later runs would treat as done.
    partial = talknet_dir.with_name(talknet_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cloning {TALKNET_REPO} into {talknet_dir} ...")
    subprocess.run(["git", "clone", "--quiet", TALKNET_REPO, str(partial)], check=True)
    subprocess.run(["git", "-C", str(partial), "checkout", "--quiet", TALKNET_COMMIT], check=True)

    if talknet_dir.exists():
        talknet_dir.rmdir()  # empty dir left by an earlier run
    partial.rename(talknet_dir)
    print("✓ TalkNet cloned")


def download_weights(talknet_dir):
    """Download missing weight files with `python -m gdown <id> -O <path>`."""
    for file_id, rel_path in WEIGHTS:
        dest = Path(talknet_dir) / rel_path
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"✓ {rel_path} already downloaded")
            continue

        if importlib.util.find_spec("gdown") is None:
            raise RuntimeError("gdown is not installed. Run: pip install gdown")

        # Download to .part and rename, so a failed download is retried next run
        part = dest.with_name(dest.name + ".part")
        part.unlink(missing_ok=True)
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {rel_path} ...")
        subprocess.run(
            [sys.executable, "-m", "gdown", file_id, "-O", str(part)], check=True
        )
        if not part.is_file() or part.stat().st_size == 0:
            raise RuntimeError(f"Download of {rel_path} produced no file")
        os.replace(part, dest)
        print(f"✓ {rel_path} downloaded")


def patch_numpy_aliases(talknet_dir):
    """
    Replace NumPy aliases removed in NumPy 1.24/2.0 (np.int -> int, ...) in
    TalkNet's .py files. Returns the list of files changed.
    """
    patched = []
    for py_file in sorted(Path(talknet_dir).rglob("*.py")):
        try:
            # newline="" keeps CRLF files as CRLF
            with open(py_file, encoding="utf-8", newline="") as f:
                original = f.read()
        except (OSError, UnicodeDecodeError):
            continue

        modified = original
        for pattern, replacement in NUMPY_ALIASES:
            modified = pattern.sub(replacement, modified)

        if modified != original:
            with open(py_file, "w", encoding="utf-8", newline="") as f:
                f.write(modified)
            patched.append(py_file)
            print(f"  Patched NumPy aliases: {py_file.relative_to(talknet_dir)}")

    if not patched:
        print("✓ NumPy aliases already patched")
    return patched


def setup_talknet(talknet_dir=DEFAULT_TALKNET_DIR):
    """
    Run all setup steps (each is skipped when already done).
    Raises RuntimeError, subprocess.CalledProcessError or OSError on failure.
    """
    clone_talknet(talknet_dir)
    download_weights(talknet_dir)
    patch_numpy_aliases(talknet_dir)


def main():
    parser = argparse.ArgumentParser(description="Set up TalkNet-ASD for auto_reframe.py")
    parser.add_argument(
        "--talknet-dir",
        default=str(DEFAULT_TALKNET_DIR),
        help="Where to clone TalkNet-ASD (default: temp_work/TalkNet-ASD)",
    )
    args = parser.parse_args()

    try:
        setup_talknet(args.talknet_dir)
    except (RuntimeError, subprocess.CalledProcessError, OSError) as e:
        print(f"❌ TalkNet setup failed: {e}")
        sys.exit(1)
    print(f"✓ TalkNet ready: {args.talknet_dir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    main()
