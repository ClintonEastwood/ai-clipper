import os
import sys
import glob
import json
import re
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path

from talknet_speaker_detection import analyze_talknet_layout, analyze_talknet_per_scene
from setup_talknet import DEFAULT_TALKNET_DIR, setup_talknet

PYTHON = sys.executable

# demoTalkNet.py re-encodes every input with `-r 25`, so TalkNet frame numbers
# are always at 25 fps, whatever the clip's own frame rate is.
TALKNET_FPS = 25.0


def run_talknet(video_path, talknet_dir):
    """
    Run TalkNet on the given video and return the path to the pywork directory.
    """
    # demoTalkNet.py runs its own FFmpeg commands through a shell with unquoted
    # paths, so give it a name with no spaces or shell characters.
    video_name = re.sub(r"[^A-Za-z0-9_-]", "_", Path(video_path).stem)

    # TalkNet's output dir logic
    demo_videos_dir = os.path.join(talknet_dir, "demo_videos")
    os.makedirs(demo_videos_dir, exist_ok=True)

    # Copy video to talknet demo folder (required by demoTalkNet.py)
    dest_video = os.path.join(demo_videos_dir, f"{video_name}.mp4")
    if os.path.abspath(video_path) != os.path.abspath(dest_video):
        import shutil

        shutil.copy2(video_path, dest_video)

    pretrain_model = os.path.join(talknet_dir, "pretrain_TalkSet.model")
    if not os.path.exists(pretrain_model):
        print(f"Error: TalkNet model not found at {pretrain_model}")
        return None

    print(f"Running TalkNet on {video_name}...")
    cmd = [
        PYTHON, "demoTalkNet.py",
        "--videoName", video_name,
        "--videoFolder", demo_videos_dir,
        "--pretrainModel", pretrain_model,
    ]
    result = subprocess.run(cmd, cwd=talknet_dir)

    if result.returncode != 0:
        print(f"Error running TalkNet on {video_name}")
        return None

    pywork_path = os.path.join(demo_videos_dir, video_name, "pywork")
    return pywork_path


# Crop sizes below were tuned on 1080p input; compute_crop() scales them by
# frame height so a 480p two-shot gets the same framing as a 1080p one.
REF_HEIGHT = 1080
MIN_CROP_W = 400       # px at 1080p
MAX_CROP_W = 1080      # px at 1080p
DEFAULT_FACE_S = 150   # face half-size (px at 1080p) when TalkNet gave none
FACE_WIDTH_RATIO = 0.35  # face (2*s) fills ~35% of the crop width
HEADROOM = 0.60          # face center sits 60% down from the crop top

SINGLE_ASPECT = 9 / 16  # crop w/h, scaled to 1080x1920
SPLIT_ASPECT = 9 / 8    # each half, scaled to 1080x960

ENCODE_ARGS = [
    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
    "-c:a", "aac", "-b:a", "128k",
]


def compute_crop(coord, width, height, aspect):
    """
    Crop box around one face, as (w, h, x, y) in source pixels.

    coord: {"x", "y", "s"} face center and half-size from TalkNet.
    aspect: crop width / height (SINGLE_ASPECT or SPLIT_ASPECT).
    The box keeps the aspect ratio, is shrunk if needed to fit inside the
    width x height frame, and is shifted (not shrunk) to stay inside it.
    """
    scale = height / REF_HEIGHT
    cx = coord["x"]
    cy = coord.get("y", height / 2)
    s = coord.get("s") or DEFAULT_FACE_S * scale

    crop_w = min(MAX_CROP_W * scale, max(MIN_CROP_W * scale, 2 * s / FACE_WIDTH_RATIO))
    crop_h = crop_w / aspect
    fit = min(1.0, width / crop_w, height / crop_h)
    # Even sizes keep FFmpeg's yuv420p crop exact
    crop_w = max(2, int(crop_w * fit) // 2 * 2)
    crop_h = max(2, int(crop_h * fit) // 2 * 2)

    crop_x = int(min(max(cx - crop_w / 2, 0), width - crop_w))
    crop_y = int(min(max(cy - crop_h * HEADROOM, 0), height - crop_h))
    return crop_w, crop_h, crop_x, crop_y


def _video_filter_args(mode, coords, width, height):
    """FFmpeg video filter arguments for one layout. Output is always 1080x1920."""
    if mode == "single" and coords:
        w, h, x, y = compute_crop(coords[0], width, height, SINGLE_ASPECT)
        return ["-vf", f"crop={w}:{h}:{x}:{y},scale=1080:1920"]

    if mode == "split" and len(coords) == 2:
        # Split Screen for 2 Speakers (Top/Bottom) - OpusClip style tight crop
        w1, h1, x1, y1 = compute_crop(coords[0], width, height, SPLIT_ASPECT)
        w2, h2, x2, y2 = compute_crop(coords[1], width, height, SPLIT_ASPECT)
        fc = (
            f"[0:v]crop={w1}:{h1}:{x1}:{y1},scale=1080:960[top];"
            f"[0:v]crop={w2}:{h2}:{x2}:{y2},scale=1080:960[bottom];"
            f"[top][bottom]vstack[out]"
        )
    else:
        # Fallback to Pad mode (Blur Background)
        fc = (
            "[0:v]split[original][copy];"
            "[copy]scale=270:480,boxblur=10:5,scale=1080:1920[blurred];"  # blur at 1/4 size: same look, ~14x faster
            "[original]scale=1080:-1[scaled];"
            "[blurred][scaled]overlay=0:(H-h)/2[out]"
        )
    return ["-filter_complex", fc, "-map", "[out]", "-map", "0:a"]


def apply_ffmpeg_layout(input_video, output_video, layout, width=1920, height=1080):
    """
    Generates and runs the FFmpeg command based on the layout mode.
    """
    mode = layout.get("mode", "pad")
    coords = layout.get("target_coordinates", [])

    print(f"Applying layout mode: {mode.upper()}")

    cmd = (
        ["ffmpeg", "-y", "-i", input_video]
        + _video_filter_args(mode, coords, width, height)
        + ENCODE_ARGS
        + [output_video]
    )
    print(f"Executing FFmpeg: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def get_video_fps(video_path):
    """Get video FPS via ffprobe. Returns 25.0 on failure."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "0",
            "-of",
            "csv=p=0",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        s = result.stdout.strip()
        try:
            if "/" in s:
                num, den = s.split("/")
                return float(num) / float(den)
            return float(s)
        except (ValueError, ZeroDivisionError):
            pass
    return 25.0


def get_video_duration(video_path):
    """Get video duration in seconds via ffprobe. Returns None on failure."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "0",
            "-of",
            "csv=p=0",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    return None


def get_video_size(video_path):
    """Get video (width, height) via ffprobe. Returns (1920, 1080) on failure."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "0",
            "-of",
            "csv=p=0",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            w, h = result.stdout.strip().split(",")[:2]
            return int(w), int(h)
        except ValueError:
            pass
    return 1920, 1080


def _build_segment_cmd(
    input_video, seg_path, start_t, duration, mode, coords, width=1920, height=1080
):
    """
    Build an FFmpeg command to extract and crop one time segment.
    Uses -ss before -i (fast input seek) + -t duration for precision.
    All outputs are 1080x1920 so concat -c copy works cleanly.
    """
    seek = ["-ss", f"{start_t:.6f}", "-t", f"{duration:.6f}"]
    return (
        ["ffmpeg", "-y"]
        + seek
        + ["-i", input_video]
        + _video_filter_args(mode, coords, width, height)
        + ENCODE_ARGS
        + [seg_path]
    )


def apply_ffmpeg_per_scene(
    input_video, output_video, scene_layouts, width=1920, height=1080
):
    """
    Encode each scene segment with its own crop, then concat losslessly.
    Falls back to apply_ffmpeg_layout if only one scene or scene list is empty.
    """
    if not scene_layouts:
        apply_ffmpeg_layout(
            input_video, output_video, {"mode": "pad", "target_coordinates": []},
            width, height,
        )
        return

    if len(scene_layouts) == 1:
        s = scene_layouts[0]
        apply_ffmpeg_layout(
            input_video,
            output_video,
            {"mode": s["mode"], "target_coordinates": s["target_coordinates"]},
            width,
            height,
        )
        return

    # Use a local temp directory instead of /tmp/
    local_temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_work")
    os.makedirs(local_temp_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="reframe_segs_", dir=local_temp_dir)
    segment_files = []

    try:
        for i, scene in enumerate(scene_layouts):
            start_t = scene["start_time"]
            end_t = scene["end_time"]
            duration = end_t - start_t

            if duration < 0.1:
                continue  # skip micro-segments that would confuse FFmpeg

            seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.mp4")
            mode = scene["mode"]
            coords = scene["target_coordinates"]

            print(
                f"  [{i + 1}/{len(scene_layouts)}] {start_t:.2f}s-{end_t:.2f}s [{mode.upper()}]"
            )

            cmd = _build_segment_cmd(
                input_video, seg_path, start_t, duration, mode, coords, width, height
            )
            subprocess.run(cmd, check=True)
            segment_files.append(seg_path)

        if not segment_files:
            apply_ffmpeg_layout(
                input_video, output_video, {"mode": "pad", "target_coordinates": []},
                width, height,
            )
            return

        # Write concat manifest and join all segments (copy, no re-encode)
        print(f"  Concatenating {len(segment_files)} segments...")
        concat_txt = os.path.join(tmp_dir, "concat.txt")
        with open(concat_txt, "w") as f:
            for seg in segment_files:
                # Use forward slashes and escape for FFmpeg concat on Windows
                f.write(f"file '{seg.replace(chr(92), '/')}'\n")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_txt,
                "-c",
                "copy",
                output_video,
            ],
            check=True,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-Reframe Vertical Clips using TalkNet"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing landscape MP4 clips"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save vertical clips"
    )
    parser.add_argument(
        "--talknet-dir",
        default=str(DEFAULT_TALKNET_DIR),
        help="TalkNet-ASD repo directory (cloned and set up on first run)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process all mp4 files in input directory
    clip_files = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))

    if not clip_files:
        print(f"No .mp4 files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(clip_files)} clips to process.")

    # Clone TalkNet, download weights, patch NumPy aliases (skips what's done)
    try:
        setup_talknet(args.talknet_dir)
    except (RuntimeError, subprocess.CalledProcessError, OSError) as e:
        print(f"TalkNet setup failed: {e}")
        sys.exit(1)

    failed = []  # a run where every clip failed (e.g. a broken cuDNN) used to exit 0
    # FFmpeg writes here first, then the clip is moved into place: a run cut off
    # mid-write used to leave a broken .mp4 that the exists-check skipped forever.
    # A subfolder, not a .part name, because later steps glob "*.mp4" (dotfiles too).
    partial_dir = os.path.join(args.output_dir, ".partial")
    for clip in clip_files:
        clip_name = os.path.basename(clip)
        output_path = os.path.join(args.output_dir, clip_name)

        print(f"\n{'=' * 50}\nProcessing {clip_name}\n{'=' * 50}")

        if os.path.exists(output_path):
            print(f"Skipping {clip_name}, output already exists.")
            continue

        # 1. Run TalkNet
        pywork_path = run_talknet(clip, args.talknet_dir)

        if not pywork_path:
            print(f"TalkNet failed on {clip_name}, skipping.")
            failed.append(clip_name)
            continue

        # 2. Get video metadata
        fps = get_video_fps(clip)
        duration = get_video_duration(clip)
        width, height = get_video_size(clip)
        print(
            f"  Video: {width}x{height}, {fps:.2f} fps, {duration:.2f}s"
            if duration
            else f"  Video: {width}x{height}, {fps:.2f} fps"
        )

        # 3. Per-scene layout analysis
        tracks_path = os.path.join(pywork_path, "tracks.pckl")
        scores_path = os.path.join(pywork_path, "scores.pckl")

        print("  Analyzing per-scene layout...")
        scene_layouts = analyze_talknet_per_scene(
            tracks_path, scores_path, fps=TALKNET_FPS,
            video_width=width, video_height=height, video_duration=duration
        )

        if not scene_layouts:
            print("  Per-scene analysis failed, falling back to global layout...")
            layout = analyze_talknet_layout(
                tracks_path, scores_path, video_width=width, video_height=height
            )
            if "error" in layout:
                print(
                    f"  Global analysis error: {layout['error']}, using pad fallback."
                )
                layout = {"mode": "pad", "target_coordinates": []}
            scene_layouts = [
                {
                    "start_time": 0.0,
                    "end_time": duration or 9999,
                    "mode": layout["mode"],
                    "target_coordinates": layout["target_coordinates"],
                }
            ]

        # 4. Apply FFmpeg per-scene crop + concat
        try:
            os.makedirs(partial_dir, exist_ok=True)
            partial_path = os.path.join(partial_dir, clip_name)
            apply_ffmpeg_per_scene(clip, partial_path, scene_layouts, width, height)
            os.replace(partial_path, output_path)
            # Sidecar so later steps (add_subtitles.py) know where faces are
            layout_path = os.path.splitext(output_path)[0] + ".layout.json"
            with open(layout_path, "w") as f:
                json.dump(
                    [{k: s[k] for k in ("start_time", "end_time", "mode")} for s in scene_layouts],
                    f, indent=2,
                )
            print(f"✓ Successfully reframed {clip_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ FFmpeg failed for {clip_name}: {e}")
            failed.append(clip_name)

    shutil.rmtree(partial_dir, ignore_errors=True)
    if failed:
        print(f"\n✗ {len(failed)}/{len(clip_files)} clips failed (rerun to retry only those): {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    main()
