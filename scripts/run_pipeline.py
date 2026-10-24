#!/usr/bin/env python3
"""
AI Clipper - pipeline runner

Runs the full pipeline end-to-end: audio chunking → transcription → analysis →
merging → video cutting → TalkNet reframe → captions → badge overlays.
One command: a podcast video in, vertical clips with captions and badges out.

Usage:
    python scripts/run_pipeline.py --input input/video.mp4 --topics controversial funny
    python scripts/run_pipeline.py --input input/video.mp4 --topics funny --cloud --dry-run
    python scripts/run_pipeline.py --config pipeline.yaml
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def override_argv(args_list):
    """Temporarily replace sys.argv for scripts that use argparse."""
    old_argv = sys.argv
    sys.argv = args_list
    try:
        yield
    finally:
        sys.argv = old_argv


DONE_MARKER = ".step_done"


def step_complete(output_dir, pattern):
    """A step is complete only if it finished: run_step() writes DONE_MARKER after
    success. Any output file is not enough, an interrupted run leaves partial ones."""
    return (Path(output_dir) / DONE_MARKER).exists()


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def format_duration_minutes(seconds):
    """Format seconds into minutes with one decimal."""
    return f"{seconds / 60:.1f} minutes"


def run_step(step_num, total, emoji, label, func, skip=False, skip_reason="", done_dir=None):
    """Execute a pipeline step with timing and error handling."""
    print(f"\n[{step_num}/{total}] {emoji} {label}...")
    if skip:
        print(f"      ⏭️  Skipped ({skip_reason})")
        return 0.0
    start = time.time()
    try:
        func()
        elapsed = time.time() - start
        print(f"      ✅ Complete ({format_time(elapsed)})")
        if done_dir:
            (Path(done_dir) / DONE_MARKER).touch()
        return elapsed
    except SystemExit as e:
        if e.code is None or e.code == 0:
            elapsed = time.time() - start
            print(f"      ✅ Complete ({format_time(elapsed)})")
            if done_dir:
                (Path(done_dir) / DONE_MARKER).touch()
            return elapsed
        elapsed = time.time() - start
        print(f"      ❌ Failed (exit code {e.code}, {format_time(elapsed)})")
        raise RuntimeError(f"Step {step_num} '{label}' failed (exit code {e.code})")
    except RuntimeError:
        raise
    except Exception as e:
        elapsed = time.time() - start
        print(f"      ❌ Failed ({format_time(elapsed)})")
        raise RuntimeError(f"Step {step_num} '{label}' failed: {e}")


def load_config(config_path):
    """Load pipeline config from YAML file."""
    try:
        import yaml
    except ImportError:
        print("❌ PyYAML required for --config. Install: pip install pyyaml")
        sys.exit(1)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Clipper - pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python scripts/run_pipeline.py --input input/video.mp4 --topics controversial funny

  # Cloud LLM analysis
  python scripts/run_pipeline.py --input input/video.mp4 --topics funny --cloud --model cogito-2.1:671b

  # Dry run (preview commands)
  python scripts/run_pipeline.py --input input/video.mp4 --topics controversial --dry-run

  # Resume from step 4
  python scripts/run_pipeline.py --input input/video.mp4 --topics controversial --from 4

  # No GPU face tracking: whole frame on a blurred background (still captioned + badged)
  python scripts/run_pipeline.py --input input/video.mp4 --topics funny --vertical-mode pad

  # Landscape output (no reframe, captions or badge)
  python scripts/run_pipeline.py --input input/video.mp4 --topics funny --skip-vertical

  # Using config file
  python scripts/run_pipeline.py --config pipeline.yaml
        """,
    )

    # Required
    parser.add_argument("--input", help="Input video file (.mp4)")
    parser.add_argument("--topics", nargs="+", help="Topics to find (space-separated)")

    # Output
    parser.add_argument(
        "--output-dir",
        default="output/",
        help="Base output directory (default: output/)",
    )

    # Analysis
    parser.add_argument(
        "--cloud", action="store_true", help="Use cloud LLM (analyze_content_cloud.py)"
    )
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--api-key", default=None, help="API key for cloud LLM")
    parser.add_argument(
        "--language",
        default="Indonesian",
        help="Content language (default: Indonesian)",
    )

    # Video
    parser.add_argument(
        "--vertical-mode",
        choices=["auto", "crop", "pad"],
        default="auto",
        help="Vertical conversion: auto = TalkNet follows the speaker (CUDA GPU), "
        "pad = whole frame on a blurred background, crop = fixed crop (default: auto)",
    )
    parser.add_argument(
        "--pad-blur",
        action="store_true",
        default=True,
        help="Blurred background in pad mode (default: enabled)",
    )
    parser.add_argument(
        "--no-pad-blur", action="store_true", help="Disable blurred background"
    )
    parser.add_argument(
        "--crop-position",
        choices=["left", "center", "right"],
        default="center",
        help="Crop position for crop mode (default: center)",
    )
    parser.add_argument(
        "--codec",
        choices=["copy", "libx264"],
        default=None,
        help="Video codec (default: libx264 for vertical, copy otherwise)",
    )

    # Hook title
    parser.add_argument(
        "--badge-duration",
        type=float,
        default=4.5,
        help="Hook title display seconds (default: 4.5, like OpusClip)",
    )

    # Pipeline Control
    parser.add_argument(
        "--force", action="store_true", help="Re-run all steps (ignore checkpoints)"
    )
    parser.add_argument(
        "--from",
        type=int,
        dest="from_step",
        default=1,
        choices=range(1, 9),
        metavar="N",
        help="Start from step N (1-8)",
    )
    parser.add_argument(
        "--skip-subtitles", action="store_true", help="Skip captions (step 7)"
    )
    parser.add_argument(
        "--caption-style",
        choices=["karaoke", "normal"],
        default="karaoke",
        help="karaoke: the spoken word turns yellow (default); normal: plain white subtitles",
    )
    parser.add_argument(
        "--skip-badge", action="store_true", help="Skip the hook title overlay (step 8)"
    )
    parser.add_argument(
        "--skip-vertical",
        action="store_true",
        help="Output landscape clips instead of vertical",
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Skip steps 1-2 (use existing transcripts)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    parser.add_argument("--config", default=None, help="Path to pipeline YAML config")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")

    # Advanced (passthrough)
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=30,
        help="Audio chunk length in seconds (default: 30)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Audio chunk overlap in seconds (default: 0)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.67,
        help="Clip score threshold, 0-1 (default: 0.67: about 33 clips from a 46-min podcast with gemma4:12b-it-q4_K_M)",
    )
    parser.add_argument(
        "--strategy",
        choices=["story", "smart", "aggressive", "conservative"],
        default="story",
        help="Merge strategy (default: story: keep the analyzer's stories as cut)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=30,
        help="Minimum clip duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60,
        help="Maximum clip duration in seconds (story clips over it are fit by the LLM, see merge_rulebased --fit-to)",
    )
    parser.add_argument(
        "--optimizer",
        action="store_true",
        default=False,
        help="Enable post-merge optimizer (combine adjacent clips with small gaps)",
    )
    parser.add_argument(
        "--optimizer-max-gap",
        type=float,
        default=5.0,
        help="Max gap for optimizer merging in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--audio-rank",
        action="store_true",
        default=False,
        help="Blend laughter + loudness into the clip ranking (ranking.md; AST model on CPU, ~5 min per hour of audio)",
    )
    parser.add_argument(
        "--edge-check",
        action="store_true",
        default=False,
        help="Step 3: one more LLM call per clip to fix where it starts and ends",
    )
    parser.add_argument(
        "--window-overlap",
        type=float,
        default=0.0,
        help="Step 3: overlap the 5-minute transcript windows by this many seconds (default: 0)",
    )

    return parser


def merge_argv(args, paths):
    """argv for merge_rulebased. Story runs fit long clips with the LLM the analysis used (cloud too: the even
    trim scored worst, and an unusable reply drops the clip) and drop the show's opening; other strategies trim."""
    argv = [
        "merge_rulebased.py",
        "--input",
        paths["analysis"],
        "--output",
        paths["merged"],
        "--strategy",
        args.strategy,
        "--audio",
        args.input,
    ]
    if args.min_duration:
        argv.extend(["--min-duration", str(args.min_duration)])
    if args.strategy == "story":
        argv.extend(["--drop-openings", "--transcripts", paths["transcripts"]])
        if args.max_duration:
            argv.extend(["--fit-to", str(args.max_duration)])
        model = args.model
        if not model and args.cloud:
            from analyze_content_cloud import DEFAULT_MODEL as model
        if model:  # local default: merge_rulebased's own (gemma4)
            argv.extend(["--model", model])
    elif args.max_duration:
        argv.extend(["--max-duration", str(args.max_duration)])
    if args.verbose:
        argv.append("--verbose")
    return argv


def warn_long_clips(merged_json, max_duration):
    """An old merged_segments.json may hold clips the cutter now drops (default limit went 120 -> 60 s)."""
    try:
        with open(merged_json, encoding="utf-8") as f:
            n = sum(1 for s in json.load(f).get("segments", []) if s.get("duration", 0) > max_duration)
    except Exception:  # best-effort warning: a bad file must not crash a resumed run
        return
    if n:
        print(f"      ⚠️  {n} clips are over max_duration ({max_duration:g} s) and the cutter will drop them; re-run with --force to merge again")


def apply_config(args, config):
    """Apply YAML config values where CLI args weren't explicitly set."""
    # Compare with the parser's own defaults: a hand-kept copy drifted (min_confidence
    # 0.7 vs 0.85, durations None vs 40/120) and silently ignored those YAML keys.
    defaults = vars(build_parser().parse_args([]))
    config = config or {}  # an empty file loads as None
    # The old config.yaml layout (model:/analysis:/paths: sections) put a whole dict into
    # args.model and silently dropped topics/input; say so instead.
    nested = [k for k, v in config.items() if isinstance(v, dict)]
    if nested:
        print(f"❌ Config sections aren't supported ({', '.join(nested)}): put keys at the top level, see pipeline.yaml.example")
        sys.exit(1)
    keys = (
        "input", "topics", "output_dir", "cloud", "model", "api_key", "language",
        "vertical_mode", "pad_blur", "crop_position", "codec", "badge_duration",
        "chunk_length", "overlap", "min_confidence", "strategy", "min_duration",
        "max_duration", "verbose", "skip_badge", "skip_subtitles", "caption_style", "skip_vertical", "optimizer",
        "optimizer_max_gap", "audio_rank", "edge_check", "window_overlap",
    )
    for key in keys:
        if key in config and getattr(args, key, None) == defaults[key]:
            setattr(args, key, config[key])


def build_paths(args):
    """Output folders, plus which one each video step reads for this mode."""
    base = Path(args.output_dir)
    paths = {
        "audio_chunks": str(base / "audio_chunks"),
        "transcripts": str(base / "transcripts"),
        "analysis": str(base / "analysis_results"),
        "merged": str(base / "analysis_results" / "merged_optimized"),
        "merged_json": str(base / "analysis_results" / "merged_optimized" / "merged_segments.json"),
        "clips": str(base / "clips"),
        "clips_vertical": str(base / "clips_vertical"),
        "clips_subtitled": str(base / "clips_subtitled"),
        "clips_final": str(base / "clips_final"),
    }
    vertical = not args.skip_vertical
    reframe = vertical and args.vertical_mode == "auto"
    # TalkNet reframes landscape clips, so in auto mode the cutter writes landscape
    paths["cut"] = paths["clips_vertical"] if vertical and not reframe else paths["clips"]
    paths["clips_metadata"] = str(Path(paths["cut"]) / "clips_metadata.json")
    paths["badge_input"] = paths["clips_subtitled"] if vertical and not args.skip_subtitles else paths["clips_vertical"]
    return paths


def free_gpu():
    """Before TalkNet and the caption ASR: Ollama keeps the LLM in VRAM for 5 min after the
    merge step (gemma4:12b is 8 GB of a 12 GB card), and this process still caches the
    step-2 ASR memory. Release both; best effort, a failure here must not stop the run."""
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import requests

        url = "http://localhost:11434"
        for m in requests.get(f"{url}/api/ps", timeout=5).json().get("models", []):
            if m.get("size_vram"):
                requests.post(f"{url}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
                print(f"      Unloaded {m['name']} from the GPU")
    except Exception:
        pass


def print_dry_run(args, paths):
    """Print equivalent CLI commands without executing."""
    print("\n[DRY RUN] Pipeline would execute:\n")

    scripts_dir = Path(__file__).parent
    use_vertical = not args.skip_vertical
    reframe = use_vertical and args.vertical_mode == "auto"
    use_captions = use_vertical and not args.skip_subtitles
    use_badge = not args.skip_badge and use_vertical

    # Determine codec
    codec = args.codec or ("libx264" if use_vertical else "copy")

    # Step 1
    print(f"[1/8] python {scripts_dir / 'chunk_audio.py'} \\")
    print(f"        --input {args.input} --output {paths['audio_chunks']} \\")
    print(f"        --chunk-length {args.chunk_length} --overlap {args.overlap}")
    print()

    # Step 2
    print(f"[2/8] python {scripts_dir / 'transcribe.py'} \\")
    print(f"        --input {paths['audio_chunks']} --output {paths['transcripts']} \\")
    print(
        f"        --merge-window 300 --chunk-duration {args.chunk_length} --overlap {args.overlap}"
    )
    print()

    # Step 3
    analyze_script = "analyze_content_cloud.py" if args.cloud else "analyze_content.py"
    print(f"[3/8] python {scripts_dir / analyze_script} \\")
    print(f"        --input {paths['transcripts']} --output {paths['analysis']} \\")
    topics_str = " ".join(args.topics)
    extra = ""
    if args.model:
        extra += f" --model {args.model}"
    if args.cloud and args.api_key:
        extra += " --api-key ***"  # the preview goes to logs and terminal history
    print(
        f"        --topics {topics_str} --language {args.language} --min-confidence {args.min_confidence}{extra}"
    )
    print()

    # Step 4
    print(f"[4/8] python {scripts_dir / 'merge_rulebased.py'} \\")
    print(f"        --input {paths['analysis']} --output {paths['merged']} \\")
    print(f"        --strategy {args.strategy} --audio {args.input}")
    print()

    # Step 5
    print(f"[5/8] python {scripts_dir / 'video_cutter.py'} \\")
    print(f"        --input {args.input} \\")
    print(f"        --segments {paths['merged_json']} \\")
    print(f"        --output {paths['cut']} --codec {codec}")
    if args.optimizer:
        print(f"        --optimizer --optimizer-max-gap {args.optimizer_max_gap}")
    if use_vertical and not reframe:
        blur_flag = " --pad-blur" if args.pad_blur else ""
        print(f"        --vertical --vertical-mode {args.vertical_mode}{blur_flag}")
    print()

    # Step 6
    if reframe:
        print(f"[6/8] python {scripts_dir / 'auto_reframe.py'} \\")
        print(f"        --input-dir {paths['cut']} --output-dir {paths['clips_vertical']}")
    else:
        print("[6/8] (skipped)")
    print()

    # Step 7
    if use_captions:
        print(f"[7/8] python {scripts_dir / 'add_subtitles.py'} \\")
        print(f"        --input-dir {paths['clips_vertical']} --output-dir {paths['clips_subtitled']} --language {args.language} --caption-style {args.caption_style}")
    else:
        print("[7/8] (skipped)")
    print()

    # Step 8
    if use_badge:
        print(f"[8/8] python {scripts_dir / 'add_badge.py'} \\")
        print(f"        --input-dir {paths['badge_input']} --output-dir {paths['clips_final']} \\")
        print(
            f"        --metadata {paths['clips_metadata']} --duration {args.badge_duration}"
        )
    else:
        print(f"[8/8] (skipped)")
    print()


def run_pipeline(args):
    """Run the full pipeline."""
    scripts_dir = Path(__file__).resolve().parent

    # Ensure scripts dir is on sys.path for imports
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    paths = build_paths(args)
    use_vertical = not args.skip_vertical
    reframe = use_vertical and args.vertical_mode == "auto"
    use_captions = use_vertical and not args.skip_subtitles
    use_badge = not args.skip_badge and use_vertical
    codec = args.codec or ("libx264" if use_vertical else "copy")
    clips_dir = paths["cut"]

    total_steps = 8
    step_times = {}

    # Print header
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  AI CLIPPER - Pipeline                                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Topics: {', '.join(args.topics)}")
    if args.cloud:
        print(f"LLM:    Cloud ({args.model or 'default'})")
    if args.force:
        print(f"Mode:   Force re-run (ignoring checkpoints)")
    if args.from_step > 1:
        print(f"Mode:   Starting from step {args.from_step}")

    # ── Step 1: Audio Chunking ──
    should_skip = (
        args.from_step > 1
        or args.skip_transcribe
        or (not args.force and step_complete(paths["audio_chunks"], "*.wav"))
    )
    skip_reason = ""
    if args.from_step > 1:
        skip_reason = f"starting from step {args.from_step}"
    elif args.skip_transcribe:
        skip_reason = "--skip-transcribe"
    elif not args.force and step_complete(paths["audio_chunks"], "*.wav"):
        count = len(list(Path(paths["audio_chunks"]).glob("*.wav")))
        skip_reason = f"{count} chunks already exist"

    def do_step1():
        from chunk_audio import chunk_audio

        chunk_audio(
            args.input,
            paths["audio_chunks"],
            chunk_length=args.chunk_length,
            overlap=args.overlap,
        )

    step_times[1] = run_step(
        1,
        total_steps,
        "📦",
        "Chunking audio",
        do_step1,
        skip=should_skip,
        skip_reason=skip_reason,
        done_dir=paths["audio_chunks"],
    )

    # ── Step 2: Transcription ──
    should_skip = (
        args.from_step > 2
        or args.skip_transcribe
        or (not args.force and step_complete(paths["transcripts"], "*_merged*.txt"))
    )
    skip_reason = ""
    if args.from_step > 2:
        skip_reason = f"starting from step {args.from_step}"
    elif args.skip_transcribe:
        skip_reason = "--skip-transcribe"
    elif not args.force and step_complete(paths["transcripts"], "*_merged*.txt"):
        count = len(list(Path(paths["transcripts"]).glob("*_merged*.txt")))
        skip_reason = f"{count} transcript(s) already exist"

    def do_step2():
        from transcribe import main as transcribe_main

        argv = [
            "transcribe.py",
            "--input",
            paths["audio_chunks"],
            "--output",
            paths["transcripts"],
            "--merge-window",
            "300",
            "--chunk-duration",
            str(args.chunk_length),
            "--overlap",
            str(args.overlap),
        ]
        with override_argv(argv):
            transcribe_main()

    step_times[2] = run_step(
        2,
        total_steps,
        "📝",
        "Transcribing",
        do_step2,
        skip=should_skip,
        skip_reason=skip_reason,
        done_dir=paths["transcripts"],
    )

    # ── Step 3: Content Analysis ──
    should_skip = args.from_step > 3 or (
        not args.force and step_complete(paths["analysis"], "*_analysis.json")
    )
    skip_reason = ""
    if args.from_step > 3:
        skip_reason = f"starting from step {args.from_step}"
    elif not args.force and step_complete(paths["analysis"], "*_analysis.json"):
        count = len(list(Path(paths["analysis"]).glob("*_analysis.json")))
        skip_reason = f"{count} analysis file(s) already exist"

    def do_step3():
        if args.cloud:
            from analyze_content_cloud import main as analyze_main

            script_name = "analyze_content_cloud.py"
        else:
            from analyze_content import main as analyze_main

            script_name = "analyze_content.py"

        argv = [
            script_name,
            "--input",
            paths["transcripts"],
            "--output",
            paths["analysis"],
            "--topics",
            *args.topics,
            "--language",
            args.language,
            "--min-confidence",
            str(args.min_confidence),
        ]
        if args.model:
            argv.extend(["--model", args.model])
        if args.cloud and args.api_key:
            argv.extend(["--api-key", args.api_key])
        if args.verbose:
            argv.append("--verbose")
        if args.edge_check:
            argv.append("--edge-check")
        if args.window_overlap:
            argv.extend(["--window-overlap", str(args.window_overlap)])

        with override_argv(argv):
            result = analyze_main()
            if result and result != 0:
                raise RuntimeError(f"Analysis returned error code {result}")

    step_times[3] = run_step(
        3,
        total_steps,
        "🔍",
        "Analyzing content",
        do_step3,
        skip=should_skip,
        skip_reason=skip_reason,
        done_dir=paths["analysis"],
    )

    # ── Step 4: Segment Merging ──
    should_skip = args.from_step > 4 or (
        not args.force and Path(paths["merged_json"]).exists()
    )
    skip_reason = ""
    if args.from_step > 4:
        skip_reason = f"starting from step {args.from_step}"
    elif not args.force and Path(paths["merged_json"]).exists():
        skip_reason = "merged_segments.json already exists"
        warn_long_clips(paths["merged_json"], args.max_duration)

    def do_step4():
        from merge_rulebased import main as merge_main

        argv = merge_argv(args, paths)

        with override_argv(argv):
            result = merge_main()
            if result and result != 0:
                raise RuntimeError(f"Merge returned error code {result}")

        rank_argv = ["rank_clips.py", "--merged", paths["merged_json"]]
        if args.audio_rank:
            rank_argv += ["--media", args.input]
        from rank_clips import main as rank_main

        with override_argv(rank_argv):
            rank_main()

    step_times[4] = run_step(
        4,
        total_steps,
        "🔗",
        "Merging segments",
        do_step4,
        skip=should_skip,
        skip_reason=skip_reason,
    )

    # ── Step 5: Video Cutting ──
    clip_pattern = "*.mp4"
    should_skip = args.from_step > 5 or (
        not args.force and step_complete(clips_dir, clip_pattern)
    )
    skip_reason = ""
    if args.from_step > 5:
        skip_reason = f"starting from step {args.from_step}"
    elif not args.force and step_complete(clips_dir, clip_pattern):
        count = len(list(Path(clips_dir).glob(clip_pattern)))
        skip_reason = f"{count} clip(s) already exist"

    def do_step5():
        from video_cutter import main as cutter_main

        argv = [
            "video_cutter.py",
            "--input",
            args.input,
            "--segments",
            paths["merged_json"],
            "--output",
            clips_dir,
            "--codec",
            codec,
        ]
        if use_vertical and not reframe:
            argv.extend(["--vertical", "--vertical-mode", args.vertical_mode])
            if args.pad_blur and args.vertical_mode == "pad":
                argv.append("--pad-blur")
            if args.vertical_mode == "crop":
                argv.extend(["--crop-position", args.crop_position])
        if args.optimizer:
            argv.append("--optimizer")
            argv.extend(["--optimizer-max-gap", str(args.optimizer_max_gap)])
        if args.min_duration:
            argv.extend(["--min-duration", str(args.min_duration)])
        if args.max_duration:
            argv.extend(["--max-duration", str(args.max_duration)])
        if args.verbose:
            argv.append("--verbose")

        with override_argv(argv):
            cutter_main()

    step_times[5] = run_step(
        5,
        total_steps,
        "✂️ ",
        "Cutting video clips",
        do_step5,
        skip=should_skip,
        skip_reason=skip_reason,
        done_dir=clips_dir,
    )

    # ── Step 6: TalkNet Reframe ──
    should_skip = (
        not reframe
        or args.from_step > 6
        or (not args.force and step_complete(paths["clips_vertical"], "*.mp4"))
    )
    skip_reason = ""
    if not reframe:
        skip_reason = "landscape mode" if not use_vertical else f"--vertical-mode {args.vertical_mode} (cut as vertical)"
    elif args.from_step > 6:
        skip_reason = f"starting from step {args.from_step}"
    elif not args.force and step_complete(paths["clips_vertical"], "*.mp4"):
        count = len(list(Path(paths["clips_vertical"]).glob("*.mp4")))
        skip_reason = f"{count} reframed clip(s) already exist"

    def do_step6():
        free_gpu()
        from auto_reframe import main as reframe_main

        argv = ["auto_reframe.py", "--input-dir", clips_dir, "--output-dir", paths["clips_vertical"]]
        with override_argv(argv):
            reframe_main()

    step_times[6] = run_step(
        6,
        total_steps,
        "🎯",
        "Reframing on the speaker (TalkNet)",
        do_step6,
        skip=should_skip,
        skip_reason=skip_reason,
        done_dir=paths["clips_vertical"],
    )

    # ── Step 7: Captions ──
    should_skip = (
        not use_captions
        or args.from_step > 7
        or (not args.force and step_complete(paths["clips_subtitled"], "*.mp4"))
    )
    skip_reason = ""
    if not use_captions:
        skip_reason = "--skip-subtitles" if args.skip_subtitles else "landscape mode"
    elif args.from_step > 7:
        skip_reason = f"starting from step {args.from_step}"
    elif not args.force and step_complete(paths["clips_subtitled"], "*.mp4"):
        count = len(list(Path(paths["clips_subtitled"]).glob("*.mp4")))
        skip_reason = f"{count} captioned clip(s) already exist"

    def do_step7():
        free_gpu()
        from add_subtitles import main as subtitles_main

        argv = [
            "add_subtitles.py",
            "--input-dir",
            paths["clips_vertical"],
            "--output-dir",
            paths["clips_subtitled"],
            "--language",
            args.language,
            "--caption-style",
            args.caption_style,
        ]
        with override_argv(argv):
            subtitles_main()

    step_times[7] = run_step(
        7,
        total_steps,
        "💬",
        "Adding captions",
        do_step7,
        skip=should_skip,
        skip_reason=skip_reason,
        done_dir=paths["clips_subtitled"],
    )

    # ── Step 8: Badge Overlays ──
    should_skip_badge = (
        not use_badge
        or args.from_step > 8
        or (not args.force and step_complete(paths["clips_final"], "*.mp4"))
    )
    skip_reason = ""
    if not use_badge:
        if args.skip_badge:
            skip_reason = "--skip-badge"
        else:
            skip_reason = "landscape mode (no badge)"
    elif args.from_step > 8:
        skip_reason = f"starting from step {args.from_step}"
    elif not args.force and step_complete(paths["clips_final"], "*.mp4"):
        count = len(list(Path(paths["clips_final"]).glob("*.mp4")))
        skip_reason = f"{count} badged clip(s) already exist"

    def do_step8():
        free_gpu()  # the caption ASR is still cached; the hook titles load the LLM again
        from add_badge import main as badge_main

        argv = [
            "add_badge.py",
            "--input-dir",
            paths["badge_input"],
            "--output-dir",
            paths["clips_final"],
            "--metadata",
            paths["clips_metadata"],
            "--duration",
            str(args.badge_duration),
            "--language",
            args.language,
        ]
        model = args.model
        if not model and args.cloud:
            from analyze_content_cloud import DEFAULT_MODEL as model
        if model:  # local default: add_badge's own (gemma4), like merge_rulebased
            argv.extend(["--title-model", model])
        if args.verbose:
            argv.append("--verbose")

        with override_argv(argv):
            badge_main()

    step_times[8] = run_step(
        8,
        total_steps,
        "🏷️ ",
        "Adding hook titles",
        do_step8,
        skip=should_skip_badge,
        skip_reason=skip_reason,
        done_dir=paths["clips_final"],
    )

    # ── Summary ──
    total_time = sum(step_times.values())

    # Count final clips
    if use_badge:
        final_dir = paths["clips_final"]
    elif use_vertical:
        final_dir = paths["badge_input"]
    else:
        final_dir = clips_dir
    final_clips = (
        list(Path(final_dir).glob("*.mp4")) if Path(final_dir).exists() else []
    )
    clip_count = len(final_clips)

    # Calculate total duration from clips metadata or segments
    total_duration = 0
    if Path(paths["merged_json"]).exists():
        try:
            with open(paths["merged_json"], "r") as f:
                data = json.load(f)
            segments = data.get("segments", data if isinstance(data, list) else [])
            for seg in segments:
                total_duration += seg.get("end_time", 0) - seg.get("start_time", 0)
        except Exception:
            pass

    print()
    print("══════════════════════════════════════════════════════════════")
    print("✅ PIPELINE COMPLETE")
    print("══════════════════════════════════════════════════════════════")
    print(f"Total time: {format_time(total_time)}")
    print(f"Final clips: {clip_count} files in {final_dir}")
    if total_duration > 0:
        print(f"Total duration: {format_duration_minutes(total_duration)} of content")
    print(f"Ready to upload!")
    print("══════════════════════════════════════════════════════════════")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        if not Path(args.config).exists():
            print(f"❌ Config file not found: {args.config}")
            sys.exit(1)
        config = load_config(args.config)
        apply_config(args, config)

    # Handle --no-pad-blur
    if args.no_pad_blur:
        args.pad_blur = False

    # Validate required arguments
    if not args.input:
        print("❌ --input is required (or set 'input' in config file)")
        sys.exit(1)
    if not args.topics:
        print("❌ --topics is required (or set 'topics' in config file)")
        sys.exit(1)

    # Validate input file exists (skip for dry-run)
    if not args.dry_run and not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    paths = build_paths(args)

    if args.dry_run:
        print_dry_run(args, paths)
        return

    run_pipeline(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    main()
