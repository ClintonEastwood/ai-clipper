#!/usr/bin/env python3
"""
add_badge.py - hook title overlay

Writes a hook title per clip (one LLM call on what is said in the clip, e.g.
"Kereta telat sejam: siapa sebenarnya salah?"), draws it as a white
rounded card with black Montserrat Bold (bundled in assets/fonts), and overlays
it with FFmpeg for the first N seconds. Without an LLM it uses the analysis
badge_label. Titles are cached in <output-dir>/titles.json.

Metadata sources (in priority order):
  1. --metadata clips_metadata.json  (output by video_cutter.py)
  2. --segments merged_segments.json (fallback: matches by start_time in filename)

Usage:
    python scripts/add_badge.py \\
        --input-dir clips_vertical/ \\
        --output-dir clips_vertical_badged/ \\
        --metadata clips_vertical/clips_metadata.json

    # Using merged_segments.json as fallback (no badge_label → generated from reason)
    python scripts/add_badge.py \\
        --input-dir clips_vertical/ \\
        --output-dir clips_vertical_badged/ \\
        --segments data/analysis_results/merged_optimized/merged_segments.json

Badge position on vertical clips follows the auto_reframe.py layout sidecar
(<clip>.layout.json, looked up next to the clip or in --layout-dir):
  - pad (or no sidecar) -> in the blur area, ending above the content strip
  - single / split      -> y=110, above the top speaker's head
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ─── Font paths (bundled first, then system: macOS + Windows + Linux) ────────
TEXT_FONT_PATHS = [
    str(Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Montserrat-Bold.ttf"),
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]


def load_text_font(size: int) -> ImageFont.FreeTypeFont:
    for path in TEXT_FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    print(f"  ⚠ WARNING: No system font found! Badge text will be tiny.")
    print(f"  Searched: {TEXT_FONT_PATHS}")
    return ImageFont.load_default()


def generate_label_from_reason(reason: str, max_chars: int = 28) -> str:
    """Extract first clause from reason field as fallback badge label."""
    if not reason:
        return "Highlight"
    # Take first sentence / clause
    first_clause = re.split(r'[.|;|\|]', reason)[0].strip()
    if len(first_clause) > max_chars:
        first_clause = first_clause[:max_chars].rsplit(' ', 1)[0] + "..."
    return first_clause or "Highlight"


# ─── Badge sizing constants ──────────────────────────────────────────────────
MAX_LABEL_CHARS = 60       # Truncate labels beyond this length (titles wrap, up to 3 lines)
MIN_FONT_SIZE = 44         # Never scale font below this
MAX_BADGE_WIDTH_RATIO = 0.75  # OpusClip's card is 810 of 1080 px
TITLE_FONT_SIZE = 56       # Montserrat Bold; OpusClip's 3-line card is ~270 px tall
TITLE_LINES = 3
TITLE_BG = (255, 255, 255, 255)
TITLE_FG = (0, 0, 0, 255)
TITLE_RADIUS = 14
TITLE_MAX_CHARS = 48       # OpusClip's titles run 36-45 characters

# ─── Badge position (top edge, px on the 1080x1920 vertical canvas) ──────────
# A 1-line badge is ~90 px tall, a 3-line title card ~270 px.
#   pad:    the 1080x608 content strip starts at y=656: the card goes at y=500,
#           or higher so that it ends PAD_GAP above the strip.
#   single: the face is scaled up 1.8x or more; its box top usually lands at
#           y≈380-700, so y=500 would cover it.
#   split:  the top speaker's face center is at y≈576 of the 960 px top half
#           (bbox top ≈387), so y=500 would cover it too.
# y=110 leaves a gap above the hair (y=180 touched it on real clips); y≈60 would
# sit under the phone status bar and the apps' top tabs.
CANVAS_H = 1920
BADGE_Y_PAD = 500
PAD_STRIP_TOP = 656
PAD_GAP = 20
BADGE_Y_FACE = 110
BADGE_Y_LANDSCAPE = 40


def truncate_label(label: str, max_chars: int = MAX_LABEL_CHARS) -> str:
    """Truncate label with ellipsis if too long for badge readability."""
    if len(label) <= max_chars:
        return label
    return label[:max_chars].rsplit(' ', 1)[0] + "..."


def wrap_lines(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
    """Greedy word wrap to lines no wider than max_w pixels."""
    lines: List[str] = []
    for word in text.split():
        if lines and font.getlength(f"{lines[-1]} {word}") <= max_w:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines


def create_badge_png(
    label: str,
    bg_color: Tuple[int, int, int, int] = TITLE_BG,
    output_path: str = "badge.png",
    font_size: int = TITLE_FONT_SIZE,
    padding_x: int = 34,
    padding_y: int = 26,
    max_width: int = 810,
) -> Tuple[int, int]:
    """
    Draw the title card PNG: text wrapped to at most TITLE_LINES centered lines
    in a rounded box (OpusClip: white card, black bold text). The font shrinks
    when the text needs more lines. Returns (width, height).
    """
    label = truncate_label(label)
    while True:
        font = load_text_font(font_size)
        lines = wrap_lines(label, font, max_width - 2 * padding_x)
        if len(lines) <= TITLE_LINES or font_size <= MIN_FONT_SIZE:
            break
        font_size = max(MIN_FONT_SIZE, int(font_size * 0.9))

    ascent, descent = font.getmetrics()
    line_h = round(font_size * 1.22)
    text_w = max(round(font.getlength(line)) for line in lines)
    w = min(max_width, text_w + 2 * padding_x)
    h = line_h * (len(lines) - 1) + ascent + descent + 2 * padding_y

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(0, 0), (w - 1, h - 1)], radius=TITLE_RADIUS, fill=bg_color)
    for n, line in enumerate(lines):
        x = (w - font.getlength(line)) / 2
        draw.text((x, padding_y + n * line_h), line, font=font, fill=TITLE_FG)

    img.save(output_path, format="PNG")
    return w, h


# ─── Hook title (LLM) ────────────────────────────────────────────────────────
TITLE_PROMPT = """You write the on-screen hook title for a short vertical video (TikTok, Reels, Shorts).

What is said in the clip ({language}):
{text}

Write one title in {language} that makes a scrolling viewer stop and watch: a short question or a bold claim about the clip's main point, true to what is said.
Rules: 4-9 words, at most {max_chars} characters, sentence case, no emoji, no hashtags, no quotation marks.
Reply as JSON: {{"title": "..."}}"""


def clean_title(title: str, max_chars: int = TITLE_MAX_CHARS) -> Optional[str]:
    """The LLM's title, tidied; None if it is unusable (then the badge_label is used)."""
    title = re.sub(r"[\"“”#*]", "", str(title or "")).strip().strip("'").strip()
    title = re.sub(r"\s+", " ", title)
    if not title or len(title) > max_chars + 12 or len(title.split()) < 2:
        return None
    return title[0].upper() + title[1:]


def hook_title(text: str, chat, language: str) -> Optional[str]:
    """One LLM call: what is said in the clip -> hook title, or None."""
    if not text.strip():
        return None
    msgs = [{"role": "user", "content": TITLE_PROMPT.format(
        language=language, text=text.strip()[:4000], max_chars=TITLE_MAX_CHARS)}]
    try:
        return clean_title(json.loads(chat(msgs)).get("title"))
    except Exception as e:
        print(f"  ⚠ Hook title failed ({type(e).__name__}), using the badge label")
        return None


def clip_text(clip_path: Path, meta: Dict) -> str:
    """What is said in the clip: the caption words (add_subtitles.py writes
    <clip>.words.json next to the clip), else the transcript text from the cutter."""
    words = clip_path.with_suffix(".words.json")
    if words.exists():
        try:
            return " ".join(w["text"] for w in json.loads(words.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return meta.get("text", "")


def get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """Use ffprobe to get video width and height."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 1920, 1080  # default fallback
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if streams:
        return streams[0].get("width", 1920), streams[0].get("height", 1080)
    return 1920, 1080


def load_layout(clip_path: str, layout_dir: Optional[str] = None) -> List[Dict]:
    """
    Read the auto_reframe.py sidecar <clip>.layout.json:
    [{start_time, end_time, mode}, ...]. Returns [] if there is none.
    """
    folder = layout_dir or os.path.dirname(clip_path)
    sidecar = os.path.join(folder, Path(clip_path).stem + ".layout.json")
    if not os.path.isfile(sidecar):
        return []
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ⚠ Could not read {sidecar}: {e}")
        return []
    return data if isinstance(data, list) else []


def badge_y_for(width: int, height: int, layout: List[Dict], badge_duration: float,
                badge_h: int = 0) -> int:
    """
    Top edge of the badge in pixels.

    Landscape: y=40. Vertical: if any scene on screen while the badge shows
    (t < badge_duration) is single or split, the badge goes above the top
    face (BADGE_Y_FACE); otherwise (pad, or no layout) into the blur area
    (BADGE_Y_PAD, raised so a badge_h tall card ends above the content strip).
    Scaled for vertical canvases other than 1920 px tall.
    """
    if height <= width:
        return BADGE_Y_LANDSCAPE
    shown = [
        scene for scene in layout
        if scene.get("start_time", 0.0) < badge_duration and scene.get("end_time", 0.0) > 0.0
    ]
    has_face = any(scene.get("mode") in ("single", "split") for scene in shown)
    scale = height / CANVAS_H
    if has_face:
        return round(BADGE_Y_FACE * scale)
    return max(round(BADGE_Y_LANDSCAPE * scale), round(min(BADGE_Y_PAD, PAD_STRIP_TOP - PAD_GAP - badge_h / scale) * scale))


def overlay_badge_ffmpeg(
    input_video: str,
    badge_png: str,
    output_video: str,
    badge_duration: float,
    x: str,
    y: int,
) -> bool:
    """Overlay badge PNG onto video using FFmpeg overlay filter. Instant appear/disappear."""
    filter_complex = (
        f"[0:v][1:v]overlay={x}:{y}:enable='between(t,0,{badge_duration})',setsar=1"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", badge_png,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "copy",
        output_video,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ FFmpeg error:\n{result.stderr[-2000:]}")
        return False
    return True


def load_metadata_from_clips_metadata(metadata_path: str) -> Dict[str, Dict]:
    """Load clips_metadata.json produced by video_cutter.py."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata_from_segments(segments_path: str) -> Dict[float, Dict]:
    """Load merged_segments.json and index by start_time for filename matching."""
    with open(segments_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    index = {}
    for seg in segments:
        start = seg.get("start_time", 0)
        if isinstance(start, str):
            start = float(start.rstrip("s"))
        index[start] = seg
    return index


def match_segment_by_filename(filename: str, segments_index: Dict[float, Dict]) -> Optional[Dict]:
    """Extract start_time from filename and find matching segment."""
    # Pattern: topic_<start>s_<dur>s_...mp4  OR  HH-MM-SS_...mp4  OR  clip_NNN_...mp4
    match = re.search(r'_(\d+)s_\d+s', filename)
    if match:
        start = float(match.group(1))
        seg = segments_index.get(start)
        if seg:
            return seg
        # Fuzzy: find closest within 2s
        for seg_start, seg in segments_index.items():
            if abs(seg_start - start) <= 2.0:
                return seg
    return None


def process_clip(
    clip_path: Path,
    output_dir: Path,
    badge_label: str,
    topic: str,
    badge_duration: float,
    verbose: bool,
    layout_dir: Optional[str] = None,
) -> bool:
    """Process a single clip: create badge, overlay, save."""
    output_path = output_dir / clip_path.name

    if not badge_label:
        badge_label = "Highlight"

    # Auto-detect orientation; on vertical clips the reframe layout decides y
    w, h = get_video_dimensions(str(clip_path))
    is_vertical = h > w
    layout = load_layout(str(clip_path), layout_dir) if is_vertical else []
    badge_x = "(W-w)/2"  # FFmpeg expression: center horizontally

    if verbose:
        print(f"  Video: {w}x{h} ({'vertical' if is_vertical else 'landscape'})")
        print(f"  Badge: '{badge_label}'")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        badge_png = tmp.name

    try:
        max_badge_w = int(w * MAX_BADGE_WIDTH_RATIO)
        bw, bh = create_badge_png(
            label=badge_label,
            output_path=badge_png,
            max_width=max_badge_w,
        )
        badge_y = badge_y_for(w, h, layout, badge_duration, bh)
        if is_vertical:
            modes = sorted({scene.get("mode", "?") for scene in layout})
            print(f"  Layout: {'/'.join(modes) if modes else 'no sidecar (pad)'} → card y={badge_y}, {bw}x{bh}")
        if verbose:
            print(f"  Badge PNG: {bw}x{bh}px")

        ok = overlay_badge_ffmpeg(
            input_video=str(clip_path),
            badge_png=badge_png,
            output_video=str(output_path),
            badge_duration=badge_duration,
            x=badge_x,
            y=badge_y,
        )
        return ok
    finally:
        Path(badge_png).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Add a hook title card to video clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With clips_metadata.json (preferred - has badge_label from LLM)
  python scripts/add_badge.py \\
      --input-dir clips_vertical/ \\
      --output-dir clips_vertical_badged/ \\
      --metadata clips_vertical/clips_metadata.json

  # Fallback using merged_segments.json (badge_label generated from reason)
  python scripts/add_badge.py \\
      --input-dir clips_vertical/ \\
      --output-dir clips_vertical_badged/ \\
      --segments data/analysis_results/merged_optimized/merged_segments.json

  # Adjust badge display duration
  python scripts/add_badge.py \\
      --input-dir clips_vertical/ \\
      --output-dir clips_vertical_badged/ \\
      --metadata clips_vertical/clips_metadata.json \\
      --duration 5
        """
    )

    parser.add_argument("--input-dir", required=True, help="Directory with input .mp4 clips")
    parser.add_argument("--output-dir", required=True, help="Directory for output clips with badges")
    parser.add_argument("--metadata", default=None,
                        help="Path to clips_metadata.json (from video_cutter.py)")
    parser.add_argument("--segments", default=None,
                        help="Path to merged_segments.json (fallback when no metadata)")
    parser.add_argument("--duration", type=float, default=4.5,
                        help="Seconds to show the title (default: 4.5, like OpusClip)")
    parser.add_argument("--title-model", default="gemma4:12b-it-q4_K_M",
                        help="Ollama model that writes the hook titles (default: gemma4:12b-it-q4_K_M)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--language", default="Indonesian", help="Language of the clips and titles")
    parser.add_argument("--no-titles", action="store_true",
                        help="No LLM: show the analysis badge_label instead of a hook title")
    parser.add_argument("--layout-dir", default=None,
                        help="Folder with <clip>.layout.json from auto_reframe.py "
                             "(default: next to each clip)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output per clip")

    args = parser.parse_args()

    if not args.metadata and not args.segments:
        print("❌ Provide --metadata or --segments")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather clip files
    clips = sorted(input_dir.glob("*.mp4"))
    if not clips:
        print(f"❌ No .mp4 files found in {input_dir}")
        sys.exit(1)

    # Load metadata
    clips_meta: Dict[str, Dict] = {}
    segments_index: Dict[float, Dict] = {}

    if args.metadata and Path(args.metadata).exists():
        clips_meta = load_metadata_from_clips_metadata(args.metadata)
        print(f"✓ Loaded clips_metadata.json ({len(clips_meta)} entries)")
    elif args.segments and Path(args.segments).exists():
        segments_index = load_metadata_from_segments(args.segments)
        print(f"✓ Loaded merged_segments.json ({len(segments_index)} segments)")
    else:
        print(f"❌ Metadata file not found: {args.metadata or args.segments}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("ADD HOOK TITLE")
    print(f"{'='*70}")
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Clips:      {len(clips)}")
    print(f"Duration:   {args.duration}s")
    print(f"{'='*70}\n")

    titles_path = output_dir / "titles.json"
    titles: Dict[str, str] = {}
    if titles_path.exists():
        try:
            titles = json.loads(titles_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            titles = {}
    chat = None
    if not args.no_titles:
        from fit_window import ollama_chat

        chat = ollama_chat(args.title_model, args.ollama_url)
        print(f"Hook titles: {args.title_model} (cache: {titles_path})\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, clip_path in enumerate(clips, 1):
        filename = clip_path.name
        print(f"[{i}/{len(clips)}] {filename}")

        # Resolve badge data
        badge_label = ""
        topic = "default"

        if filename in clips_meta:
            meta = clips_meta[filename]
            badge_label = meta.get("badge_label", "")
            topic = meta.get("topic", "default")
            if not badge_label:
                badge_label = generate_label_from_reason(meta.get("reason", ""))
                print(f"  ⚠ No badge_label, generated from reason: '{badge_label}'")
        elif segments_index:
            seg = match_segment_by_filename(filename, segments_index)
            if seg:
                badge_label = seg.get("badge_label", "")
                topic = seg.get("topic", "default")
                if not badge_label:
                    badge_label = generate_label_from_reason(seg.get("reason", ""))
                    print(f"  ⚠ No badge_label, generated from reason: '{badge_label}'")
            else:
                print(f"  ⚠ No matching segment found for {filename}, using defaults")
                badge_label = "Highlight"
        else:
            print(f"  ⚠ {filename} not in metadata, using defaults")
            badge_label = "Highlight"

        if chat and filename not in titles:
            title = hook_title(clip_text(clip_path, clips_meta.get(filename, {})), chat, args.language)
            if title:
                titles[filename] = title
                titles_path.write_text(json.dumps(titles, ensure_ascii=False, indent=1), encoding="utf-8")
        if chat and titles.get(filename):
            badge_label = titles[filename]

        print(f"  Title: {badge_label} | topic={topic}")

        ok = process_clip(
            clip_path=clip_path,
            output_dir=output_dir,
            badge_label=badge_label,
            topic=topic,
            badge_duration=args.duration,
            verbose=args.verbose,
            layout_dir=args.layout_dir,
        )

        if ok:
            out_size = (output_dir / filename).stat().st_size / (1024 * 1024)
            print(f"  ✅ Done ({out_size:.1f} MB)")
            success_count += 1
        else:
            print(f"  ❌ Failed")
            fail_count += 1
        print()

    print("="*70)
    print(f"✅ Success: {success_count}/{len(clips)}")
    if fail_count:
        print(f"❌ Failed:  {fail_count}/{len(clips)}")
    print(f"📁 Output:  {output_dir}")
    print("="*70)
    if fail_count:  # exit 0 here made run_pipeline report PIPELINE COMPLETE with no final clips
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    main()
