#!/usr/bin/env python3
"""
add_subtitles.py - word-highlight captions

Transcribes each vertical clip with Qwen3-ASR + Qwen3-ForcedAligner (word
timestamps), groups words into captions of up to 3 words, and burns them in
with FFmpeg (libass): a heavy slanted comic font (Luckiest Guy, bundled in
assets/fonts), thick black outline, white text.
  --caption-style karaoke (default): the word being spoken is yellow, and
      each caption pops in.
  --caption-style normal: plain white subtitles, no highlight.

Caption position follows the reframe layout when a `<clip>.layout.json`
sidecar exists (written by auto_reframe.py):
  - split  -> on the seam between the two speakers (y=960)
  - single / pad -> lower third, below the face

Usage:
    python scripts/add_subtitles.py \\
        --input-dir clips_vertical/ \\
        --output-dir clips_subtitled/ \\
        --language Indonesian
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# 1080x1920 canvas positions (caption center)
Y_SEAM = 960      # split screen: between top and bottom speaker
Y_LOWER = 1560    # single / pad: about 81% down, below the face

FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
SLANT = -0.18     # \fax shear: Luckiest Guy is upright; the captions lean right

CAPTION_STYLES = ("karaoke", "normal")
# Karaoke: a caption appears a little small and is full size 0.1 s later
POP_IN = r"\fscx85\fscy85\t(0,100,\fscx100\fscy100)"
WHITE = "&HFFFFFF&"

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,1,9,3,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(t):
    """Seconds -> ASS timestamp H:MM:SS.cc"""
    cs = int(round(max(t, 0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_color(hex_rgb):
    """'FFD500' (RRGGBB) -> '&H00D5FF&' (ASS uses BBGGRR)"""
    r, g, b = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    return f"&H{b}{g}{r}&"


def clean_word(text):
    # Braces and backslashes would be read as ASS override tags
    return text.replace("{", "").replace("}", "").replace("\\", "").strip()


def load_layout(clip_path):
    """Read the auto_reframe.py sidecar if present: [{start_time, end_time, mode}, ...]"""
    sidecar = os.path.splitext(clip_path)[0] + ".layout.json"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            return json.load(f)
    return []


def caption_y(t, layout):
    for scene in layout:
        if scene["start_time"] <= t < scene["end_time"]:
            return Y_SEAM if scene["mode"] == "split" else Y_LOWER
    return Y_LOWER


def group_words(words, max_words=3, max_chars=18, max_gap=0.5):
    """Split the word list into short captions (break on count, length, or pause).
    18 chars of Luckiest Guy 108 is ~920 px; wider captions are shrunk by fit_size()."""
    groups, current = [], []
    for w in words:
        if current:
            chars = sum(len(x["text"]) + 1 for x in current) + len(w["text"])
            if (len(current) >= max_words or chars > max_chars
                    or w["start"] - current[-1]["end"] > max_gap):
                groups.append(current)
                current = []
        current.append(w)
    if current:
        groups.append(current)
    return groups


MAX_TEXT_W = 960  # px between the 60 px side margins
FONT_FILES = {"Luckiest Guy": "LuckiestGuy-Regular.ttf"}


def fit_size(text, font, size):
    """Font size that keeps the caption inside MAX_TEXT_W (a 19-letter word like
    KETIDAKBERIMANANNYA is 1220 px at 108). Measured with the bundled font file;
    other fonts are estimated at 0.62 em per character."""
    try:
        from PIL import ImageFont

        width = ImageFont.truetype(str(FONTS_DIR / FONT_FILES[font]), size).getlength(text)
    except (KeyError, OSError, ImportError):
        width = 0.62 * size * len(text)
    width += 2 * 9 + 0.18 * size  # outline both sides + the \fax slant
    return size if width <= MAX_TEXT_W else int(size * MAX_TEXT_W / width)


def colored(texts, yellow, hl):
    """Caption text with word number `yellow` in the highlight color (None: all white)."""
    return " ".join(f"{{\\c{hl}}}{t}{{\\c{WHITE}}}" if k == yellow else t for k, t in enumerate(texts))


def build_ass(words, layout, font, size, highlight, uppercase, style="karaoke"):
    lines = [ASS_HEADER.format(font=font, size=size)]
    hl = ass_color(highlight)
    groups = group_words(words)
    for gi, group in enumerate(groups):
        # Keep the caption up a little after it ends, but never past the next one
        start = group[0]["start"]
        end = group[-1]["end"] + 0.3
        if gi + 1 < len(groups):
            end = min(end, groups[gi + 1][0]["start"])
        end = max(end, start + 0.05)

        texts = [clean_word(w["text"]) for w in group]
        texts = [t.upper() for t in texts] if uppercase else texts
        tags = f"\\pos(540,{caption_y(start, layout)})\\fax{SLANT}"
        fs = fit_size(" ".join(texts), font, size)
        if fs != size:
            tags += f"\\fs{fs}"
        if style == "karaoke":
            # One event per word: the whole caption stays up, the spoken word is yellow;
            # it pops in when it appears, not again on every word
            cuts = [start] + [w["start"] for w in group[1:]] + [end]
            events = [(min(cuts[k], end - 0.01), max(cuts[k + 1], cuts[k] + 0.05), k, POP_IN if k == 0 else "")
                      for k in range(len(group))]
        else:  # normal: plain white subtitles
            events = [(start, end, None, "")]
        for a, b, yellow, pop in events:
            lines.append(
                f"Dialogue: 0,{ass_time(a)},{ass_time(min(b, end))},Caption,,0,0,0,,"
                f"{{{tags}{pop}}}{colored(texts, yellow, hl)}"
            )
    return "\n".join(lines) + "\n"


def transcribe_words(model, clip_path, language, tmp_dir):
    """Extract 16 kHz mono audio and return [{text, start, end}, ...]"""
    wav = os.path.join(tmp_dir, "audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", clip_path, "-ac", "1", "-ar", "16000", wav],
        check=True,
    )
    results = model.transcribe(audio=wav, language=language, return_time_stamps=True)
    if not results or results[0].time_stamps is None:
        return []
    words = []
    for item in results[0].time_stamps.items:
        text = clean_word(item.text)
        if text:
            words.append({"text": text, "start": float(item.start_time), "end": float(item.end_time)})
    return words


def burn_subtitles(clip_path, ass_path, output_path):
    ass_dir, ass_name = os.path.split(os.path.abspath(ass_path))
    # The bundled fonts; libass still finds system fonts. Escape the drive-letter colon.
    fonts = FONTS_DIR.as_posix().replace(":", "\\:")
    vf = f"ass={ass_name}:fontsdir='{fonts}'"
    # Run from the .ass folder so the filter path has no drive-letter colon to escape
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", os.path.abspath(clip_path), "-vf", vf,
         "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-c:a", "copy",
         "-movflags", "+faststart", os.path.abspath(output_path)],
        check=True, cwd=ass_dir,
    )


def load_asr_model():
    import torch
    from qwen_asr import Qwen3ASRModel

    print("Loading Qwen3-ASR model...")
    return Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,
        max_new_tokens=2048,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    )


def main():
    parser = argparse.ArgumentParser(description="Burn word-highlight captions into vertical clips")
    parser.add_argument("--input-dir", required=True, help="Directory with vertical .mp4 clips")
    parser.add_argument("--output-dir", required=True, help="Directory for captioned clips")
    parser.add_argument("--language", default="Indonesian", help="ASR language (default: Indonesian)")
    parser.add_argument("--font", default="Luckiest Guy", help="Caption font name (bundled in assets/fonts)")
    parser.add_argument("--font-size", type=int, default=108)
    parser.add_argument("--caption-style", choices=CAPTION_STYLES, default="karaoke",
                        help="karaoke: the spoken word is yellow (default); normal: plain white subtitles")
    parser.add_argument("--highlight", default="FFE11A", help="Karaoke color of the spoken word, RRGGBB")
    parser.add_argument("--no-uppercase", action="store_true", help="Keep original casing")
    parser.add_argument("--retranscribe", action="store_true", help="Ignore cached <clip>.words.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    clips = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    if not clips:
        print(f"No .mp4 files found in {args.input_dir}")
        sys.exit(1)
    print(f"Found {len(clips)} clips to caption.")

    model = None
    failed = []
    for clip in clips:
        name = Path(clip).stem
        output_path = os.path.join(args.output_dir, f"{name}.mp4")
        words_path = os.path.join(args.output_dir, f"{name}.words.json")
        print(f"\n{'=' * 50}\nProcessing {name}.mp4\n{'=' * 50}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            if os.path.exists(words_path) and not args.retranscribe:
                with open(words_path, encoding="utf-8") as f:
                    words = json.load(f)
                print(f"  Using cached words ({len(words)})")
            else:
                if model is None:
                    model = load_asr_model()
                words = transcribe_words(model, clip, args.language, tmp_dir)
                with open(words_path, "w", encoding="utf-8") as f:
                    json.dump(words, f, ensure_ascii=False, indent=1)
                print(f"  Transcribed {len(words)} words")

            if not words:
                print("  No speech found, skipping.")
                continue

            layout = load_layout(clip)
            print(f"  Layout scenes: {len(layout)}" if layout else "  No layout sidecar, captions at lower third")

            ass_path = os.path.join(tmp_dir, "captions.ass")
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(build_ass(words, layout, args.font, args.font_size,
                                  args.highlight, not args.no_uppercase, args.caption_style))
            try:
                burn_subtitles(clip, ass_path, output_path)
                if layout:  # add_badge.py reads it next to the clip
                    shutil.copy(os.path.splitext(clip)[0] + ".layout.json", args.output_dir)
                print(f"✓ Captioned {name}.mp4")
            except subprocess.CalledProcessError as e:
                print(f"✗ FFmpeg failed for {name}.mp4: {e}")
                failed.append(f"{name}.mp4")

    if failed:
        print(f"\n✗ {len(failed)}/{len(clips)} clips failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    main()
