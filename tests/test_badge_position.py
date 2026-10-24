import json

import pytest

from add_badge import (
    BADGE_Y_FACE,
    BADGE_Y_PAD,
    badge_y_for,
    create_badge_png,
    load_layout,
)
from auto_reframe import SINGLE_ASPECT, SPLIT_ASPECT, compute_crop


def scene(start, end, mode):
    return {"start_time": start, "end_time": end, "mode": mode}


def test_landscape_unchanged():
    assert badge_y_for(1920, 1080, [], 4.0) == 40
    assert badge_y_for(1920, 1080, [scene(0, 10, "split")], 4.0) == 40


def test_vertical_without_layout_keeps_pad_position():
    # video_cutter.py pad/blur clips have no sidecar
    assert badge_y_for(1080, 1920, [], 4.0) == BADGE_Y_PAD == 500


@pytest.mark.parametrize("mode,expected", [
    ("pad", BADGE_Y_PAD),
    ("single", BADGE_Y_FACE),
    ("split", BADGE_Y_FACE),
])
def test_vertical_by_mode(mode, expected):
    assert badge_y_for(1080, 1920, [scene(0, 30, mode)], 4.0) == expected


def test_only_scenes_shown_with_badge_count():
    # Split starts after the badge is gone: the pad position is fine
    layout = [scene(0, 5, "pad"), scene(5, 30, "split")]
    assert badge_y_for(1080, 1920, layout, 4.0) == BADGE_Y_PAD
    # Split starts while the badge is still on screen: move it up
    layout = [scene(0, 2, "pad"), scene(2, 30, "split")]
    assert badge_y_for(1080, 1920, layout, 4.0) == BADGE_Y_FACE


def test_scales_with_canvas_height():
    assert badge_y_for(720, 1280, [scene(0, 30, "split")], 4.0) == round(BADGE_Y_FACE * 1280 / 1920)
    assert badge_y_for(720, 1280, [], 4.0) == round(BADGE_Y_PAD * 1280 / 1920)


# --- Badge vs. where the top face lands after auto_reframe ------------------

def top_face_box_on_canvas(face, width, height, aspect, out_h):
    """Face (y_top, y_bottom) in the output canvas for the top/only crop."""
    cw, ch, cx, cy = compute_crop(face, width, height, aspect)
    k = out_h / ch
    center = (face["y"] - cy) * k
    return center - face["s"] * k, center + face["s"] * k


def badge_height(tmp_path, label="Highlight"):
    _, h = create_badge_png(label, (50, 50, 50, 210), str(tmp_path / "b.png"), max_width=918)
    return h


@pytest.mark.parametrize("width,height", [(1920, 1080), (854, 480)])
def test_split_badge_clear_of_top_face(tmp_path, width, height):
    k = height / 1080
    face = {"x": 500 * k, "y": 400 * k, "s": 90 * k}  # typical podcast two-shot
    top, _ = top_face_box_on_canvas(face, width, height, SPLIT_ASPECT, 960)
    y = badge_y_for(1080, 1920, [scene(0, 30, "split")], 4.0)
    assert y + badge_height(tmp_path) < top
    # The old fixed y=500 overlapped the top speaker's face
    assert 500 + badge_height(tmp_path) > top


@pytest.mark.parametrize("width,height", [(1920, 1080), (854, 480)])
def test_single_badge_clear_of_face(tmp_path, width, height):
    k = height / 1080
    face = {"x": 700 * k, "y": 400 * k, "s": 120 * k}
    top, _ = top_face_box_on_canvas(face, width, height, SINGLE_ASPECT, 1920)
    y = badge_y_for(1080, 1920, [scene(0, 30, "single")], 4.0)
    assert y + badge_height(tmp_path) < top
    assert 500 + badge_height(tmp_path) > top


# --- Sidecar lookup ---------------------------------------------------------

def test_load_layout_next_to_clip(tmp_path):
    clip = tmp_path / "clip_001.mp4"
    layout = [scene(0, 3.5, "single"), scene(3.5, 9, "split")]
    (tmp_path / "clip_001.layout.json").write_text(json.dumps(layout))
    assert load_layout(str(clip)) == layout


def test_load_layout_from_layout_dir(tmp_path):
    # e.g. badging add_subtitles.py output, sidecars still in clips_vertical/
    reframed, subtitled = tmp_path / "clips_vertical", tmp_path / "clips_subtitled"
    reframed.mkdir()
    subtitled.mkdir()
    layout = [scene(0, 9, "split")]
    (reframed / "clip_001.layout.json").write_text(json.dumps(layout))
    clip = subtitled / "clip_001.mp4"
    assert load_layout(str(clip)) == []
    assert load_layout(str(clip), str(reframed)) == layout


def test_load_layout_missing_or_bad(tmp_path):
    clip = tmp_path / "clip.mp4"
    assert load_layout(str(clip)) == []
    (tmp_path / "clip.layout.json").write_text("{not json")
    assert load_layout(str(clip)) == []
    (tmp_path / "clip.layout.json").write_text('{"mode": "split"}')
    assert load_layout(str(clip)) == []


# --- OpusClip-style hook title card ------------------------------------------

from add_badge import PAD_STRIP_TOP, clean_title, clip_text, hook_title  # noqa: E402

HOOK = "Kereta telat sejam: siapa sebenarnya salah?"  # 43 chars


def test_title_wraps_to_at_most_three_lines(tmp_path):
    from PIL import Image

    w, h = create_badge_png(HOOK, output_path=str(tmp_path / "t.png"), max_width=810)
    assert w <= 810 and 120 < h < 300  # 2-3 lines, like OpusClip's ~270 px card
    corner = Image.open(tmp_path / "t.png").getpixel((w // 2, 3))
    assert corner[:3] == (255, 255, 255)


@pytest.mark.parametrize("mode", ["single", "split"])
def test_two_line_title_clear_of_face(tmp_path, mode):
    # The title card is taller than the old 1-line pill: it must still end above the face
    face = {"x": 700, "y": 400, "s": 120} if mode == "single" else {"x": 500, "y": 400, "s": 90}
    aspect, out_h = (SINGLE_ASPECT, 1920) if mode == "single" else (SPLIT_ASPECT, 960)
    top, _ = top_face_box_on_canvas(face, 1920, 1080, aspect, out_h)
    _, h = create_badge_png("Apakah kerja dari rumah bikin malas?", output_path=str(tmp_path / "t.png"))
    assert badge_y_for(1080, 1920, [scene(0, 30, mode)], 4.5, h) + h < top


def test_pad_card_ends_above_the_content_strip(tmp_path):
    _, h = create_badge_png(HOOK + " " + HOOK, output_path=str(tmp_path / "t.png"))
    y = badge_y_for(1080, 1920, [], 4.5, h)
    assert y + h < PAD_STRIP_TOP and y >= 40


def test_clean_title():
    assert clean_title('"apakah kerja dari rumah enak?"') == "Apakah kerja dari rumah enak?"
    assert clean_title("#Kopi #Pagi *viral*") == "Kopi Pagi viral"
    assert clean_title("") is None and clean_title("Kopi") is None
    assert clean_title("kata " * 30) is None


def test_hook_title_falls_back_on_bad_reply():
    assert hook_title("teks", lambda m: '{"title": "Kerja dari rumah, masihkah produktif?"}', "Indonesian") == \
        "Kerja dari rumah, masihkah produktif?"
    assert hook_title("teks", lambda m: "not json", "Indonesian") is None
    assert hook_title("   ", lambda m: '{"title": "x y"}', "Indonesian") is None


def test_clip_text_prefers_caption_words(tmp_path):
    clip = tmp_path / "a.mp4"
    assert clip_text(clip, {"text": "dari cutter"}) == "dari cutter"
    (tmp_path / "a.words.json").write_text(json.dumps([{"text": "kalau"}, {"text": "hujan"}]))
    assert clip_text(clip, {"text": "dari cutter"}) == "kalau hujan"
