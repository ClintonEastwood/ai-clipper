import itertools

import pytest

from auto_reframe import (
    SINGLE_ASPECT,
    SPLIT_ASPECT,
    _video_filter_args,
    compute_crop,
)

FRAME_SIZES = [(1920, 1080), (1280, 720), (854, 480), (853, 480), (640, 480), (1080, 1920)]
ASPECTS = [SINGLE_ASPECT, SPLIT_ASPECT]


@pytest.mark.parametrize("width,height", FRAME_SIZES)
@pytest.mark.parametrize("aspect", ASPECTS)
def test_crop_always_inside_frame(width, height, aspect):
    xs = [0, 1, width * 0.25, width / 2, width - 1, width]
    ys = [0, 1, height * 0.3, height / 2, height - 1, height]
    for cx, cy, s in itertools.product(xs, ys, [0, 5, 40, 150, 600, 5000]):
        w, h, x, y = compute_crop({"x": cx, "y": cy, "s": s}, width, height, aspect)
        assert w >= 2 and h >= 2
        assert w % 2 == 0 and h % 2 == 0
        assert 0 <= x and x + w <= width, (cx, cy, s, w, x)
        assert 0 <= y and y + h <= height, (cx, cy, s, h, y)


@pytest.mark.parametrize("width,height", FRAME_SIZES)
@pytest.mark.parametrize("aspect", ASPECTS)
def test_crop_keeps_aspect_ratio(width, height, aspect):
    for s in [0, 40, 150, 600]:
        w, h, _, _ = compute_crop({"x": width / 2, "y": height / 2, "s": s}, width, height, aspect)
        # Only even-pixel rounding may move it
        assert w / h == pytest.approx(aspect, rel=0.03)


def test_1080p_split_crop_unchanged():
    # Same numbers as before the helper (up to rounding to even pixels)
    assert compute_crop({"x": 500, "y": 400, "s": 90}, 1920, 1080, SPLIT_ASPECT) == (514, 456, 243, 126)
    # Single crop at 1080p is height-limited: full height, 9:16
    assert compute_crop({"x": 700, "y": 400, "s": 120}, 1920, 1080, SINGLE_ASPECT) == (606, 1080, 397, 0)


def test_480p_split_matches_1080p_framing():
    # Same two-shot at 1080p and at 854x480: the crop should scale with the
    # frame, so the face fills the same share of the split-screen half.
    k = 480 / 1080
    face_1080 = {"x": 560, "y": 400, "s": 90}
    face_480 = {"x": 560 * k, "y": 400 * k, "s": 90 * k}

    w1, h1, x1, y1 = compute_crop(face_1080, 1920, 1080, SPLIT_ASPECT)
    w2, h2, x2, y2 = compute_crop(face_480, 854, 480, SPLIT_ASPECT)

    assert w2 == pytest.approx(w1 * k, abs=2)
    assert h2 == pytest.approx(h1 * k, abs=2)
    # Face share of the crop width is the same (~35%)
    assert (2 * face_480["s"]) / w2 == pytest.approx((2 * face_1080["s"]) / w1, rel=0.02)


def test_480p_split_is_tight_and_centered():
    # Before: the 400 px floor (tuned for 1080p) gave a 400x355 crop on
    # 854x480, the face filled ~20% of it and the box was pushed to y=0.
    face = {"x": 250, "y": 180, "s": 40}
    w, h, x, y = compute_crop(face, 854, 480, SPLIT_ASPECT)
    assert w < 300
    assert (2 * face["s"]) / w == pytest.approx(0.35, abs=0.02)
    # Face horizontally centered, with the intended 60% headroom
    assert (face["x"] - x) / w == pytest.approx(0.5, abs=0.02)
    assert (face["y"] - y) / h == pytest.approx(0.6, abs=0.02)


def test_crop_shifts_at_frame_edge():
    # Face near the left edge: box slides right instead of leaving the frame
    w, h, x, y = compute_crop({"x": 20, "y": 30, "s": 40}, 854, 480, SPLIT_ASPECT)
    assert x == 0 and y == 0


def test_missing_scale_uses_default_scaled_to_frame():
    w1080, *_ = compute_crop({"x": 960, "y": 540}, 1920, 1080, SPLIT_ASPECT)
    w480, *_ = compute_crop({"x": 427, "y": 240, "s": 0.0}, 854, 480, SPLIT_ASPECT)
    assert w480 == pytest.approx(w1080 * 480 / 1080, abs=2)


def test_filter_args_split_480p():
    coords = [{"x": 250, "y": 180, "s": 40}, {"x": 600, "y": 190, "s": 42}]
    args = _video_filter_args("split", coords, 854, 480)
    assert args[0] == "-filter_complex"
    assert "crop=228:202:136:58,scale=1080:960[top]" in args[1]
    assert args[2:] == ["-map", "[out]", "-map", "0:a"]


def test_filter_args_fallbacks_to_pad():
    # split needs two faces; anything else falls back to the blurred pad
    args = _video_filter_args("split", [{"x": 1, "y": 1, "s": 1}], 854, 480)
    assert "boxblur" in args[1]
    assert "boxblur" in _video_filter_args("pad", [], 854, 480)[1]
    assert _video_filter_args("single", [{"x": 427, "y": 240, "s": 60}], 854, 480)[0] == "-vf"
