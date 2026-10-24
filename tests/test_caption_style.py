import re

import add_subtitles
from add_subtitles import build_ass, group_words

YELLOW = "&H1AE1FF&"


def words(*spec):
    """("kata", start) pairs, 0.3 s per word."""
    return [{"text": t, "start": s, "end": s + 0.3} for t, s in spec]


def dialogues(ass):
    """[(start, end, text with tags)] of the Dialogue lines."""
    out = []
    for line in ass.splitlines():
        if line.startswith("Dialogue"):
            parts = line.split(",", 9)
            out.append((parts[1], parts[2], parts[9]))
    return out


def yellow_words(text):
    return re.findall(r"\\c" + re.escape(YELLOW) + r"\}([^{ ]+)", text)


def test_captions_are_up_to_three_words():
    ws = words(("udah", 0.0), ("dimasak", 0.3), ("tapi", 0.6), ("kesempatannya", 0.9), ("ya", 1.2))
    assert [[w["text"] for w in g] for g in group_words(ws)] == [
        ["udah", "dimasak", "tapi"], ["kesempatannya", "ya"]]


def test_karaoke_spoken_word_turns_yellow_left_to_right():
    # Three words on screen; each one is yellow while it is said, then the next
    ass = build_ass(words(("udah", 0.0), ("dimasak", 0.4), ("tapi", 0.8)), [], "Luckiest Guy", 108, "FFE11A", True)
    events = dialogues(ass)
    assert [yellow_words(t) for _, _, t in events] == [["UDAH"], ["DIMASAK"], ["TAPI"]]
    assert all("UDAH" in t and "DIMASAK" in t and "TAPI" in t for _, _, t in events)  # all 3 stay up
    assert [(a, b) for a, b, _ in events] == [("0:00:00.00", "0:00:00.40"), ("0:00:00.40", "0:00:00.80"),
                                              ("0:00:00.80", "0:00:01.40")]
    # Pops in once when the caption appears, not again on every word
    assert add_subtitles.POP_IN in events[0][2] and add_subtitles.POP_IN not in events[1][2]


def test_karaoke_starts_a_new_caption_after_three_words():
    ws = words(*[(t, i * 0.4) for i, t in enumerate("satu dua tiga empat lima".split())])
    texts = [t for _, _, t in dialogues(build_ass(ws, [], "F", 108, "FFE11A", True))]
    assert "EMPAT" not in texts[2] and "EMPAT" in texts[3] and "SATU" not in texts[3]


def test_normal_is_plain_white_subtitles():
    ws = words(*[(t, i * 0.4) for i, t in enumerate("kalau hujan dan macet tidak ada".split())])
    events = dialogues(build_ass(ws, [], "Luckiest Guy", 108, "FFE11A", True, style="normal"))
    assert [t.split("}")[-1] for _, _, t in events] == ["KALAU HUJAN DAN", "MACET TIDAK ADA"]
    assert all(YELLOW not in t and add_subtitles.POP_IN not in t for _, _, t in events)


def test_ass_slanted_lower_third_bundled_font():
    ass = build_ass(words(("kalau", 0.0), ("hujan", 0.3)), [], "Luckiest Guy", 108, "FFE11A", True, style="normal")
    (_, _, text), = dialogues(ass)
    assert text == r"{\pos(540,1560)\fax-0.18}KALAU HUJAN"
    assert "Luckiest Guy,108" in ass


def test_long_word_is_shrunk_to_fit_the_frame():
    # KETIDAKBERDAYAANNYA (19 letters) is 1220 px at size 108: wider than the 1080 px frame
    (_, _, text), = dialogues(build_ass(words(("ketidakberdayaannya", 0.0)), [], "Luckiest Guy", 108, "FFE11A", True))
    size = int(re.search(r"\\fs(\d+)", text).group(1))
    assert 70 <= size < 108
    (_, _, text), = dialogues(build_ass(words(("hujan", 0.0)), [], "Luckiest Guy", 108, "FFE11A", True))
    assert not re.search(r"\\fs\d", text)  # short captions keep the full size
