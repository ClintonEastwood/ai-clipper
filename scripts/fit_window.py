"""Fit long story clips into a duration limit by letting the LLM pick a window of the clip's own transcript lines.
Trimming a clip's middle chops its setup and punchline; a run of whole lines that keeps both, with the edges
then moved into pauses, stays followable and ends clean."""
import glob
import json
import os

import requests

from analyze_content import transcript_lines
from audio_cues import snap_within

PROMPT = """Below is a short-video clip cut from an Indonesian podcast, as numbered transcript lines (about 10 s each, no punctuation) with their times in the clip. It is {dur} s long and must become {lo} to {hi} s.
Pick ONE continuous run of lines, first_line to last_line, that a viewer can follow on its own: its first line makes sense without anything said before it (no word pointing back to something the viewer did not hear, no answer to a question that was cut off), and it keeps the line that gives the clip its point (the punchline, answer, reveal or key reaction). Cut filler at the start or the end, never the point. By the times shown, the run must last {lo} to {hi} s (end of last_line minus start of first_line).

{shown}

Output ONLY JSON: {{"first_line": <n>, "last_line": <n>}}"""


def load_lines(transcripts_dir):
    """Every (text, start, end) line of every .txt file in the directory, de-duplicated and sorted."""
    lines = set()
    for path in glob.glob(os.path.join(transcripts_dir, "*.txt")):
        with open(path, encoding="utf-8") as f:
            lines.update(transcript_lines(f.read()))
    return sorted(lines, key=lambda l: (l[1], l[2]))


def ollama_chat(model, url):
    """chat(messages) -> reply text. num_ctx matches analyze_content: another size reloads the model."""
    def chat(messages):
        r = requests.post(f"{url}/api/chat", json={
            "model": model, "messages": messages, "stream": False, "format": "json", "think": False,
            "options": {"num_ctx": 8192, "temperature": 0}}, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]
    return chat


def fit_clip(lines, start, end, pauses, chat, min_len=30.0, max_len=60.0):
    """(new_start, new_end, text) for the window of the clip's lines the LLM picks, or None."""
    idx = [k for k, (_, s, e) in enumerate(lines) if start <= (s + e) / 2 <= end]
    if not idx:
        return None
    last = len(idx) - 1
    a_of = lambda n: start if n == 0 else lines[idx[n]][1]
    b_of = lambda n: end if n == last else lines[idx[n]][2]
    shown = "\n".join(f"{n}: [{a_of(n) - start:.0f}-{b_of(n) - start:.0f} s] {lines[idx[n]][0]}" for n in range(last + 1))
    msgs = [{"role": "user", "content": PROMPT.format(dur=f"{end - start:.0f}", lo=f"{min_len:.0f}",
                                                       hi=f"{max_len:.0f}", shown=shown)}]
    for _ in range(2):
        try:
            reply = chat(msgs)
            pick = json.loads(reply)
            i, j = int(pick["first_line"]), int(pick["last_line"])
        except Exception as ex:
            print(f"   Fit {start:.1f}-{end:.1f}s: {type(ex).__name__}")
            return None
        if 0 <= i <= j <= last and min_len <= round(b_of(j) - a_of(i), 2) <= max_len:
            s, e = snap_within(a_of(i), b_of(j), pauses, min_len, max_len)
            print(f"   Fit {start:.1f}-{end:.1f}s ({end - start:.0f} s): lines {i}-{j} of 0-{last} -> "
                  f"{s:.1f}-{e:.1f}s ({e - s:.0f} s)")
            return s, e, " ".join(lines[idx[n]][0] for n in range(i, j + 1))
        msg = (f"Lines {i}-{j} last {round(b_of(j) - a_of(i), 2):.0f} s" if 0 <= i <= j <= last
               else f"Lines must be 0 to {last} with first_line <= last_line")
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"{msg}; the run must last {min_len:.0f} to {max_len:.0f} s. Pick again."}]
    print(f"   Fit {start:.1f}-{end:.1f}s: no valid window after 2 picks")
    return None


def _secs(x):
    return float(str(x).rstrip("s"))


def fit_long(segments, lines, pauses, chat, min_len, max_len):
    """Segments with every clip over max_len fit into min_len..max_len; clips that cannot be fit are dropped."""
    out, over, fitted = [], 0, 0
    for seg in segments:
        s, e = _secs(seg["start_time"]), _secs(seg["end_time"])
        if round(e - s, 2) <= max_len:
            out.append(seg)
            continue
        over += 1
        r = fit_clip(lines, s, e, pauses, chat, min_len, max_len)
        if r:
            fitted += 1
            out.append({**seg, "start_time": r[0], "end_time": r[1], "duration": round(r[1] - r[0], 2),
                        "text": r[2], "fitted_from": [s, e]})
    best = {}
    for seg in out:
        key = (seg["start_time"], seg["end_time"])
        if key not in best or seg.get("confidence", 0) > best[key].get("confidence", 0):
            best[key] = seg
    out = [seg for seg in out if best[(seg["start_time"], seg["end_time"])] is seg]
    if over:
        print(f"   Fitted {fitted}/{over} clips over {max_len:.0f} s into {min_len:.0f}-{max_len:.0f} s windows "
              f"({over - fitted} dropped)")
    return out


WINDOW = 300.0  # s: the show's opening is in the first 5 min, so only clips starting there are checked
OPENING_PROMPT = """Below is a clip cut from an Indonesian podcast episode, as transcript lines (speech recognition, no punctuation). It starts {start} into an episode that is {episode} long.
Is this clip the show's opening: the host greeting the audience or the guest, welcoming them, or introducing the guest or the episode's topic, rather than a story, argument or exchange that stands on its own?

{text}

Output ONLY JSON: {{"opening": true or false, "reason": "<one sentence>"}}"""


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def drop_openings(segments, lines, chat, window=WINDOW):
    """Segments without the show's opening clips. An unusable reply keeps the clip: a filter must not throw
    away content because the LLM failed."""
    if not lines:
        return segments
    episode = lines[-1][2]
    out, checked, dropped = [], 0, 0
    for seg in segments:
        s, e = _secs(seg["start_time"]), _secs(seg["end_time"])
        if s >= window:
            out.append(seg)
            continue
        checked += 1
        text = "\n".join(t for t, a, b in lines if s <= (a + b) / 2 <= e)
        if not text:  # the model must never drop a clip it cannot see
            out.append(seg)
            continue
        msg = [{"role": "user", "content": OPENING_PROMPT.format(start=mmss(s), episode=mmss(episode), text=text)}]
        try:
            pick = json.loads(chat(msg))
            opening = pick["opening"]
            if not isinstance(opening, bool):
                raise ValueError("opening is not a bool")
        except Exception as ex:
            print(f"   Opening check {s:.1f}-{e:.1f}s: unusable reply ({type(ex).__name__}), clip kept")
            out.append(seg)
            continue
        if opening:
            dropped += 1
            print(f"   Opening check {s:.1f}-{e:.1f}s: dropped, {pick.get('reason', '')}")
        else:
            out.append(seg)
    print(f"Opening check: {checked} clips in the first {window / 60:.0f} min, {dropped} dropped")
    return out
