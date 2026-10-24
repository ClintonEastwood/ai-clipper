"""Rank the merged clips best first: adds rank and rank_score to merged_segments.json, writes ranking.md.

rank_score is the clip's confidence (the LLM's scores). With --media it is blended with how much
laughter and loudness the clip has (AUDIO_WEIGHT, fixed before testing, never tuned on the test podcast).
"""
import argparse
import json
import os

from audio_cues import clip_audio_score

AUDIO_WEIGHT = 0.2


def _t(x):
    return float(str(x).rstrip("s"))


def rank(segments, cues=None):
    for s in segments:
        conf = float(s.get("confidence", 0))
        s["audio_score"] = clip_audio_score(_t(s["start_time"]), _t(s["end_time"]), cues) if cues else None
        s["rank_score"] = round(conf if cues is None else (1 - AUDIO_WEIGHT) * conf + AUDIO_WEIGHT * s["audio_score"], 3)
    for i, s in enumerate(sorted(segments, key=lambda s: -s["rank_score"]), 1):
        s["rank"] = i
    return segments


def ranking_table(segments):
    rows = ["| Rank | Score | Start | Length | Badge | Title |", "|---|---|---|---|---|---|"]
    for s in sorted(segments, key=lambda s: s["rank"]):
        start, end = _t(s["start_time"]), _t(s["end_time"])
        rows.append(f"| {s['rank']} | {s['rank_score']:.2f} | {int(start // 60)}:{int(start % 60):02d} ({start:.0f}s) | "
                    f"{end - start:.0f}s | {s.get('badge_label', '')} | {s.get('title', '')} |")
    return "\n".join(rows) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged", required=True, help="merged_segments.json (updated in place)")
    p.add_argument("--media", help="source video: blend in laughter + loudness (cached as audio_cues.json beside --merged)")
    a = p.parse_args()
    with open(a.merged, encoding="utf-8") as f:
        data = json.load(f)
    cues = None
    if a.media:
        from audio_cues import cached, decode_audio, laughter_and_loudness
        cues = cached(os.path.join(os.path.dirname(a.merged), "audio_cues.json"),
                      lambda: laughter_and_loudness(decode_audio(a.media)), a.media)
    rank(data["segments"], cues)
    with open(a.merged, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    out = os.path.join(os.path.dirname(a.merged), "ranking.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(ranking_table(data["segments"]))
    print(f"Ranked {len(data['segments'])} clips -> {out}")
    return 0


if __name__ == "__main__":
    main()
