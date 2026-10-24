"""Audio cues for cutting and ranking: pauses (ffmpeg silencedetect), laughter and loudness."""
import json
import os
import re
import subprocess


def _stamp(src):
    return f"{os.path.abspath(src)} {os.path.getsize(src)} {os.path.getmtime(src):.0f}" if os.path.exists(src) else ""


def cached(path, make, src=None):
    """Load JSON from path, or run make() and save it there.
    With src: a cache made from another file is redone (same output dir, new video)."""
    side = path + ".src"
    fresh = os.path.exists(path)
    if fresh and src and os.path.exists(side):  # a cache with no .src stamp (copied in by hand) is trusted
        with open(side, encoding="utf-8") as f:
            fresh = f.read() == _stamp(src)
    if fresh:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    data = make()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    if src:
        with open(side, "w", encoding="utf-8") as f:
            f.write(_stamp(src))
    return data


def parse_silencedetect(log):
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    return [(max(0.0, s), e) for s, e in zip(starts, ends)]


def detect_pauses(media_path, noise_db=-35, min_len=0.25):
    """[(start, end)] of every stretch quieter than noise_db for at least min_len s."""
    # one fixed threshold; make it relative to the episode's loudness if other podcasts misfire
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", media_path, "-vn", "-af",
                        f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    if r.returncode and "does not contain any stream" not in r.stderr:  # that one = no audio stream: no pauses
        last = (r.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"ffmpeg could not read {media_path}: {last}")
    return parse_silencedetect(r.stderr)


def snap_to_pauses(start, end, pauses, max_shift=3.0):
    """Move a clip's edges outward into the nearest pause (at most max_shift s), so no cut lands mid-word.
    The start moves to 0.15 s before speech resumes; the end to 0.15 s after it stops."""
    if not any(a <= start <= b for a, b in pauses):
        before = [(a, b) for a, b in pauses if start - max_shift <= b <= start]
        if before:
            a, b = max(before, key=lambda p: p[1])
            start = max(a, b - 0.15)
    if not any(a <= end <= b for a, b in pauses):
        after = [(a, b) for a, b in pauses if end <= a <= end + max_shift]
        if after:
            a, b = min(after, key=lambda p: p[0])
            end = min(b, a + 0.15)
    return round(start, 2), round(end, 2)


def snap_within(start, end, pauses, min_len, max_len, max_shift=3.0):
    """Snap edges into pauses without leaving min_len..max_len: outward first, else pull an edge inward.
    Inward edges also sit 0.15 s from speech (the cut-off word is dropped). No fit: the edges unchanged."""
    out_s, out_e = snap_to_pauses(start, end, pauses, max_shift)
    in_s, in_e = start, end
    if not any(a <= start <= b for a, b in pauses):
        after = [(a, b) for a, b in pauses if start <= a <= start + max_shift]
        if after:
            a, b = min(after, key=lambda p: p[0])
            in_s = max(a, b - 0.15)
    if not any(a <= end <= b for a, b in pauses):
        before = [(a, b) for a, b in pauses if end - max_shift <= a and b <= end]
        if before:
            a, b = max(before, key=lambda p: p[0])
            in_e = min(b, a + 0.15)
    for s, e in [(out_s, out_e), (out_s, in_e), (in_s, out_e), (in_s, in_e)]:
        if min_len <= round(e - s, 2) <= max_len:
            return round(s, 2), round(e, 2)
    return start, end


def snap_all(segments, media_path, cache_dir, max_len):
    """Snap every clip into pauses, unless that makes it longer than max_len (the cutter drops those)."""
    os.makedirs(cache_dir, exist_ok=True)
    pauses = cached(os.path.join(cache_dir, "pauses.json"), lambda: detect_pauses(media_path), media_path)
    moved = 0
    for seg in segments:
        s, e = snap_to_pauses(float(seg["start_time"]), float(seg["end_time"]), pauses)
        if e - s <= max_len and (s, e) != (seg["start_time"], seg["end_time"]):
            seg["start_time"], seg["end_time"], seg["duration"] = s, e, round(e - s, 2)
            moved += 1
    print(f"   Snapped {moved}/{len(segments)} clips into pauses ({len(pauses)} pauses found)")
    return segments


LAUGH_LABELS = {"Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"}
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"  # AudioSet: 527 sound labels, 6 of them laughter


def decode_audio(media_path, sr=16000):
    """Mono float32 samples via ffmpeg (no extra audio library); empty when there is no audio stream."""
    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", media_path, "-vn", "-ac", "1", "-ar", str(sr),
                        "-f", "s16le", "-"], capture_output=True)
    if r.returncode:
        print("   No audio decoded: audio score 0 for every clip")
    raw = r.stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def ast_laughter(sr):
    """classify(samples) -> 0-1 chance of laughter, from the AST AudioSet model on CPU."""
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
    model = ASTForAudioClassification.from_pretrained(AST_MODEL).eval()
    ids = [int(i) for i, label in model.config.id2label.items() if label in LAUGH_LABELS]

    def classify(x):
        with torch.no_grad():
            p = torch.sigmoid(model(**fe(x, sampling_rate=sr, return_tensors="pt")).logits[0])  # multi-label
        return float(p[ids].max())
    return classify


def laughter_and_loudness(samples, sr=16000, classify=None, window=10.0, hop=5.0):
    """Per second: laughter chance (max over the windows covering it) and loudness in dB."""
    import numpy as np
    n_sec = int(len(samples) / sr)
    if not n_sec:  # no audio, or under 1 s: nothing to classify
        return {"laugh": [], "db": []}
    db = [float(20 * np.log10(np.sqrt(np.mean(samples[i * sr:(i + 1) * sr] ** 2)) + 1e-9)) for i in range(n_sec)]
    classify = classify or ast_laughter(sr)
    laugh = [0.0] * n_sec
    w, h = int(window * sr), int(hop * sr)
    for i in range(0, max(1, len(samples) - w + 1), h):
        p = classify(samples[i:i + w])
        for s in range(i // sr, min(n_sec, (i + w) // sr)):
            laugh[s] = max(laugh[s], p)
    return {"laugh": laugh, "db": db}


def clip_audio_score(start, end, cues):
    """0-1: where the clip's laughter peak and loudness (90th percentile) rank among all seconds of the episode."""
    import numpy as np
    laugh, db = np.asarray(cues["laugh"], dtype=float), np.asarray(cues["db"], dtype=float)
    if not len(laugh):
        return 0.0
    s = min(int(start), len(laugh) - 1)
    e = min(max(s + 1, int(end)), len(laugh))
    pct = lambda arr, v: float((arr <= v).mean()) if np.ptp(arr) else 0.0  # all equal (silent video): no cue
    return round((pct(laugh, laugh[s:e].max()) + pct(db, np.percentile(db[s:e], 90))) / 2, 3)


if __name__ == "__main__":  # python scripts/audio_cues.py <media> <audio_cues.json>
    import sys
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")       # CPU on purpose ("" unsets it on Windows)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # progress bars are non-ASCII: conda run crashes
    cached(sys.argv[2], lambda: laughter_and_loudness(decode_audio(sys.argv[1])), sys.argv[1])
    print("wrote", sys.argv[2])
