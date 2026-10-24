import os
import json
import pickle
import numpy as np
import argparse

def get_track_data(track_item):
    """
    Safely extract 'frame' and 'bbox'/'proc_track' from a TalkNet track item.
    Returns: (frames, center_xs, center_ys, scales)
    scales = proc_track['s'] = half-size of face bounding box in pixels.
    Larger value = bigger face = closer to camera = more likely a real person.
    """
    if isinstance(track_item, dict) and 'track' in track_item and 'proc_track' in track_item:
        frames    = track_item['track'].get('frame', [])
        center_xs = track_item['proc_track'].get('x', [])
        center_ys = track_item['proc_track'].get('y', [])
        scales    = track_item['proc_track'].get('s', [])
        return frames, center_xs, center_ys, scales

    # Fallback for different TalkNet structures
    frames = []
    if isinstance(track_item, dict):
        for k in ('frame', 'track', 'frames'):
            if k in track_item:
                val = track_item[k]
                if isinstance(val, dict) and 'frame' in val:
                    frames = val['frame']
                else:
                    frames = val
                break
    return frames, [], [], []


# A face that never speaks is kept only if it moves. Props (a toy figure, an ad
# poster) measured 0.001-0.003 on the 1080p and 480p test clips, listeners
# 0.089-0.224. Movement is measured in face sizes, not pixels, so the same
# threshold works on 1080p and 480p: the std of the face center and size,
# divided by the mean face half-size. 0.01 = 1% of the half-size. Each scene
# logs sil_motion, so the threshold can be tuned on real clips.
MIN_MOTION = 0.01


def track_motion(xs, ys, ss):
    """
    Scale-free movement of a face track (x, y center and s half-size per frame).
    Returns 0.0 for a still detection or when sizes are unknown.
    """
    if len(xs) < 2 or len(ss) == 0:
        return 0.0
    mean_s = float(np.mean(ss))
    if mean_s <= 0:
        return 0.0
    spread = np.var(xs) + np.var(ys) + np.var(ss)
    return float(np.sqrt(spread) / mean_s)


def looks_human(track_stats):
    """A face is a person if it speaks at least once or moves (props sit still)."""
    spoke = track_stats.get('global_speaking_frames', track_stats.get('speaking_frames', 0)) > 0
    return spoke or track_stats.get('motion', 0.0) >= MIN_MOTION


def analyze_talknet_layout(tracks_path, scores_path, min_speaking_percentage=15.0,
                           video_width=1920, video_height=1080):
    """
    Analyze TalkNet tracks and scores to determine the video layout mode
    (single speaker, split screen, or wide pad) and their center coordinates.
    """
    if not os.path.exists(tracks_path) or not os.path.exists(scores_path):
        return {"error": "tracks.pckl or scores.pckl not found"}

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)
    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    if len(tracks) != len(scores):
        return {"error": f"Mismatch: {len(tracks)} tracks vs {len(scores)} scores"}

    track_stats = []
    for i, (track, score) in enumerate(zip(tracks, scores)):
        frames, center_xs, center_ys, scales = get_track_data(track)
        num_frames = len(frames)
        
        if num_frames == 0:
            continue
            
        score_arr = np.array(score)
        speaking_frames = np.sum(score_arr > 0)
        speaking_percentage = (speaking_frames / num_frames) * 100
        
        motion = track_motion(center_xs, center_ys, scales)
        
        # If this track NEVER speaks AND does not move, it's a prop
        if speaking_frames == 0 and motion < MIN_MOTION:
            print(f"  Warning: Ignoring global Track {i} (prop/poster detected, motion {motion:.3f})")
            continue
        
        # Calculate the average center position of the face
        avg_x, avg_y = video_width / 2, video_height / 2  # Fallback to frame center
        if len(center_xs) > 0 and len(center_ys) > 0:
            if speaking_frames > 0:
                speaking_indices = np.where(score_arr > 0)[0]
                avg_x = float(np.mean([center_xs[idx] for idx in speaking_indices if idx < len(center_xs)]))
                avg_y = float(np.mean([center_ys[idx] for idx in speaking_indices if idx < len(center_ys)]))
            else:
                avg_x = float(np.mean(center_xs))
                avg_y = float(np.mean(center_ys))
            
        track_stats.append({
            "track_id": i,
            "speaking_percentage": float(speaking_percentage),
            "speaking_frames": int(speaking_frames),
            "num_frames": num_frames,
            "motion": motion,
            "avg_x": avg_x,
            "avg_y": avg_y
        })
    
    # Cluster tracks by left/right X position (left = x < half width, right = x >= half width)
    half_width = video_width / 2
    left_tracks = [t for t in track_stats if t["avg_x"] < half_width]
    right_tracks = [t for t in track_stats if t["avg_x"] >= half_width]

    def best_track(tracks):
        """Pick the track with most speaking frames; fall back to most frames if none speak."""
        if not tracks:
            return None
        speaking = [t for t in tracks if t["speaking_frames"] > 0]
        pool = speaking if speaking else tracks
        return max(pool, key=lambda t: t["speaking_frames"] if speaking else t["num_frames"])

    left = best_track(left_tracks)
    right = best_track(right_tracks)

    result = {
        "mode": "pad",
        "speakers": track_stats,
        "all_tracks_analyzed": len(track_stats),
        "target_coordinates": []
    }

    if left and right:
        left_frames = left["num_frames"]
        right_frames = right["num_frames"]
        total_frames = left_frames + right_frames
        left_ratio = left_frames / total_frames
        right_ratio = right_frames / total_frames

        MIN_RATIO = 0.20        # Each cluster must be ≥20% of combined frames
        MIN_FRAMES = 50         # Each cluster must have ≥50 frames (~2s at 25fps)
        
        both_substantial = (
            left_ratio >= MIN_RATIO and right_ratio >= MIN_RATIO and
            left_frames >= MIN_FRAMES and right_frames >= MIN_FRAMES and
            looks_human(left) and looks_human(right)
        )

        if both_substantial:
            result["mode"] = "split"
            result["target_coordinates"] = [
                {"x": left["avg_x"], "y": left["avg_y"]},
                {"x": right["avg_x"], "y": right["avg_y"]}
            ]
            print(f"  Split mode: left {left_frames}fr ({left_ratio:.0%}, spk {left['speaking_percentage']:.1f}%), right {right_frames}fr ({right_ratio:.0%}, spk {right['speaking_percentage']:.1f}%)")
        else:
            # One cluster is too small -> likely a false detection, use dominant side
            dominant = left if left_frames >= right_frames else right
            result["mode"] = "single"
            result["target_coordinates"] = [{"x": dominant["avg_x"], "y": dominant["avg_y"]}]
            print(f"  Single mode: dominant side has {max(left_frames, right_frames)}fr, other side only {min(left_frames, right_frames)}fr (filtered as false detection)")
    elif left or right:
        speaker = left or right
        if speaker:
            result["mode"] = "single"
            result["target_coordinates"] = [{"x": speaker["avg_x"], "y": speaker["avg_y"]}]
        else:
            result["mode"] = "pad"
            result["target_coordinates"] = []

    return result

def detect_scenes_from_tracks(track_data, merge_gap=0, min_scene_frames=12):
    """
    Infer scene (camera shot) boundaries from TalkNet track frame ranges.

    TalkNet processes each camera cut independently, so tracks within the same
    scene have overlapping frame ranges. Tracks in different scenes have a gap
    between them (TalkNet skips no frames within a scene, but there are gaps
    at scene boundaries).

    merge_gap: frames within this distance are merged into one scene
               (~0.5s at 25fps; handles minor detection holes within a shot)
    min_scene_frames: scenes shorter than this merge into the previous one
                      (avoids 1-2 frame micro-segments)
    Returns: list of (start_frame, end_frame) tuples, sorted by start_frame
    """
    if not track_data:
        return []

    intervals = [(td['min_frame'], td['max_frame'])
                 for td in track_data if td['frames']]
    if not intervals:
        return []

    intervals.sort()

    # Merge overlapping or close-together intervals into one scene
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # Absorb very short scenes into the previous one to avoid tiny segments
    result = []
    for scene in merged:
        if result and (scene[1] - scene[0]) < min_scene_frames:
            result[-1][1] = scene[1]
        else:
            result.append(scene)

    return [(s[0], s[1]) for s in result]


def _decide_layout(scene_track_stats, video_width=1920, video_height=1080, min_frames=10):
    """
    Given per-scene track stats, decide: 'single', 'split', or 'pad'.
    Returns (mode, target_coordinates).

    Each stats dict needs avg_x, avg_y, avg_s, speaking_frames, speaking_pct,
    num_frames and motion (see track_motion). A side whose face never speaks
    is kept as a listener only if it moves; still detections (KAWS figures,
    ad posters) are treated as props. TalkNet tracks are per shot, so whether
    the person spoke in another shot is not known here.
    """
    half_width = video_width / 2
    left_tracks = [t for t in scene_track_stats if t['avg_x'] < half_width]
    right_tracks = [t for t in scene_track_stats if t['avg_x'] >= half_width]

    def best_track(pool):
        if not pool:
            return None
        speaking = [t for t in pool if t['speaking_frames'] > 0]
        if speaking:
            return max(speaking, key=lambda t: t['speaking_frames'])
        # No one speaks on this side: prefer a moving face over a still prop
        moving = [t for t in pool if looks_human(t)]
        return max(moving or pool, key=lambda t: t['num_frames'])

    left = best_track(left_tracks)
    right = best_track(right_tracks)

    if left and right:
        lf = left['num_frames']
        rf = right['num_frames']
        total = lf + rf

        MIN_RATIO    = 0.20   # each side must be ≥20% of combined frames
        MIN_FRAMES   = min_frames
        MIN_SPEAKING = 2.0    # ≥2% speaking → confirmed real human
        MIN_SIZE_RATIO = 0.65 # non-speaking face must be ≥65% size of speaking face

        frames_ok = (lf / total >= MIN_RATIO and rf / total >= MIN_RATIO and
                     lf >= MIN_FRAMES and rf >= MIN_FRAMES)

        left_speaks  = left['speaking_pct']  >= MIN_SPEAKING
        right_speaks = right['speaking_pct'] >= MIN_SPEAKING

        if frames_ok and left_speaks and right_speaks:
            return "split", [
                {"x": left['avg_x'],  "y": left['avg_y'], "s": left['avg_s']},
                {"x": right['avg_x'], "y": right['avg_y'], "s": right['avg_s']},
            ]

        elif frames_ok and (left_speaks or right_speaks):
            speaking_s  = left['avg_s']  if left_speaks  else right['avg_s']
            silent_s    = right['avg_s'] if left_speaks  else left['avg_s']
            silent_side = right          if left_speaks  else left

            # A listener moves; a prop (poster, figure) stays still
            silent_is_human = looks_human(silent_side)
            silent_motion = silent_side.get('motion', 0.0)
            size_ok = (speaking_s > 0 and silent_s >= speaking_s * MIN_SIZE_RATIO)

            if silent_is_human and size_ok:
                print(f"    Split (listener): spk_s={speaking_s:.1f}, sil_s={silent_s:.1f}, sil_motion={silent_motion:.3f}")
                return "split", [
                    {"x": left['avg_x'],  "y": left['avg_y'], "s": left['avg_s']},
                    {"x": right['avg_x'], "y": right['avg_y'], "s": right['avg_s']},
                ]
            else:
                reason = "static, likely a prop" if not silent_is_human else "too small"
                dominant = left if left_speaks else right
                print(f"    Single (silent face filtered - {reason}): spk_s={speaking_s:.1f}, sil_s={silent_s:.1f}, sil_motion={silent_motion:.3f}")
                return "single", [{"x": dominant['avg_x'], "y": dominant['avg_y'], "s": dominant['avg_s']}]

        else:
            if left['speaking_frames'] > 0 and right['speaking_frames'] == 0:
                dominant = left
            elif right['speaking_frames'] > 0 and left['speaking_frames'] == 0:
                dominant = right
            else:
                dominant = left if lf >= rf else right
            return "single", [{"x": dominant['avg_x'], "y": dominant['avg_y'], "s": dominant['avg_s']}]

    elif left or right:
        speaker = left or right
        if speaker:
            return "single", [{"x": speaker['avg_x'], "y": speaker['avg_y'], "s": speaker['avg_s']}]
        else:
            return "pad", []

    return "pad", []


def analyze_talknet_per_scene(tracks_path, scores_path,
                               fps=25.0, video_width=1920, video_height=1080,
                               video_duration=None):
    """
    Analyze TalkNet output per camera shot and return a list of scene layout dicts.

    Each dict has:
        start_time, end_time  (seconds, for FFmpeg -ss / -t)
        mode                  ('single', 'split', or 'pad')
        target_coordinates    list of {x, y} face center points
    """
    if not os.path.exists(tracks_path) or not os.path.exists(scores_path):
        return []

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)
    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    if len(tracks) != len(scores):
        print(f"  Warning: track/score count mismatch ({len(tracks)} vs {len(scores)})")
        return []

    # Build per-track data with frame-level positions and speaking scores
    track_data = []
    global_speaking_frames = {} # Track global speaking frames
    
    for i, (track_item, score) in enumerate(zip(tracks, scores)):
        frames, center_xs, center_ys, scales = get_track_data(track_item)
        if len(frames) == 0:
            continue

        score_arr = np.array(score)
        
        # Calculate total speaking frames for this track
        spk_frames = np.sum(score_arr > 0)
        global_speaking_frames[i] = spk_frames
        
        motion = track_motion(center_xs, center_ys, scales)
        
        # If this track NEVER speaks AND does not move, it's a prop
        if spk_frames == 0 and motion < MIN_MOTION:
            print(f"  Warning: Ignoring Track {i} (prop/poster detected, motion {motion:.3f})")
            continue
            
        frame_positions = {}
        for j, frame_num in enumerate(frames):
            cx = float(center_xs[j]) if j < len(center_xs) else video_width / 2
            cy = float(center_ys[j]) if j < len(center_ys) else video_height / 2
            spk = float(score_arr[j]) if j < len(score_arr) else 0.0
            s   = float(scales[j])    if j < len(scales)    else 0.0
            frame_positions[int(frame_num)] = {'x': cx, 'y': cy, 'score': spk, 's': s}

        track_data.append({
            'track_id':  i,
            'frames':    [int(f) for f in frames],
            'min_frame': int(min(frames)),
            'max_frame': int(max(frames)),
            'frame_positions': frame_positions,
            'global_speaking_frames': spk_frames,
        })

    # Step 1: detect scene boundaries from track frame ranges
    scene_boundaries = detect_scenes_from_tracks(track_data)
    if not scene_boundaries:
        return []

    # Step 2: for each scene, compute per-track stats and decide layout
    scene_layouts = []
    for scene_start, scene_end in scene_boundaries:
        scene_track_stats = []
        for td in track_data:
            scene_frames = [f for f in td['frames'] if scene_start <= f <= scene_end]
            if len(scene_frames) < 2:
                continue
            positions = [td['frame_positions'][f] for f in scene_frames
                         if f in td['frame_positions']]
            if not positions:
                continue
            spk_count = sum(1 for p in positions if p['score'] > 0)
            s_values  = [p['s'] for p in positions if p['s'] > 0]
            motion    = track_motion([p['x'] for p in positions],
                                     [p['y'] for p in positions],
                                     [p['s'] for p in positions])
            scene_track_stats.append({
                'track_id':        td['track_id'],
                'avg_x':           float(np.mean([p['x'] for p in positions])),
                'avg_y':           float(np.mean([p['y'] for p in positions])),
                'avg_s':           float(np.mean(s_values)) if s_values else 0.0,
                'speaking_frames': spk_count,
                'global_speaking_frames': td.get('global_speaking_frames', spk_count),
                'speaking_pct':    (spk_count / len(positions)) * 100,
                'num_frames':      len(positions),
                'motion':          motion,
            })

        # Minimum frames per cluster proportional to scene length
        min_fr = max(5, int(fps * 0.4))
        mode, coords = _decide_layout(scene_track_stats, video_width, video_height, min_fr)

        start_t = scene_start / fps
        end_t   = (scene_end + 1) / fps
        print(f"    Scene [{scene_start}-{scene_end}] {start_t:.2f}s-{end_t:.2f}s → {mode.upper()}")
        scene_layouts.append({
            "start_time":        start_t,
            "end_time":          end_t,
            "mode":              mode,
            "target_coordinates": coords,
        })

    # Step 3: fill any gaps (pre-roll, post-roll, inter-scene transitions)
    # Use the adjacent scene's layout so transitions look natural
    if not scene_layouts:
        return []

    filled = []

    # Gap before first scene
    if scene_layouts[0]["start_time"] > 0.05:
        filled.append({
            "start_time": 0.0,
            "end_time":   scene_layouts[0]["start_time"],
            "mode":       scene_layouts[0]["mode"],
            "target_coordinates": scene_layouts[0]["target_coordinates"],
        })

    for i, layout in enumerate(scene_layouts):
        filled.append(layout)
        # Gap to next scene?
        if i + 1 < len(scene_layouts):
            gap_start = layout["end_time"]
            gap_end   = scene_layouts[i + 1]["start_time"]
            if gap_end - gap_start > 0.05:
                # Carry the current layout through the transition gap
                filled.append({
                    "start_time": gap_start,
                    "end_time":   gap_end,
                    "mode":       layout["mode"],
                    "target_coordinates": layout["target_coordinates"],
                })

    # Gap after last scene
    if video_duration and scene_layouts[-1]["end_time"] < video_duration - 0.05:
        filled.append({
            "start_time": scene_layouts[-1]["end_time"],
            "end_time":   video_duration,
            "mode":       scene_layouts[-1]["mode"],
            "target_coordinates": scene_layouts[-1]["target_coordinates"],
        })

    return filled


def main():
    parser = argparse.ArgumentParser(description="TalkNet Speaker Detection and Layout Analyzer")
    parser.add_argument("--pywork", required=True, help="Path to TalkNet pywork directory containing .pckl files")
    parser.add_argument("--output", help="Path to save layout JSON")
    
    args = parser.parse_args()
    
    tracks_path = os.path.join(args.pywork, "tracks.pckl")
    scores_path = os.path.join(args.pywork, "scores.pckl")
    
    layout = analyze_talknet_layout(tracks_path, scores_path)
    
    if "error" in layout:
        print(f"Error: {layout['error']}")
        return
        
    print(f"Determined Layout Mode: {layout['mode'].upper()}")
    print(f"Active Speakers: {len(layout['speakers'])}")
    for spk in layout.get('speakers', []):
        print(f"  Track {spk['track_id']}: {spk['speaking_percentage']:.1f}% speaking at X:{spk['avg_x']:.1f}, Y:{spk['avg_y']:.1f}")
        
    if args.output:
        with open(args.output, "w") as f:
            json.dump(layout, f, indent=2)
        print(f"Saved layout analysis to {args.output}")

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    main()