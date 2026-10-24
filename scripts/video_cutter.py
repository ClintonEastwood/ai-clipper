#!/usr/bin/env python3
"""
Video Cutter - Extract clips from video based on merged segments JSON

Cuts video clips using segment timestamps from merge_rulebased.py.
Optionally runs post-merge optimization to combine adjacent segments.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def seconds_to_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def validate_video_file(video_path):
    """Check if video file exists and is readable"""
    path = Path(video_path)
    if not path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return False
    if not path.is_file():
        print(f"❌ Error: Not a file: {video_path}")
        return False
    return True

def validate_segments_file(json_path):
    """Check if segments JSON exists and is valid"""
    path = Path(json_path)
    if not path.exists():
        print(f"❌ Error: Segments file not found: {json_path}")
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'segments' not in data:
            print(f"❌ Error: Invalid segments JSON - missing 'segments' key")
            return False
        return True
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file: {e}")
        return False

def cut_video_clip(video_path, start_time, duration, output_path, codec='copy', crf=23, vertical=False, vertical_mode='crop', crop_position='center', pad_blur=False, verbose=False):
    """
    Extract a single clip from video using ffmpeg

    Args:
        video_path: Input video file path
        start_time: Start time in seconds
        duration: Duration in seconds
        output_path: Output clip file path
        codec: 'copy' for fast lossless, 'libx264' for re-encode
        crf: Quality for re-encode (18=high, 23=default, 28=low)
        vertical: Convert to vertical 1080x1920 format (default: False)
        vertical_mode: 'crop' (cut sides) or 'pad' (add bars top/bottom) (default: 'crop')
        crop_position: Horizontal crop position for crop mode: 'left', 'center', 'right' (default: 'center')
        pad_blur: Use blurred background instead of black bars for pad mode (default: False)
        verbose: Show ffmpeg output
    """

    # Note: vertical format requires re-encoding (can't use codec=copy with filters)
    if vertical and codec == 'copy':
        codec = 'libx264'
        if verbose:
            print("  ⚠️  Vertical mode requires re-encoding (switching from copy to libx264)")

    if codec == 'copy' and not vertical:
        # Fast lossless copy (no re-encode)
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite
            '-ss', str(start_time),  # Start time
            '-i', str(video_path),   # Input file
            '-t', str(duration),     # Duration
            '-c', 'copy',            # Copy codec (fast)
            '-avoid_negative_ts', '1',  # Fix timestamp issues
            str(output_path)
        ]
    else:
        # Re-encode with specified codec
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(start_time),
            '-i', str(video_path),
            '-t', str(duration),
        ]

        # Add vertical conversion filters if requested
        if vertical:
            if vertical_mode == 'crop':
                # MODE 1: CROP - Cut the sides to fit 9:16
                # Convert 1920x1080 landscape to 1080x1920 vertical (9:16 ratio)
                # Step 1: Crop to 608x1080 (9:16 ratio)
                # Step 2: Scale to 1080x1920

                # Calculate crop position
                # Source: 1920x1080, Target ratio for crop: 608x1080 (9:16)
                crop_width = 608
                crop_height = 1080

                if crop_position == 'left':
                    crop_x = 0
                elif crop_position == 'right':
                    crop_x = 1920 - crop_width  # 1312
                else:  # center
                    crop_x = (1920 - crop_width) // 2  # 656

                crop_y = 0

                # FFmpeg filter: crop then scale
                video_filter = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale=1080:1920"

                cmd.extend([
                    '-vf', video_filter,
                    '-c:v', codec,
                    '-crf', str(crf),
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    str(output_path)
                ])

            else:  # vertical_mode == 'pad'
                # MODE 2: PAD - Keep full video, add bars top/bottom
                # Scale 1920x1080 to fit 1080 width -> 1080x608
                # Then pad to 1080x1920 (add 656px top + 656px bottom)

                if pad_blur:
                    # Advanced: Blurred background
                    # Requires -filter_complex (not -vf) because of multiple inputs
                    # [0:v] = main input, split into 2 streams: blur background + main video
                    video_filter = (
                        # Blur a 1/4-size copy, then upscale: looks the same, ~14x faster than boxblur on 1080x1920
                        "[0:v]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=10:5,scale=1080:1920[bg];"
                        "[0:v]scale=1080:-1[fg];"
                        "[bg][fg]overlay=(W-w)/2:(H-h)/2"
                    )

                    cmd.extend([
                        '-filter_complex', video_filter,
                        '-c:v', codec,
                        '-crf', str(crf),
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        str(output_path)
                    ])
                else:
                    # Simple: Black bars (letterbox)
                    # Scale to 1080 width (maintains aspect ratio) -> 1080x608
                    # Pad to 1080x1920 with black bars (656px top, 656px bottom)
                    video_filter = "scale=1080:608,pad=1080:1920:0:656:black"

                    cmd.extend([
                        '-vf', video_filter,
                        '-c:v', codec,
                        '-crf', str(crf),
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        str(output_path)
                    ])
        else:
            # Standard re-encode (no filters)
            cmd.extend([
                '-c:v', codec,           # Video codec
                '-crf', str(crf),        # Quality
                '-c:a', 'aac',           # Audio codec
                '-b:a', '192k',          # Audio bitrate
                str(output_path)
            ])

    # Run ffmpeg
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.PIPE if not verbose else None,
        text=True
    )

    return result.returncode == 0, result.stderr if result.returncode != 0 else None

def sanitize_filename(text, max_length=50):
    """Create safe filename from text"""
    # Remove unsafe characters
    safe = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_'))
    # Limit length
    safe = safe[:max_length].strip()
    # Replace spaces with underscores
    safe = safe.replace(' ', '_')
    return safe

def analyze_gaps(segments: List[Dict[str, Any]], max_gap: float = 5.0, max_duration: float = 120.0) -> List[Dict[str, Any]]:
    """
    Analyze gaps between adjacent segments to find merge candidates

    Args:
        segments: List of segment dictionaries
        max_gap: Maximum gap to consider for merging (seconds)
        max_duration: Maximum combined duration (seconds)

    Returns:
        List of merge candidate dictionaries
    """
    candidates = []

    for i in range(len(segments) - 1):
        curr = segments[i]
        next_seg = segments[i + 1]

        gap = next_seg['start_time'] - curr['end_time']
        combined_duration = next_seg['end_time'] - curr['start_time']

        # Only consider excellent candidates (gap < max_gap)
        if gap <= max_gap and combined_duration <= max_duration:
            candidates.append({
                'curr_index': i,
                'next_index': i + 1,
                'gap': gap,
                'combined_duration': combined_duration
            })

    return candidates

def create_merged_segment(curr_seg: Dict[str, Any], next_seg: Dict[str, Any], gap: float) -> Dict[str, Any]:
    """Create a merged segment from two adjacent segments"""
    merged = {
        'topic': f"{curr_seg['topic']}+{next_seg['topic']}" if curr_seg['topic'] != next_seg['topic'] else curr_seg['topic'],
        'start_time': curr_seg['start_time'],
        'end_time': next_seg['end_time'],
        'duration': next_seg['end_time'] - curr_seg['start_time'],
        'confidence': (curr_seg.get('confidence', 0.5) + next_seg.get('confidence', 0.5)) / 2,
        'text': curr_seg['text'] + " " + next_seg['text'],
        'reason': f"{curr_seg.get('reason', '')} | {next_seg.get('reason', '')}",
        'source_file': f"{curr_seg.get('source_file', 'unknown')} + {next_seg.get('source_file', 'unknown')}",
        'badge_label': curr_seg.get('badge_label', '') or next_seg.get('badge_label', ''),
        'badge_emoji': curr_seg.get('badge_emoji', '') or next_seg.get('badge_emoji', ''),
        'post_merge_metadata': {
            'merged_from_indices': [curr_seg.get('original_index', -1), next_seg.get('original_index', -1)],
            'gap_seconds': gap,
            'merge_type': 'post_optimization',
            'original_durations': [curr_seg['duration'], next_seg['duration']]
        }
    }

    return merged

def optimize_segments(segments_data: Dict[str, Any], max_gap: float = 5.0) -> Dict[str, Any]:
    """
    Optimize segments by merging adjacent ones with small gaps

    Args:
        segments_data: Original segments JSON data
        max_gap: Maximum gap for merging (seconds)

    Returns:
        Optimized segments JSON data
    """
    segments = segments_data.get('segments', [])

    # Add original index to segments
    for i, seg in enumerate(segments):
        seg['original_index'] = i

    # Find merge candidates
    candidates = analyze_gaps(segments, max_gap=max_gap)

    print(f"\n🔍 Post-Merge Optimizer: Found {len(candidates)} merge candidates (gap ≤ {max_gap}s)")

    # Track which segments have been merged
    merged_indices = set()
    optimized_segments = []

    for candidate in candidates:
        curr_idx = candidate['curr_index']
        next_idx = candidate['next_index']

        # Skip if either segment was already merged
        if curr_idx in merged_indices or next_idx in merged_indices:
            continue

        curr_seg = segments[curr_idx]
        next_seg = segments[next_idx]

        # Create merged segment
        merged_seg = create_merged_segment(curr_seg, next_seg, candidate['gap'])
        optimized_segments.append(merged_seg)

        # Mark as merged
        merged_indices.add(curr_idx)
        merged_indices.add(next_idx)

        print(f"   ✓ Merged segments #{curr_idx} + #{next_idx}: gap {candidate['gap']:.2f}s → {candidate['combined_duration']:.2f}s")

    # Add remaining non-merged segments
    for i, seg in enumerate(segments):
        if i not in merged_indices:
            seg_copy = seg.copy()
            seg_copy['post_merge_metadata'] = {
                'merged_from_indices': [i],
                'merge_type': 'original_kept',
                'original_durations': [seg['duration']]
            }
            optimized_segments.append(seg_copy)

    # Sort by start time
    optimized_segments.sort(key=lambda x: x['start_time'])

    print(f"   📊 Results: {len(segments)} → {len(optimized_segments)} segments (reduced by {len(segments) - len(optimized_segments)})\n")

    # Create optimized data structure
    optimized_data = {
        'pipeline_metadata': {
            'pipeline_version': 'video_cutter_with_optimizer_v1',
            'original_segments': len(segments),
            'optimized_segments': len(optimized_segments),
            'reduction_count': len(segments) - len(optimized_segments),
            'merged_count': len([s for s in optimized_segments if s.get('post_merge_metadata', {}).get('merge_type') == 'post_optimization'])
        },
        'merge_config': {
            'max_gap': max_gap,
            'max_duration': 120.0
        },
        'segments': optimized_segments
    }

    return optimized_data

def cut_all_clips(video_path, segments_json, output_dir, codec='copy', crf=23,
                  name_format='topic', min_duration=None, max_duration=None,
                  filter_topics=None, min_confidence=None, optimizer_mode='all',
                  vertical=False, vertical_mode='crop', crop_position='center', pad_blur=False,
                  verbose=False, dry_run=False):
    """
    Extract all clips from segments JSON

    Args:
        video_path: Input video file
        segments_json: Path to merged_segments.json
        output_dir: Output directory for clips
        codec: 'copy' (fast) or 'libx264' (re-encode)
        crf: Quality for re-encode (18-28)
        name_format: Naming format ('topic', 'timestamp', 'sequential')
        min_duration: Filter segments shorter than this (seconds)
        max_duration: Filter segments longer than this (seconds)
        filter_topics: List of topics to include (None = all)
        min_confidence: Minimum confidence threshold (0.0-1.0)
        optimizer_mode: 'all' (extract all segments) or 'merge-only' (only merged segments)
        vertical: Convert to vertical 1080x1920 format for TikTok/Reels (default: False)
        vertical_mode: 'crop' (cut sides) or 'pad' (add bars top/bottom) (default: 'crop')
        crop_position: Horizontal crop position for crop mode: 'left', 'center', 'right' (default: 'center')
        pad_blur: Use blurred background instead of black bars for pad mode (default: False)
        verbose: Show detailed ffmpeg output
        dry_run: Print what would be done without executing
    """

    # Load segments
    with open(segments_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get('segments', [])

    # Filter by optimizer mode
    if optimizer_mode == 'merge-only':
        # Only include segments that were newly merged
        segments = [s for s in segments
                   if s.get('post_merge_metadata', {}).get('merge_type') == 'post_optimization']
        print(f"🎯 Optimizer mode: merge-only - extracting only {len(segments)} merged segments\n")
    elif optimizer_mode == 'all':
        print(f"🎯 Optimizer mode: all - extracting all {len(segments)} segments\n")

    # Apply filters
    filtered_segments = []
    for seg in segments:
        # Duration filter
        duration = seg['duration']
        if min_duration and duration < min_duration:
            continue
        if max_duration and duration > max_duration:
            continue

        # Topic filter
        if filter_topics and seg['topic'] not in filter_topics:
            continue

        # Confidence filter
        if min_confidence and seg.get('confidence', 0) < min_confidence:
            continue

        filtered_segments.append(seg)

    total = len(filtered_segments)

    if total == 0:
        print("❌ No segments match the specified filters!")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Print summary
    print("\n" + "="*80)
    print("VIDEO CUTTER - BATCH CLIP EXTRACTION")
    print("="*80)
    print(f"Input video:     {video_path}")
    print(f"Segments file:   {segments_json}")
    print(f"Output dir:      {output_dir}")
    print(f"Total segments:  {len(segments)}")
    print(f"After filters:   {total}")
    print(f"Codec:           {codec}")
    if codec != 'copy' or vertical:
        print(f"Quality (CRF):   {crf}")
    if vertical:
        print(f"Format:          VERTICAL 1080x1920 (TikTok/Reels) 📱")
        print(f"Vertical mode:   {vertical_mode.upper()}")
        if vertical_mode == 'crop':
            print(f"Crop position:   {crop_position}")
        elif vertical_mode == 'pad':
            print(f"Padding style:   {'Blurred background' if pad_blur else 'Black bars'}")
    else:
        print(f"Format:          Original (landscape)")
    print(f"Dry run:         {dry_run}")
    print("="*80 + "\n")

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be created\n")

    # Process each segment
    success_count = 0
    fail_count = 0
    total_duration = 0

    for i, seg in enumerate(filtered_segments, 1):
        start = seg['start_time']
        duration = seg['duration']
        topic = seg['topic']
        confidence = seg.get('confidence', 0)
        text = seg.get('text', '')

        # Generate filename based on format
        if name_format == 'topic':
            # topic_timestamp_duration.mp4
            text_preview = sanitize_filename(text[:30])
            filename = f"{topic}_{int(start)}s_{int(duration)}s_{text_preview}.mp4"
        elif name_format == 'timestamp':
            # HH-MM-SS_duration.mp4
            time_str = seconds_to_timestamp(start).replace(':', '-')
            filename = f"{time_str}_{int(duration)}s.mp4"
        else:  # sequential
            # clip_001.mp4
            filename = f"clip_{i:03d}_{topic}.mp4"

        output_file = output_path / filename

        # Display info
        print(f"[{i}/{total}] {topic.upper()}")
        print(f"  Time:       {seconds_to_timestamp(start)} ({start:.2f}s)")
        print(f"  Duration:   {duration:.1f}s")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Text:       {text[:70]}{'...' if len(text) > 70 else ''}")

        # Show merge/expansion info
        if 'merge_metadata' in seg:
            meta = seg['merge_metadata']
            print(f"  📎 Merged:  {meta['num_merged']} segments (gap: {meta['gap_seconds']:.1f}s)")
        elif 'expansion_metadata' in seg:
            print(f"  ↔️  Expanded with context")

        print(f"  Output:     {filename}")

        if dry_run:
            print(f"  Status:     ⏭️  Skipped (dry run)")
        else:
            print(f"  Progress:   ", end='', flush=True)
            success, error = cut_video_clip(
                video_path, start, duration, output_file,
                codec=codec, crf=crf, vertical=vertical,
                vertical_mode=vertical_mode, crop_position=crop_position,
                pad_blur=pad_blur, verbose=verbose
            )

            if success:
                # Get output file size
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✅ Done ({size_mb:.2f} MB)")
                success_count += 1
                total_duration += duration
            else:
                print(f"❌ Failed")
                if error and verbose:
                    print(f"  Error: {error[:200]}")
                fail_count += 1

        print()

    # Final summary
    print("="*80)
    if not dry_run:
        print(f"✅ Successfully extracted: {success_count}/{total} clips")
        if fail_count > 0:
            print(f"❌ Failed: {fail_count}/{total} clips")
        print(f"📊 Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"📁 Output directory: {output_dir}")

        # Save clips_metadata.json sidecar
        clips_metadata = {}
        for i, seg in enumerate(filtered_segments, 1):
            start = seg['start_time']
            duration = seg['duration']
            topic = seg['topic']
            text = seg.get('text', '')

            if name_format == 'topic':
                text_preview = sanitize_filename(text[:30])
                filename = f"{topic}_{int(start)}s_{int(duration)}s_{text_preview}.mp4"
            elif name_format == 'timestamp':
                time_str = seconds_to_timestamp(start).replace(':', '-')
                filename = f"{time_str}_{int(duration)}s.mp4"
            else:
                filename = f"clip_{i:03d}_{topic}.mp4"

            clips_metadata[filename] = {
                'topic': topic,
                'badge_label': seg.get('badge_label', ''),
                'badge_emoji': seg.get('badge_emoji', ''),
                'confidence': seg.get('confidence', 0),
                'start_time': start,
                'duration': duration,
                'reason': seg.get('reason', ''),
                'text': text,  # add_badge.py writes the hook title from it when no captions ran
            }

        metadata_path = output_path / 'clips_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(clips_metadata, f, ensure_ascii=False, indent=2)
        print(f"📋 Metadata saved: {metadata_path}")
    else:
        print(f"🔍 Dry run complete - {total} clips would be extracted")
        print(f"📊 Total duration: {sum(s['duration'] for s in filtered_segments):.1f}s")
    print("="*80 + "\n")
    return fail_count

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract video clips based on merged segments JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - extract all clips (fast, lossless)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/

  # Re-encode for better compatibility (slower, smaller files)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/ --codec libx264 --crf 23

  # Filter by duration (only 30-60s clips)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/ --min-duration 30 --max-duration 60

  # Filter by topic
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/ --topics controversial

  # Filter by confidence (high quality only)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/ --min-confidence 0.8

  # Dry run (preview what will be done)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/ --dry-run

  # Different naming formats
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/ --name-format sequential
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/ --name-format timestamp

  # WITH POST-MERGE OPTIMIZER (combines adjacent segments)
  # Extract all segments (including merged ones)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/ --optimizer --optimizer-mode all

  # Extract only the newly merged segments (not the originals)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/ --optimizer --optimizer-mode merge-only

  # Adjust the gap threshold for merging (default: 5s)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips/ --optimizer --optimizer-max-gap 3.0

  # VERTICAL FORMAT (1080x1920 for TikTok/Reels)

  # Option 1: CROP mode (cuts the sides to fit 9:16)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips_vertical/ --vertical

  # Crop with custom position (left/right)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips_vertical/ --vertical --crop-position left

  # Option 2: PAD mode (keeps full video, adds black bars top/bottom)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips_vertical/ --vertical --vertical-mode pad

  # Pad with blurred background (professional look)
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips_vertical/ --vertical --vertical-mode pad --pad-blur

  # High quality vertical
  python video_cutter.py --input input/video.mp4 --segments data/analysis_results/merged_optimized/merged_segments.json --output clips_vertical/ --vertical --crf 20

Quality Notes:
  - codec=copy: Fast, lossless, large files (recommended for first pass)
  - codec=libx264 crf=18: High quality, larger files
  - codec=libx264 crf=23: Default quality (good balance)
  - codec=libx264 crf=28: Lower quality, smaller files (for web upload)
        """
    )

    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                        help='Input video file (.mp4)')
    parser.add_argument('--segments', '-s',
                        default='data/analysis_results/merged_optimized/merged_segments.json',
                        help='Path to merged segments JSON (default: %(default)s)')
    parser.add_argument('--output', '-o', default='clips/',
                        help='Output directory for clips (default: %(default)s)')

    # Encoding options
    parser.add_argument('--codec', choices=['copy', 'libx264'], default='copy',
                        help='Video codec: copy (fast) or libx264 (re-encode) (default: %(default)s)')
    parser.add_argument('--crf', type=int, default=23, choices=range(18, 29),
                        help='Quality for re-encode: 18=high, 23=default, 28=low (default: %(default)s)')

    # Vertical format conversion
    parser.add_argument('--vertical', action='store_true',
                        help='Convert to vertical 1080x1920 format for TikTok/Reels (requires re-encoding)')
    parser.add_argument('--vertical-mode', choices=['crop', 'pad'], default='crop',
                        help='Vertical conversion mode: "crop"=cut sides, "pad"=add bars top/bottom (default: %(default)s)')
    parser.add_argument('--crop-position', choices=['left', 'center', 'right'], default='center',
                        help='Horizontal crop position for crop mode (default: %(default)s)')
    parser.add_argument('--pad-blur', action='store_true',
                        help='Use blurred background instead of black bars for pad mode (default: False)')

    # Naming options
    parser.add_argument('--name-format', choices=['topic', 'timestamp', 'sequential'], default='topic',
                        help='Filename format (default: %(default)s)')

    # Filtering options
    parser.add_argument('--min-duration', type=float, metavar='SECONDS',
                        help='Minimum clip duration in seconds')
    parser.add_argument('--max-duration', type=float, metavar='SECONDS',
                        help='Maximum clip duration in seconds')
    parser.add_argument('--topics', nargs='+', metavar='TOPIC',
                        help='Filter by topics (space-separated)')
    parser.add_argument('--min-confidence', type=float, metavar='0.0-1.0',
                        help='Minimum confidence threshold (0.0-1.0)')

    # Optimizer options
    parser.add_argument('--optimizer', action='store_true',
                        help='Run post-merge optimization to combine adjacent segments (default: False)')
    parser.add_argument('--optimizer-mode', choices=['all', 'merge-only'], default='all',
                        help='Optimizer output mode: "all"=extract all segments, "merge-only"=only merged segments (default: %(default)s)')
    parser.add_argument('--optimizer-max-gap', type=float, default=5.0, metavar='SECONDS',
                        help='Maximum gap for merging adjacent segments (default: %(default)s seconds)')

    # Other options
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview what will be done without executing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed ffmpeg output')

    args = parser.parse_args()

    # Validate inputs
    if not validate_video_file(args.input):
        sys.exit(1)

    if not validate_segments_file(args.segments):
        sys.exit(1)

    # Run post-merge optimizer if requested
    segments_file = args.segments
    temp_optimized_file = None

    if args.optimizer:
        print("\n" + "="*80)
        print("🔧 POST-MERGE OPTIMIZER ENABLED")
        print("="*80)

        # Load original segments
        with open(args.segments, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        # Run optimization
        optimized_data = optimize_segments(original_data, max_gap=args.optimizer_max_gap)

        # Save to temporary file
        temp_optimized_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
        json.dump(optimized_data, temp_optimized_file, indent=2, ensure_ascii=False)
        temp_optimized_file.close()

        segments_file = temp_optimized_file.name
        print(f"✅ Optimized segments saved to temporary file\n")

    # Run cutter
    try:
        fail_count = cut_all_clips(
            video_path=args.input,
            segments_json=segments_file,
            output_dir=args.output,
            codec=args.codec,
            crf=args.crf,
            name_format=args.name_format,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            filter_topics=args.topics,
            min_confidence=args.min_confidence,
            optimizer_mode=args.optimizer_mode if args.optimizer else 'all',
            vertical=args.vertical,
            vertical_mode=args.vertical_mode,
            crop_position=args.crop_position,
            pad_blur=args.pad_blur,
            verbose=args.verbose,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temporary file if created
        if temp_optimized_file:
            import os
            try:
                os.unlink(temp_optimized_file.name)
            except:
                pass

    # Exit 0 here made run_pipeline mark the step done with no clips cut
    if fail_count:
        sys.exit(1)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    main()
