import os
import argparse
import re
from collections import defaultdict

# Heavy imports moved to main() to allow --help without dependencies

def group_by_time_window(timestamps, window_seconds=5):
    if not timestamps:
        return []

    segments = []
    current_segment = []
    segment_start = timestamps[0].start_time

    for item in timestamps:
        if item.end_time - segment_start > window_seconds:
            if current_segment:
                segments.append({
                    'text': ' '.join([w.text for w in current_segment]),
                    'start': current_segment[0].start_time,
                    'end': current_segment[-1].end_time
                })
            current_segment = [item]
            segment_start = item.start_time
        else:
            current_segment.append(item)

    if current_segment:
        segments.append({
            'text': ' '.join([w.text for w in current_segment]),
            'start': current_segment[0].start_time,
            'end': current_segment[-1].end_time
        })

    return segments

def extract_chunk_info(filename):
    """
    Extract base name and chunk number from filename.
    Example: 'podcast_0.wav' -> ('podcast', 0)
             'audio_chunk_5.wav' -> ('audio_chunk', 5)
    """
    base, ext = os.path.splitext(filename)
    match = re.match(r'(.+?)_(\d+)$', base)
    if match:
        return match.group(1), int(match.group(2))
    return base, 0

def group_audio_files(audio_files):
    """Group audio files by their base name."""
    groups = defaultdict(list)
    for audio_file in audio_files:
        base_name, chunk_num = extract_chunk_info(audio_file)
        groups[base_name].append((chunk_num, audio_file))

    # Sort each group by chunk number
    for base_name in groups:
        groups[base_name].sort(key=lambda x: x[0])

    return groups

def remove_overlap_duplicates(segments, chunk_duration, overlap):
    """
    Remove duplicate segments in overlap regions.
    Strategy: For chunk n>0, remove segments that start before chunk_start + overlap.
    This keeps the previous chunk's version (which has context after).

    Works with extended duration chunking (most common):
      Chunk 0: 0-30s, Chunk 1: 28-60s (32s), Chunk 2: 58-90s (32s)
      Overlap regions: 28-30s (chunks 0&1), 58-60s (chunks 1&2)

    Args:
        segments: List of segments with 'start', 'end', 'text', 'chunk_num'
        chunk_duration: Duration of each chunk (e.g., 30)
        overlap: Overlap duration (e.g., 2)

    Returns:
        List of segments with duplicates removed
    """
    if overlap == 0:
        return segments

    result = []
    removed_count = 0

    for seg in segments:
        chunk_num = seg.get('chunk_num', 0)

        if chunk_num == 0:
            # Chunk 0: keep all segments
            result.append(seg)
        else:
            # Chunk n>0: only keep segments that start after overlap boundary
            # For extended duration: chunk_start = n × chunk_duration - overlap
            chunk_start = chunk_num * chunk_duration - overlap
            overlap_boundary = chunk_start + overlap

            if seg['start'] >= overlap_boundary:
                result.append(seg)
            else:
                removed_count += 1
                # Uncomment for verbose logging:
                # print(f"    [Dedup] Removed duplicate from chunk {chunk_num}: {seg['start']:.2f}s")

    if removed_count > 0:
        print(f"  Removed {removed_count} duplicate segments from overlap regions")

    return result
def merge_short_segments(segments, min_duration=1.0, max_gap=1.0):
    """
    Merge short segments with adjacent segments to fix boundary fragmentation.

    Args:
        segments: List of segment dictionaries with 'text', 'start', 'end'
        min_duration: Minimum duration threshold (segments below this are candidates for merging)
        max_gap: Maximum time gap to consider segments adjacent for merging

    Returns:
        List of merged segments
    """
    if not segments:
        return []

    # Sort segments by start time
    segments = sorted(segments, key=lambda x: x['start'])

    merged = []
    i = 0

    while i < len(segments):
        current = segments[i]
        duration = current['end'] - current['start']

        # If segment is long enough, keep it as is
        if duration >= min_duration:
            merged.append(current)
            i += 1
            continue

        # Try to merge short segment with adjacent segments
        merged_with_prev = False
        merged_with_next = False

        # Try merging with previous segment
        if merged and (current['start'] - merged[-1]['end']) <= max_gap:
            merged[-1]['text'] = merged[-1]['text'] + ' ' + current['text']
            merged[-1]['end'] = current['end']
            merged_with_prev = True

        # Try merging with next segment (if not merged with previous)
        elif i + 1 < len(segments):
            next_seg = segments[i + 1]
            if (next_seg['start'] - current['end']) <= max_gap:
                # Merge current with next
                merged.append({
                    'text': current['text'] + ' ' + next_seg['text'],
                    'start': current['start'],
                    'end': next_seg['end']
                })
                merged_with_next = True
                i += 1  # Skip next segment as it's already merged

        # If couldn't merge and segment is very short (< 0.5s), skip it
        if not merged_with_prev and not merged_with_next:
            if duration < 0.5:
                print(f"    [Filter] Removed isolated fragment (<0.5s): \"{current['text']}\" at {current['start']:.2f}s")
            else:
                # Keep it if it's between 0.5s and 1s
                merged.append(current)

        i += 1
    return merged
def main():
    parser = argparse.ArgumentParser(
        description='Transcribe audio files with optional chunk merging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process individual chunks (default behavior)
  python transcribe.py --input output/ --output transcripts/ --t 5

  # Merge 30s chunks into 60s transcript files
  python transcribe.py --input output/ --output transcripts/ --t 5 --merge-window 60 --chunk-duration 30

  # Merge with custom settings
  python transcribe.py --input output/ --output transcripts/ --merge-window 120 --chunk-duration 10
        """
    )
    parser.add_argument('--input', required=True, help='Input directory with audio files')
    parser.add_argument('--output', required=True, help='Output directory name')
    parser.add_argument('--t', type=int, default=10, help='Time window in seconds for grouping words (default: 10)')
    parser.add_argument('--max-new-tokens', type=int, default=2048, help='Max new tokens (default: 2048)')
    parser.add_argument('--merge-window', type=int, default=300,
                       help='Merge multiple chunks into larger transcript files (in seconds). Default: 300 (5 minutes)')
    parser.add_argument('--chunk-duration', type=float, default=30.0,
                       help='Duration of each audio chunk (in seconds). Default: 30')
    parser.add_argument('--overlap', type=float, default=2.0,
                       help='Overlap duration in seconds used during chunking. Default: 2 (automatically adjusts timestamps and removes duplicates)')
    parser.add_argument('--fix-fragments', action='store_true', default=True,
                       help='Enable smart merging of short boundary fragments. Default: enabled')
    args = parser.parse_args()

    # Import heavy dependencies only after argparse (allows --help without dependencies)
    import torch
    from qwen_asr import Qwen3ASRModel

    os.makedirs(args.output, exist_ok=True)

    # Validate merge-window arguments
    if args.merge_window and not args.chunk_duration:
        parser.error("--chunk-duration is required when using --merge-window")

    print("Loading Qwen3-ASR model...")
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=32,
        max_new_tokens=args.max_new_tokens,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ),
    )
    print("Model loaded successfully!\n")

    audio_files = [f for f in os.listdir(args.input) if f.endswith(('.wav', '.mp3', '.flac'))]

    if not args.merge_window:
        # Original behavior: process each file individually
        print("Processing mode: Individual files\n")
        for audio_file in audio_files:
            audio_path = os.path.join(args.input, audio_file)
            print(f"Processing: {audio_file}")

            try:
                results = model.transcribe(
                    audio=audio_path,
                    language=None,
                    return_time_stamps=True,
                )

                # Check if transcription succeeded
                if not results or len(results) == 0:
                    print(f"  Warning: No results for {audio_file}, skipping...\n")
                    continue

                if results[0].time_stamps is None or not hasattr(results[0].time_stamps, 'items'):
                    print(f"  Warning: No timestamps for {audio_file} (possibly silent or too short), skipping...\n")
                    continue

                segments = group_by_time_window(results[0].time_stamps.items, window_seconds=args.t)

                if not segments:
                    print(f"  Warning: No segments generated for {audio_file}, skipping...\n")
                    continue

                # Apply fragment merging if enabled
                if args.fix_fragments:
                    original_count = len(segments)
                    segments = merge_short_segments(segments, min_duration=1.0, max_gap=1.0)
                    if len(segments) < original_count:
                        print(f"  [Fragment Fix] {original_count} -> {len(segments)} segments")

                output_file = os.path.join(args.output, f"{os.path.splitext(audio_file)[0]}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    for seg in segments:
                        f.write(f"{seg['text']}: {seg['start']}s - {seg['end']}s\n")

                print(f"  ✓ Saved: {output_file} ({len(segments)} segments)\n")

            except Exception as e:
                print(f"  ✗ Error processing {audio_file}: {e}")
                print(f"  Skipping this file and continuing...\n")
                continue

    else:
        # New behavior: merge multiple chunks
        print(f"Processing mode: Merge chunks")
        print(f"  Chunk duration: {args.chunk_duration}s")
        print(f"  Merge window: {args.merge_window}s\n")

        # Group audio files by base name
        audio_groups = group_audio_files(audio_files)

        for base_name, chunk_list in audio_groups.items():
            print(f"\n{'='*60}")
            print(f"Processing group: {base_name}")
            print(f"Total chunks: {len(chunk_list)}")
            print(f"{'='*60}\n")

            # Collect all transcriptions with global timestamps
            all_segments = []

            for chunk_num, audio_file in chunk_list:
                audio_path = os.path.join(args.input, audio_file)
                print(f"  Transcribing chunk {chunk_num}: {audio_file}")

                try:
                    results = model.transcribe(
                        audio=audio_path,
                        language=None,
                        return_time_stamps=True,
                    )

                    # Check if transcription succeeded
                    if not results or len(results) == 0:
                        print(f"    Warning: No results for {audio_file}, skipping...")
                        continue

                    if results[0].time_stamps is None or not hasattr(results[0].time_stamps, 'items'):
                        print(f"    Warning: No timestamps for {audio_file} (possibly silent or too short), skipping...")
                        continue

                    # Group by time window
                    segments = group_by_time_window(results[0].time_stamps.items, window_seconds=args.t)

                    if not segments:
                        print(f"    Warning: No segments generated for {audio_file}, skipping...")
                        continue

                    # Adjust timestamps to global time (based on chunk number)
                    # Two chunking methods:
                    # 1. Fixed duration (each chunk = 30s): chunk_start = n × (duration - overlap)
                    # 2. Extended duration (chunk n>0 = 32s): chunk_start = n × duration - overlap
                    #
                    # Extended duration is more common (what chunk_audio.py does):
                    #   Chunk 0: 0-30s, Chunk 1: 28-60s (32s), Chunk 2: 58-90s (32s)

                    if args.overlap > 0 and chunk_num > 0:
                        # Extended duration: chunk_start = n × chunk_duration - overlap
                        chunk_start_time = chunk_num * args.chunk_duration - args.overlap
                    else:
                        # No overlap or chunk 0
                        chunk_start_time = chunk_num * args.chunk_duration

                    for seg in segments:
                        seg['start'] += chunk_start_time
                        seg['end'] += chunk_start_time

                        # Mark which chunk this came from for deduplication
                        seg['chunk_num'] = chunk_num

                        all_segments.append(seg)

                    print(f"    ✓ Processed {len(segments)} segments")

                except Exception as e:
                    print(f"    ✗ Error processing {audio_file}: {e}")
                    print(f"    Skipping this chunk and continuing...")
                    continue

            print(f"\n  Total segments before processing: {len(all_segments)}")

            # Remove duplicate segments from overlap regions
            if args.overlap > 0:
                print(f"  Removing duplicates from overlap regions (overlap={args.overlap}s)...")
                all_segments = remove_overlap_duplicates(all_segments, args.chunk_duration, args.overlap)
                print(f"  Total segments after deduplication: {len(all_segments)}")



            # Apply fragment merging if enabled
            if args.fix_fragments:
                print(f"  Applying smart fragment merging...")
                all_segments = merge_short_segments(all_segments, min_duration=1.0, max_gap=1.0)
                print(f"  Total segments after merging: {len(all_segments)}")

            # Merge segments into larger transcript files
            if not all_segments:
                print(f"  Warning: No segments found for {base_name}")
                continue

            # Group segments by merge window
            merged_transcripts = defaultdict(list)
            for seg in all_segments:
                # Segments are grouped by merge window boundaries
                merge_idx = int(seg['start'] // args.merge_window)
                merged_transcripts[merge_idx].append(seg)

            # Write merged transcript files
            for merge_idx in sorted(merged_transcripts.keys()):
                segments = merged_transcripts[merge_idx]

                # Calculate window boundaries
                # First file: 0 to merge_window
                # Second file: merge_window+1 to 2*merge_window
                # etc.
                if merge_idx == 0:
                    window_start = 0
                    header_start = 0
                else:
                    window_start = merge_idx * args.merge_window
                    header_start = int(window_start) + 1  # Start at next second

                window_end = (merge_idx + 1) * args.merge_window
                actual_start = segments[0]['start']
                actual_end = segments[-1]['end']

                output_file = os.path.join(args.output, f"{base_name}_merged_{merge_idx}.txt")

                with open(output_file, 'w', encoding='utf-8') as f:
                    # Write header with clear boundaries
                    if merge_idx == 0:
                        f.write(f"# Merged transcript: 0s to {int(window_end)}s\n")
                    else:
                        f.write(f"# Merged transcript: {header_start}s to {int(actual_end)}s\n")

                    f.write(f"# Actual range: {actual_start:.2f}s - {actual_end:.2f}s\n")
                    f.write(f"# Total segments: {len(segments)}\n\n")

                    # Write segments
                    for seg in segments:
                        f.write(f"{seg['text']}: {seg['start']:.2f}s - {seg['end']:.2f}s\n")

                if merge_idx == 0:
                    print(f"  Saved: {output_file} (0s to {int(window_end)}s, {len(segments)} segments)")
                else:
                    print(f"  Saved: {output_file} ({header_start}s to {int(actual_end)}s, {len(segments)} segments)")

    print(f"\n{'='*60}")
    print("All processing complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji

    main()