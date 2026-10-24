#!/usr/bin/env python3
"""
Improved audio chunking script with automatic duration detection.
Handles overlaps correctly for transcribe.py merging.
"""

import subprocess
import os
import argparse

def get_audio_duration(file_path):
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())

def chunk_audio(input_file, output_dir, chunk_length=30, overlap=0, reencode=True):
    """
    Chunk audio file with optional overlap.

    Args:
        input_file: Path to input audio file
        output_dir: Directory to save chunks
        chunk_length: Length of each chunk in seconds (for new content)
        overlap: Overlap between chunks in seconds
        reencode: If True, re-encode for precision. If False, use -c copy (faster but less precise)

    Chunking strategy:
        - First chunk: 0 to chunk_length (no overlap)
        - Subsequent chunks: include overlap + chunk_length
        Example with chunk_length=30, overlap=2:
          Chunk 0: 0-30s (30s)
          Chunk 1: 28-60s (32s, includes 2s overlap)
          Chunk 2: 58-90s (32s, includes 2s overlap)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get duration automatically
    print(f"Analyzing {input_file}...")
    duration = get_audio_duration(input_file)
    print(f"Duration: {duration:.2f}s ({int(duration//60)}:{int(duration%60):02d})")

    # Step size: how much we advance each chunk (chunk_length for new content)
    step_size = chunk_length

    # Estimate number of chunks
    estimated_chunks = int((duration - chunk_length) / step_size) + 1
    print(f"Chunk length: {chunk_length}s")
    print(f"Overlap: {overlap}s")
    print(f"Step size: {step_size}s (advances by chunk_length)")
    print(f"Estimated chunks: {estimated_chunks}\n")

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    chunk_num = 0
    failed = []

    while True:
        output_file = os.path.join(output_dir, f"{base_name}_{chunk_num}.wav")

        # Calculate start and end times
        if chunk_num == 0:
            # First chunk: 0 to chunk_length
            start = 0
            end_time = min(chunk_length, duration)
        else:
            # Subsequent chunks: start overlap seconds before previous end
            start = chunk_num * chunk_length - overlap
            end_time = min((chunk_num + 1) * chunk_length, duration)

        # Check if we're done
        if start >= duration:
            break

        actual_duration = end_time - start

        # Build ffmpeg command
        cmd = ["ffmpeg", "-y", "-i", input_file, "-ss", str(start), "-t", str(actual_duration)]

        if reencode:
            # Re-encode for precise cuts
            cmd.extend(["-c:a", "pcm_s16le"])  # Use uncompressed WAV for best quality
        else:
            # Fast copy (less precise)
            cmd.extend(["-c", "copy"])

        cmd.append(output_file)

        print(f"Creating chunk {chunk_num}: {start:.2f}s to {end_time:.2f}s ({actual_duration:.2f}s)")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Warning: Error creating chunk {chunk_num}")
            print(result.stderr)
            failed.append(chunk_num)

        chunk_num += 1

    if failed:
        # Raising (not just warning) so run_pipeline doesn't mark the step done
        raise RuntimeError(f"{len(failed)}/{chunk_num} audio chunks failed: {failed}")

    print(f"\n✅ Created {chunk_num} chunks in {output_dir}/")

    # Print important info for transcribe.py
    print(f"\n{'='*60}")
    print("To transcribe and merge these chunks, use:")
    print(f"{'='*60}")
    print(f"python scripts/transcribe.py \\")
    print(f"  --input {output_dir}/ \\")
    print(f"  --output transcripts/ \\")
    print(f"  --chunk-duration {chunk_length} \\")
    print(f"  --overlap {overlap}")
    print(f"\nNote: --chunk-duration is {chunk_length} (step size between chunks)")
    print(f"      Each chunk includes {overlap}s overlap for context continuity")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji

    parser = argparse.ArgumentParser(
        description="Chunk audio file with optional overlap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 30-second chunks with 2-second overlap
  python chunk_audio.py --input podcast.wav --output audio_chunks --chunk-length 30 --overlap 2

  # 10-second chunks with 1-second overlap
  python chunk_audio.py --input audio.wav --output chunks --chunk-length 10 --overlap 1

  # Fast mode (no re-encoding, less precise)
  python chunk_audio.py --input podcast.wav --output audio_chunks --no-reencode
        """
    )

    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--chunk-length", type=int, default=30, help="Length of each chunk in seconds (default: 30)")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap between chunks in seconds (default: 2)")
    parser.add_argument("--no-reencode", action="store_true", help="Use fast copy mode (less precise)")

    args = parser.parse_args()

    chunk_audio(
        args.input,
        args.output,
        args.chunk_length,
        args.overlap,
        reencode=not args.no_reencode
    )