#!/usr/bin/env python3
"""
Gap Analysis Tool - Identify potential segment merge candidates

Analyzes gaps between adjacent segments to find opportunities for merging
segments that are close together into more coherent clips.
"""

import json
import argparse
from typing import List, Dict, Any
from pathlib import Path


def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def analyze_gaps(segments: List[Dict[str, Any]], max_gap: float = 30.0, max_duration: float = 120.0) -> List[Dict[str, Any]]:
    """
    Analyze gaps between adjacent segments

    Args:
        segments: List of segment dictionaries
        max_gap: Maximum gap to consider (seconds)
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

        # Check if it's a good merge candidate
        if gap <= max_gap and combined_duration <= max_duration:
            candidates.append({
                'index': i,
                'curr_index': i,
                'next_index': i + 1,
                'curr_start': curr['start_time'],
                'curr_end': curr['end_time'],
                'curr_duration': curr['duration'],
                'curr_topic': curr['topic'],
                'curr_text': curr['text'],
                'next_start': next_seg['start_time'],
                'next_end': next_seg['end_time'],
                'next_duration': next_seg['duration'],
                'next_topic': next_seg['topic'],
                'next_text': next_seg['text'],
                'gap': gap,
                'combined_duration': combined_duration,
                'same_topic': curr['topic'] == next_seg['topic']
            })

    return candidates


def print_summary(candidates: List[Dict[str, Any]]) -> None:
    """Print summary of merge candidates"""
    print("\n" + "=" * 80)
    print("GAP ANALYSIS SUMMARY - MERGE CANDIDATES")
    print("=" * 80)

    # Sort by gap (smallest first)
    sorted_candidates = sorted(candidates, key=lambda x: x['gap'])

    print(f"\nFound {len(sorted_candidates)} potential merge candidates:\n")

    # Print top candidates
    for idx, cand in enumerate(sorted_candidates, 1):
        gap_str = f"Gap {cand['gap']:+.2f}s" if cand['gap'] >= 0 else f"Gap {cand['gap']:.2f}s (overlapping!)"
        star = "⭐" if cand['gap'] < 5.0 else ""
        same_topic_marker = "🎯" if cand['same_topic'] else "🔀"

        print(f"{idx:2d}. Segment #{cand['curr_index']} → #{cand['next_index']}: "
              f"{gap_str}, Combined {cand['combined_duration']:.2f}s {star} {same_topic_marker}")

    print("\n" + "=" * 80)
    print("Legend:")
    print("  ⭐ = Excellent candidate (gap < 5s)")
    print("  🎯 = Same topic (better semantic coherence)")
    print("  🔀 = Different topics (may be less coherent)")
    print("=" * 80)


def print_detailed_report(candidates: List[Dict[str, Any]], top_n: int = 10) -> None:
    """Print detailed report of top N candidates"""
    sorted_candidates = sorted(candidates, key=lambda x: x['gap'])[:top_n]

    print("\n" + "=" * 80)
    print(f"DETAILED REPORT - TOP {top_n} MERGE CANDIDATES")
    print("=" * 80)

    for idx, cand in enumerate(sorted_candidates, 1):
        print(f"\n{'─' * 80}")
        print(f"CANDIDATE #{idx}: Segments #{cand['curr_index']} → #{cand['next_index']}")
        print(f"{'─' * 80}")

        # Current segment
        print(f"\n📌 SEGMENT #{cand['curr_index']} (Current)")
        print(f"   Topic: {cand['curr_topic']}")
        print(f"   Time: {cand['curr_start']:.2f}s - {cand['curr_end']:.2f}s ({format_time(cand['curr_start'])} - {format_time(cand['curr_end'])})")
        print(f"   Duration: {cand['curr_duration']:.2f}s")
        print(f"   Text: {cand['curr_text'][:100]}...")

        # Gap
        if cand['gap'] >= 0:
            print(f"\n   ⏸️  GAP: {cand['gap']:.2f} seconds")
        else:
            print(f"\n   ⚠️  OVERLAP: {abs(cand['gap']):.2f} seconds (segments overlap!)")

        # Next segment
        print(f"\n📌 SEGMENT #{cand['next_index']} (Next)")
        print(f"   Topic: {cand['next_topic']}")
        print(f"   Time: {cand['next_start']:.2f}s - {cand['next_end']:.2f}s ({format_time(cand['next_start'])} - {format_time(cand['next_end'])})")
        print(f"   Duration: {cand['next_duration']:.2f}s")
        print(f"   Text: {cand['next_text'][:100]}...")

        # Combined stats
        print(f"\n🔗 COMBINED RESULT:")
        print(f"   Total Duration: {cand['combined_duration']:.2f}s ({format_time(cand['combined_duration'])})")
        print(f"   Same Topic: {'Yes 🎯' if cand['same_topic'] else 'No 🔀'}")

        quality = "Excellent" if cand['gap'] < 5 else "Good" if cand['gap'] < 15 else "Fair"
        print(f"   Quality Rating: {quality}")

        if cand['combined_duration'] > 90:
            print(f"   ⚠️  Note: Combined duration is long ({cand['combined_duration']:.2f}s)")


def export_candidates(candidates: List[Dict[str, Any]], output_file: str,
                      max_gap: float = 30.0, max_duration: float = 120.0) -> None:
    """Export candidates to JSON file"""
    sorted_candidates = sorted(candidates, key=lambda x: x['gap'])

    export_data = {
        'total_candidates': len(sorted_candidates),
        'analysis_params': {
            'max_gap': max_gap,
            'max_duration': max_duration
        },
        'candidates': []
    }

    for cand in sorted_candidates:
        export_data['candidates'].append({
            'curr_segment_index': cand['curr_index'],
            'next_segment_index': cand['next_index'],
            'gap_seconds': round(cand['gap'], 2),
            'combined_duration': round(cand['combined_duration'], 2),
            'same_topic': cand['same_topic'],
            'curr_topic': cand['curr_topic'],
            'next_topic': cand['next_topic'],
            'curr_time_range': f"{cand['curr_start']:.2f}s - {cand['curr_end']:.2f}s",
            'next_time_range': f"{cand['next_start']:.2f}s - {cand['next_end']:.2f}s",
            'quality': 'excellent' if cand['gap'] < 5 else 'good' if cand['gap'] < 15 else 'fair'
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Exported {len(sorted_candidates)} candidates to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze gaps between segments to find merge candidates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python scripts/gap_analysis.py --input data/analysis_results/merged_optimized/merged_segments.json

  # Show detailed report for top 10
  python scripts/gap_analysis.py --input merged_segments.json --detailed --top 10

  # Export candidates to JSON
  python scripts/gap_analysis.py --input merged_segments.json --export candidates.json

  # Filter for very close segments only
  python scripts/gap_analysis.py --input merged_segments.json --max-gap 5
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Input merged segments JSON file')
    parser.add_argument('--max-gap', type=float, default=30.0,
                        help='Maximum gap to consider (seconds, default: 30)')
    parser.add_argument('--max-duration', type=float, default=120.0,
                        help='Maximum combined duration (seconds, default: 120)')
    parser.add_argument('--detailed', '-d', action='store_true',
                        help='Show detailed report')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of top candidates to show in detailed report (default: 10)')
    parser.add_argument('--export', '-e',
                        help='Export candidates to JSON file')

    args = parser.parse_args()

    # Load segments
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data.get('segments', [])

    if not segments:
        print("❌ Error: No segments found in input file")
        return

    print(f"\n📊 Analyzing {len(segments)} segments from: {input_file}")
    print(f"   Max gap: {args.max_gap}s")
    print(f"   Max combined duration: {args.max_duration}s")

    # Analyze gaps
    candidates = analyze_gaps(segments, args.max_gap, args.max_duration)

    # Print summary
    print_summary(candidates)

    # Print detailed report if requested
    if args.detailed:
        print_detailed_report(candidates, args.top)

    # Export if requested
    if args.export:
        export_candidates(candidates, args.export, args.max_gap, args.max_duration)

    # Statistics
    excellent = sum(1 for c in candidates if c['gap'] < 5)
    good = sum(1 for c in candidates if 5 <= c['gap'] < 15)
    fair = sum(1 for c in candidates if c['gap'] >= 15)
    same_topic = sum(1 for c in candidates if c['same_topic'])

    print("\n📈 STATISTICS:")
    print(f"   Total candidates: {len(candidates)}")
    print(f"   Excellent (gap < 5s): {excellent}")
    print(f"   Good (gap 5-15s): {good}")
    print(f"   Fair (gap > 15s): {fair}")
    print(f"   Same topic: {same_topic} ({100*same_topic/len(candidates) if candidates else 0:.1f}%)")

    print("\n💡 TIP: Use --detailed flag to see full text and more info")
    print("💡 TIP: Use --export to save candidates as JSON for automation")


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    main()
