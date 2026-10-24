#!/usr/bin/env python3
"""
merge_rulebased.py - turn the analysis results into the final clip list

The default `story` strategy keeps each story as cut and joins the same moment
found twice; `smart`, `aggressive` and `conservative` merge short picks using
time gaps, topic matching and confidence scores. Optional steps: snap clip edges
to pauses (--audio), fit long stories with one LLM call (--fit-to) and drop the
show's opening (--drop-openings). run_pipeline.py turns all three on.

Usage:
    python merge_rulebased.py \\
        --input data/analysis_results/ \\
        --output data/analysis_results/merged_rulebased/ \\
        --strategy smart
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from datetime import timedelta
import time


def parse_time(time_str: str) -> float:
    """Parse time string '497.36s' to float 497.36"""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    return float(str(time_str).rstrip('s'))


@dataclass
class MergeConfig:
    """Configuration for rule-based segment merging."""

    # Strategy
    strategy: str = "smart"

    # Tier 1: Aggressive merging (close segments)
    tier1_max_gap: float = 15.0
    tier1_same_topic_only: bool = True

    # Tier 2: Conditional merging (medium gaps)
    tier2_max_gap: float = 45.0
    tier2_decent_max_gap: float = 30.0  # gap limit at tier2_min_confidence; capped at tier2_max_gap
    tier2_min_confidence: float = 0.7
    tier2_high_confidence: float = 0.85

    # Tier 3: Context expansion
    before_context: float = 10.0
    after_context: float = 5.0
    apply_context_expansion: bool = True

    # Safety bounds
    min_segment_duration: float = 10.0
    max_merged_duration: float = 120.0
    max_segments_per_merge: int = 5

    # Advanced
    merge_cross_topic: bool = False
    preserve_original_segments: bool = True


class SegmentMerger:
    """Rule-based intelligent segment merger using tiered heuristics."""

    def __init__(self, config: MergeConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.stats = {
            'tier1_merges': 0,
            'tier2_merges': 0,
            'context_expansions': 0,
            'safety_adjustments': 0
        }

    def merge_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point for segment merging.

        Args:
            segments: List of analysis segments

        Returns:
            List of merged segments with metadata
        """
        if not segments:
            return []

        print(f"\n🔗 Rule-based merging: {len(segments)} segments")
        print(f"   Strategy: {self.config.strategy}")

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: parse_time(s.get('start_time', 0)))

        # Apply tiered merging (story clips are complete already: gluing neighbours makes 90-120 s clips)
        merged = sorted_segments if self.config.strategy == 'story' else self._apply_tiered_merging(sorted_segments)
        print(f"   After merging: {len(merged)} segments")

        # Apply context expansion
        if self.config.apply_context_expansion:
            expanded = self._expand_context(merged)
            print(f"   After context expansion: {len(expanded)} segments")
        else:
            expanded = merged

        # Enforce safety bounds
        final = self._apply_safety_bounds(expanded)
        print(f"   After safety bounds: {len(final)} segments")

        final = self._join_overlaps(final)
        print(f"   After joining overlaps: {len(final)} segments")

        # Print statistics
        if self.verbose:
            print(f"\n   Merge statistics:")
            print(f"     Tier 1 (aggressive): {self.stats['tier1_merges']}")
            print(f"     Tier 2 (conditional): {self.stats['tier2_merges']}")
            print(f"     Context expansions: {self.stats['context_expansions']}")
            print(f"     Safety adjustments: {self.stats['safety_adjustments']}")

        return final

    def _calculate_gap(self, seg_a: Dict[str, Any], seg_b: Dict[str, Any]) -> float:
        """Calculate time gap between two segments in seconds."""
        end_a = parse_time(seg_a.get('end_time', 0))
        start_b = parse_time(seg_b.get('start_time', 0))
        return start_b - end_a

    def _should_merge(self, seg_a: Dict[str, Any], seg_b: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Decide if two segments should merge.

        Returns:
            (should_merge: bool, reason: str)
        """
        gap = self._calculate_gap(seg_a, seg_b)

        # Handle overlapping segments
        if gap < 0:
            return (True, "overlap")

        same_topic = seg_a.get('topic') == seg_b.get('topic')
        conf_a = seg_a.get('confidence', 0)
        conf_b = seg_b.get('confidence', 0)

        # Tier 1: Aggressive merging (0-15s gaps)
        if gap < self.config.tier1_max_gap:
            if same_topic or not self.config.tier1_same_topic_only:
                self.stats['tier1_merges'] += 1
                return (True, "tier1_aggressive")

        # Tier 2: Conditional merging (15-45s gaps)
        if not same_topic and not self.config.merge_cross_topic:
            return (False, "different_topic")

        # Condition 1: gap < 30s (never past tier2_max_gap), same topic, decent confidence
        if gap < min(self.config.tier2_decent_max_gap, self.config.tier2_max_gap) and same_topic and min(conf_a, conf_b) >= self.config.tier2_min_confidence:
            self.stats['tier2_merges'] += 1
            return (True, "tier2_conditional_1")

        # Condition 2: gap < 45s, same topic, high confidence
        if gap < self.config.tier2_max_gap and same_topic:
            if conf_a >= self.config.tier2_high_confidence and conf_b >= self.config.tier2_high_confidence:
                self.stats['tier2_merges'] += 1
                return (True, "tier2_conditional_2")

        return (False, "no_merge")

    def _merge_two_segments(self, seg_a: Dict[str, Any], seg_b: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Merge two segments into one."""
        start_a = parse_time(seg_a.get('start_time', 0))
        end_a = parse_time(seg_a.get('end_time', 0))
        start_b = parse_time(seg_b.get('start_time', 0))
        end_b = parse_time(seg_b.get('end_time', 0))

        end = max(end_a, end_b)  # b can sit inside a (overlap): don't cut a's ending off
        merged = {
            'topic': seg_a.get('topic'),
            'start_time': min(start_a, start_b),
            'end_time': end,
            'duration': end - min(start_a, start_b),
            'confidence': min(seg_a.get('confidence', 0), seg_b.get('confidence', 0)),
            'text': seg_a.get('text', '') + ' ' + seg_b.get('text', ''),
            'reason': seg_a.get('reason', '') + ' | ' + seg_b.get('reason', ''),
            'source_file': seg_a.get('source_file', ''),
            'badge_label': seg_a.get('badge_label', '') or seg_b.get('badge_label', ''),
            'badge_emoji': seg_a.get('badge_emoji', '') or seg_b.get('badge_emoji', ''),

            # Merge metadata
            'merge_metadata': {
                'num_merged': 2,
                'merge_tier': reason,
                'gap_seconds': self._calculate_gap(seg_a, seg_b),
                'original_durations': [end_a - start_a, end_b - start_b]
            }
        }

        # Preserve original segments if configured
        if self.config.preserve_original_segments:
            merged['merged_from'] = [seg_a, seg_b]

        return merged

    def _apply_tiered_merging(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply tiered merging strategy to segments."""
        if not segments:
            return []

        merged = []
        current_group = [segments[0]]
        merge_count = 0

        for i in range(1, len(segments)):
            seg_current = segments[i]
            seg_last = current_group[-1]

            # Check if we should merge
            should_merge, reason = self._should_merge(seg_last, seg_current)

            # Check max segments per merge limit
            if len(current_group) >= self.config.max_segments_per_merge:
                should_merge = False
                reason = "max_segments_limit"

            if should_merge:
                # Add to current group
                current_group.append(seg_current)
                merge_count += 1

                if self.verbose:
                    gap = self._calculate_gap(seg_last, seg_current)
                    print(f"     Merging: {reason}, gap={gap:.1f}s")

            else:
                # Finalize current group
                if len(current_group) > 1:
                    # Merge the group
                    merged_seg = self._merge_group(current_group)
                    merged.append(merged_seg)
                else:
                    # Keep as-is
                    merged.append(current_group[0])

                # Start new group
                current_group = [seg_current]

        # Handle last group
        if len(current_group) > 1:
            merged_seg = self._merge_group(current_group)
            merged.append(merged_seg)
        else:
            merged.append(current_group[0])

        return merged

    def _merge_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a group of segments into one."""
        if len(group) == 1:
            return group[0]

        # Start with first segment
        result = group[0].copy()

        # Merge each subsequent segment
        for seg in group[1:]:
            result = self._merge_two_segments(result, seg, "group_merge")

        # Update metadata for group
        result['merge_metadata']['num_merged'] = len(group)

        return result

    def _expand_context(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add context padding to segments (Tier 3)."""
        expanded = []

        for seg in segments:
            # Skip if already merged (has context from merging)
            if 'merge_metadata' in seg and seg['merge_metadata'].get('num_merged', 1) > 1:
                expanded.append(seg)
                continue

            # Apply context expansion
            start = parse_time(seg.get('start_time', 0))
            end = parse_time(seg.get('end_time', 0))
            original_duration = end - start

            # Add padding
            new_start = max(0, start - self.config.before_context)
            new_end = end + self.config.after_context

            # Update segment
            expanded_seg = seg.copy()
            expanded_seg['original_start'] = start
            expanded_seg['original_end'] = end
            expanded_seg['start_time'] = new_start
            expanded_seg['end_time'] = new_end
            expanded_seg['duration'] = new_end - new_start
            expanded_seg['expansion_metadata'] = {
                'before_context': start - new_start,
                'after_context': new_end - end,
                'original_duration': original_duration
            }

            expanded.append(expanded_seg)
            self.stats['context_expansions'] += 1

            if self.verbose:
                print(f"     Context: {original_duration:.1f}s → {expanded_seg['duration']:.1f}s")

        return expanded

    def _apply_safety_bounds(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enforce safety constraints on segments."""
        final = []

        for seg in segments:
            start = parse_time(seg.get('start_time', 0))
            end = parse_time(seg.get('end_time', 0))
            duration = end - start

            adjusted = False

            # Enforce minimum duration
            if duration < self.config.min_segment_duration:
                # Try to expand symmetrically
                # At the video start the rest goes after. Times are rounded to the transcript's
                # centiseconds: 29.999999999999996 s was under the cutter's 30 s minimum and dropped.
                needed = self.config.min_segment_duration - duration
                new_start = round(max(0.0, start - needed / 2), 2)
                new_end = round(max(end + needed / 2, new_start + self.config.min_segment_duration), 2)

                seg['start_time'] = new_start
                seg['end_time'] = new_end
                seg['duration'] = round(new_end - new_start, 2)
                adjusted = True

                if self.verbose:
                    print(f"     Min enforce: {duration:.1f}s → {seg['duration']:.1f}s")

            # Enforce maximum duration
            elif duration > self.config.max_merged_duration:
                # Trim symmetrically
                excess = duration - self.config.max_merged_duration
                trim_each = excess / 2
                new_start = round(start + trim_each, 2)
                new_end = round(new_start + self.config.max_merged_duration, 2)

                seg['start_time'] = new_start
                seg['end_time'] = new_end
                seg['duration'] = round(new_end - new_start, 2)
                adjusted = True

                if self.verbose:
                    print(f"     Max enforce: {duration:.1f}s → {seg['duration']:.1f}s")

            if adjusted:
                self.stats['safety_adjustments'] += 1
                if 'safety_adjusted' not in seg:
                    seg['safety_adjusted'] = True

            final.append(seg)

        return final

    def _join_overlaps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Join clips that cover the same moment (overlap >= half the shorter one), so it is not posted twice.

        Tiered merging runs before padding and only within a topic, so two picks 20 s apart
        (one "controversial", one "motivational") became two clips sharing ~20 s. A small
        overlap (two stories sharing a boundary line) is left alone: joining would glue two stories.
        """
        out = []
        for seg in sorted(segments, key=lambda s: parse_time(s.get('start_time', 0))):
            start, end = parse_time(seg['start_time']), parse_time(seg['end_time'])
            if out:
                prev = out[-1]
                overlap = min(end, prev['end_time']) - start
                shorter = min(end - start, prev['end_time'] - prev['start_time'])
            if not out or overlap < shorter / 2:
                out.append(dict(seg, start_time=start, end_time=end))
                continue
            stronger = seg if seg.get('confidence', 0) > prev.get('confidence', 0) else prev
            if max(end, prev['end_time']) - prev['start_time'] <= self.config.max_merged_duration:
                out[-1] = dict(stronger, start_time=prev['start_time'], end_time=max(end, prev['end_time']),
                               joined_from=prev.get('joined_from', [prev]) + [seg])
            elif stronger is seg:  # too long to join: keep the stronger clip of the two
                out[-1] = dict(seg, start_time=start, end_time=end)
        for seg in out:
            seg['duration'] = seg['end_time'] - seg['start_time']
        return out

def load_all_segments(analysis_dir: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """Load all segments from analysis JSON files."""
    print(f"\n📊 Loading segments from {analysis_dir}...")

    all_segments = []
    analysis_files = sorted([f for f in os.listdir(analysis_dir) if f.endswith('.json')])

    for filename in analysis_files:
        filepath = os.path.join(analysis_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            segments = data.get('segments', [])

            for seg in segments:
                # Ensure numeric times
                seg['start_time'] = parse_time(seg.get('start_time', 0))
                seg['end_time'] = parse_time(seg.get('end_time', 0))
                seg['source_file'] = filename
                all_segments.append(seg)

    print(f"✓ Loaded {len(all_segments)} segments from {len(analysis_files)} files")

    if verbose and all_segments:
        durations = [parse_time(s['end_time']) - parse_time(s['start_time']) for s in all_segments]
        print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
        print(f"  Average: {sum(durations)/len(durations):.1f}s")

    return all_segments


def save_merged_results(segments: List[Dict[str, Any]],
                        output_dir: str,
                        config: MergeConfig,
                        metadata: Dict[str, Any],
                        filename: str = "merged_segments.json"):
    """Save merged segments with metadata."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Calculate statistics
    durations = [s.get('duration', 0) for s in segments]
    merge_counts = [s.get('merge_metadata', {}).get('num_merged', 1) for s in segments]

    output_data = {
        "pipeline_metadata": metadata,
        "merge_config": asdict(config),
        "statistics": {
            "total_segments": len(segments),
            "duration": {
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "avg": sum(durations) / len(durations) if durations else 0
            },
            "merging": {
                "max_merged_count": max(merge_counts) if merge_counts else 1,
                "total_merges": sum(1 for c in merge_counts if c > 1)
            }
        },
        "segments": segments
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved {len(segments)} merged segments to {output_path}")
    print(f"   Duration: {output_data['statistics']['duration']['avg']:.1f}s avg")
    print(f"   Range: {output_data['statistics']['duration']['min']:.1f}s - {output_data['statistics']['duration']['max']:.1f}s")


def generate_summary(segments: List[Dict[str, Any]], original_count: int, elapsed_time: float) -> str:
    """Generate formatted summary report."""
    if not segments:
        return "No segments generated."

    durations = [s.get('duration', 0) for s in segments]
    duration_bins = {
        '<30s': sum(1 for d in durations if d < 30),
        '30-60s': sum(1 for d in durations if 30 <= d < 60),
        '60-90s': sum(1 for d in durations if 60 <= d < 90),
        '90-120s': sum(1 for d in durations if 90 <= d <= 120),
        '>120s': sum(1 for d in durations if d > 120)
    }

    summary = []
    summary.append("\n" + "="*70)
    summary.append("RULE-BASED MERGING - SUMMARY")
    summary.append("="*70)
    summary.append(f"\n📊 Segments: {original_count} → {len(segments)} ({(1-len(segments)/original_count)*100:.1f}% reduction)")
    summary.append(f"⏱️  Processing time: {timedelta(seconds=int(elapsed_time))}")
    summary.append(f"📏 Duration: {min(durations):.1f}s - {max(durations):.1f}s (avg: {sum(durations)/len(durations):.1f}s)")

    summary.append("\n📊 Duration distribution:")
    for range_label, count in duration_bins.items():
        if count > 0:
            bar = '█' * max(1, count)
            summary.append(f"  {range_label:8s} {bar} {count}")

    summary.append("\n" + "="*70)

    return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Rule-based intelligent segment merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smart merging (balanced)
  python merge_rulebased.py \\
      --input data/analysis_results/ \\
      --output data/analysis_results/merged_rulebased/

  # Aggressive merging (more merges, longer clips)
  python merge_rulebased.py \\
      --input data/analysis_results/ \\
      --output data/analysis_results/merged_aggressive/ \\
      --strategy aggressive \\
      --tier1-gap 20 \\
      --tier2-gap 60

  # Conservative merging (fewer merges, shorter clips)
  python merge_rulebased.py \\
      --input data/analysis_results/ \\
      --output data/analysis_results/merged_conservative/ \\
      --strategy conservative \\
      --tier1-gap 10 \\
      --tier2-gap 30
        """
    )

    parser.add_argument('--input', required=True, help='Input directory with analysis JSON files')
    parser.add_argument('--output', required=True, help='Output directory for merged segments')

    # Strategy presets
    parser.add_argument('--strategy', choices=['story', 'smart', 'aggressive', 'conservative'], default='story',
                       help='Merging strategy preset (default: story)')

    # Tier 1 options
    parser.add_argument('--tier1-gap', type=float, help='Tier 1 max gap (default: 15s)')
    parser.add_argument('--no-tier1-topic-filter', action='store_true', help='Allow tier 1 cross-topic merging')

    # Tier 2 options
    parser.add_argument('--tier2-gap', type=float, help='Tier 2 max gap (default: 45s)')
    parser.add_argument('--tier2-min-confidence', type=float, help='Tier 2 min confidence (default: 0.7)')

    # Context expansion
    parser.add_argument('--before-context', type=float, help='Seconds before segment (default: 10s)')
    parser.add_argument('--after-context', type=float, help='Seconds after segment (default: 5s)')
    parser.add_argument('--no-context', action='store_true', help='Disable context expansion')

    # Safety bounds
    parser.add_argument('--min-duration', type=float, help='Minimum segment duration (default: 10s)')
    parser.add_argument('--max-duration', type=float, help='Maximum segment duration (default: 120s)')
    parser.add_argument('--max-merge-count', type=int, help='Max segments per merge (default: 5)')

    parser.add_argument('--verbose', action='store_true', help='Show detailed progress')
    parser.add_argument('--audio', help='Source video: move clip edges into the nearest pause (cached as <output>/pauses.json)')

    parser.add_argument('--fit-to', type=float, help='Clips longer than this (s) are cut by the LLM to the run of their lines that keeps setup and payoff, not trimmed; needs --transcripts')
    parser.add_argument('--drop-openings', action='store_true', help="Ask the LLM whether each clip in the first 5 min is the show's opening (greeting, introducing the guest or topic) and drop those; needs --transcripts")
    parser.add_argument('--transcripts', help='Transcript .txt files directory (for --fit-to / --drop-openings)')
    parser.add_argument('--model', default='gemma4:12b-it-q4_K_M', help='Ollama model for --fit-to')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama server URL for --fit-to')

    args = parser.parse_args()
    if args.fit_to and not args.transcripts:
        parser.error('--fit-to needs --transcripts')
    if args.drop_openings and not args.transcripts:
        parser.error('--drop-openings needs --transcripts')

    print("="*70)
    print("SEGMENT MERGING")
    print("="*70)

    # Build config
    config = MergeConfig(strategy=args.strategy)

    # Apply strategy presets
    if args.strategy == 'aggressive':
        config.tier1_max_gap = 20.0
        config.tier2_max_gap = 60.0
        config.max_merged_duration = 150.0
    elif args.strategy == 'story':
        # analyze_content cuts whole stories (hook -> payoff): keep them as cut, no padding
        config.apply_context_expansion = False
    elif args.strategy == 'conservative':
        config.tier1_max_gap = 10.0
        config.tier2_max_gap = 30.0
        config.max_merged_duration = 90.0

    # Override with CLI arguments
    if args.tier1_gap is not None:
        config.tier1_max_gap = args.tier1_gap
    if args.tier2_gap is not None:
        config.tier2_max_gap = args.tier2_gap
    if args.tier2_min_confidence is not None:
        config.tier2_min_confidence = args.tier2_min_confidence
    if args.before_context is not None:
        config.before_context = args.before_context
    if args.after_context is not None:
        config.after_context = args.after_context
    if args.no_context:
        config.apply_context_expansion = False
    if args.min_duration is not None:
        config.min_segment_duration = args.min_duration
    if args.max_duration is not None:
        config.max_merged_duration = args.max_duration
    if args.max_merge_count is not None:
        config.max_segments_per_merge = args.max_merge_count
    if args.no_tier1_topic_filter:
        config.tier1_same_topic_only = False

    start_time = time.time()

    # Load segments
    segments = load_all_segments(args.input, args.verbose)

    if not segments:
        print("\n⚠️  No segments found. Nothing to merge.")
        return 0

    original_count = len(segments)

    # Create merger
    merger = SegmentMerger(config, args.verbose)

    # Merge segments
    merged_segments = merger.merge_segments(segments)
    # snap at the merge's own limit (120 s), not --fit-to: clips snapping pushes over --fit-to are then fitted by the LLM
    if args.audio:  # line edges often fall mid-word ("ber|tindak" at 14:30 on the test podcast)
        from audio_cues import snap_all
        snap_all(merged_segments, args.audio, args.output, config.max_merged_duration)
    lines = None
    if args.fit_to or args.drop_openings:
        from fit_window import drop_openings, fit_long, load_lines, ollama_chat
        lines = load_lines(args.transcripts)
    if args.fit_to:
        from audio_cues import cached, detect_pauses
        pauses = cached(os.path.join(args.output, "pauses.json"), lambda: detect_pauses(args.audio), args.audio) if args.audio else []
        merged_segments = fit_long(merged_segments, lines, pauses,
                                   ollama_chat(args.model, args.ollama_url), config.min_segment_duration, args.fit_to)
    if args.drop_openings:  # after the fit, on the fitted clips
        merged_segments = drop_openings(merged_segments, lines, ollama_chat(args.model, args.ollama_url))

    elapsed_time = time.time() - start_time

    # Generate summary
    summary = generate_summary(merged_segments, original_count, elapsed_time)
    print(summary)

    # Save results
    metadata = {
        "pipeline_version": "rulebased_v1",
        "processing_time_seconds": elapsed_time,
        "original_segments": original_count,
        "merged_segments": len(merged_segments),
        "reduction_percent": (1 - len(merged_segments)/original_count) * 100 if original_count > 0 else 0
    }

    save_merged_results(merged_segments, args.output, config, metadata)

    print("\n✓ MERGING COMPLETE!")
    return 0


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    exit(main())
