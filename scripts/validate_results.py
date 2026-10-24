#!/usr/bin/env python3
"""
Validate and filter analysis results to remove false positives.
Checks if the LLM's reasoning actually matches the extracted text.
"""

import json
import os
import re
from typing import List, Dict, Any
import argparse


def extract_quoted_terms(text: str) -> List[str]:
    """Extract terms in quotes from text."""
    # Match single and double quotes
    patterns = [
        r"(?<!\w)'([^']+)'(?!\w)",  # Single quotes, not apostrophes ("speaker's")
        r'"([^"]+)"',  # Double quotes
    ]
    terms = []
    for pattern in patterns:
        terms.extend(re.findall(pattern, text))
    return terms


def validate_segment(segment: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """
    Validate if a segment's reason matches its extracted text.

    Args:
        segment: Segment dictionary with topic, reason, text, etc.
        strict: If True, require quoted terms to appear in text

    Returns:
        Dictionary with validation results
    """
    text = segment.get("text", "").lower()
    reason = segment.get("reason", "")

    # Extract quoted terms from reason (these should be in the text)
    quoted_terms = extract_quoted_terms(reason)

    validation = {
        "valid": True,
        "issues": [],
        "quoted_terms": quoted_terms,
        "missing_terms": []
    }

    # Check if quoted terms appear in the text
    for term in quoted_terms:
        if term.lower() not in text:
            validation["valid"] = False
            validation["missing_terms"].append(term)
            validation["issues"].append(f"Quoted term '{term}' not found in text")

    # Check minimum text length (very short segments are suspicious)
    if len(text.strip()) < 30:
        if strict:
            validation["valid"] = False
        validation["issues"].append(f"Text too short ({len(text)} chars)")

    return validation


def filter_results(
    input_file: str,
    output_file: str = None,
    strict: bool = True,
    min_confidence: float = 0.7,
    min_text_length: int = 50
) -> Dict[str, Any]:
    """
    Filter analysis results to remove false positives.

    Args:
        input_file: Path to JSON analysis file
        output_file: Path to save filtered results (optional)
        strict: Strict validation mode
        min_confidence: Minimum confidence score
        min_text_length: Minimum text length in characters

    Returns:
        Filtered results dictionary
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_count = len(data.get("segments", []))
    filtered_segments = []
    removed_segments = []

    for segment in data.get("segments", []):
        # Validate reasoning
        validation = validate_segment(segment, strict=strict)

        # Apply filters
        keep = True
        removal_reasons = []

        if not validation["valid"]:
            keep = False
            removal_reasons.extend(validation["issues"])

        if segment.get("confidence", 0) < min_confidence:
            keep = False
            removal_reasons.append(f"Low confidence: {segment.get('confidence')}")

        if len(segment.get("text", "")) < min_text_length:
            keep = False
            removal_reasons.append(f"Text too short: {len(segment.get('text', ''))} chars")

        if keep:
            filtered_segments.append(segment)
        else:
            removed_segments.append({
                "segment": segment,
                "removal_reasons": removal_reasons,
                "validation": validation
            })

    # Update results
    filtered_data = data.copy()
    filtered_data["segments"] = filtered_segments
    filtered_data["segments_found"] = len(filtered_segments)
    filtered_data["original_segments_found"] = original_count
    filtered_data["segments_removed"] = len(removed_segments)
    filtered_data["filtering"] = {
        "strict": strict,
        "min_confidence": min_confidence,
        "min_text_length": min_text_length
    }

    # Save filtered results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved filtered results to: {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Validation Summary: {os.path.basename(input_file)}")
    print(f"{'='*60}")
    print(f"Original segments: {original_count}")
    print(f"Valid segments: {len(filtered_segments)}")
    print(f"Removed segments: {len(removed_segments)}")
    print(f"{'='*60}\n")

    if removed_segments:
        print("❌ Removed Segments:")
        for i, item in enumerate(removed_segments, 1):
            seg = item["segment"]
            print(f"\n{i}. {seg.get('start_time')} - {seg.get('end_time')}")
            print(f"   Topic: {seg.get('topic')}")
            print(f"   Confidence: {seg.get('confidence')}")
            print(f"   Reasons for removal:")
            for reason in item["removal_reasons"]:
                print(f"     - {reason}")
            if item["validation"]["missing_terms"]:
                print(f"   Missing terms: {', '.join(item['validation']['missing_terms'])}")

    return filtered_data


def validate_directory(
    input_dir: str,
    output_dir: str = None,
    strict: bool = True,
    min_confidence: float = 0.7,
    min_text_length: int = 50
):
    """Validate all JSON files in a directory."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(input_dir) if f.endswith('_analysis.json')]

    print(f"\n{'='*60}")
    print(f"Validating {len(json_files)} files from {input_dir}")
    print(f"{'='*60}\n")

    total_original = 0
    total_filtered = 0
    total_removed = 0

    for json_file in sorted(json_files):
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file) if output_dir else None

        result = filter_results(
            input_path,
            output_path,
            strict=strict,
            min_confidence=min_confidence,
            min_text_length=min_text_length
        )

        total_original += result.get("original_segments_found", 0)
        total_filtered += result.get("segments_found", 0)
        total_removed += result.get("segments_removed", 0)

    print(f"\n{'='*60}")
    print(f"Overall Summary")
    print(f"{'='*60}")
    print(f"Total original segments: {total_original}")
    print(f"Total valid segments: {total_filtered}")
    print(f"Total removed segments: {total_removed}")
    if total_original:  # no segments at all used to crash with ZeroDivisionError
        print(f"Removal rate: {total_removed/total_original*100:.1f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and filter analysis results to remove false positives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single file
  python validate_results.py --input data/analysis_results/podcast_merged_0_analysis.json

  # Validate and save filtered version
  python validate_results.py --input results/file.json --output filtered/file.json

  # Validate entire directory
  python validate_results.py --input-dir data/analysis_results/ --output-dir filtered_results/

  # Relaxed filtering (less strict)
  python validate_results.py --input-dir results/ --no-strict --min-confidence 0.6
        """
    )

    parser.add_argument('--input', help='Input JSON file')
    parser.add_argument('--output', help='Output JSON file (filtered)')
    parser.add_argument('--input-dir', help='Input directory with JSON files')
    parser.add_argument('--output-dir', help='Output directory for filtered files')
    parser.add_argument('--no-strict', action='store_true', help='Use relaxed validation')
    parser.add_argument('--min-confidence', type=float, default=0.7,
                       help='Minimum confidence score (default: 0.7)')
    parser.add_argument('--min-text-length', type=int, default=50,
                       help='Minimum text length in chars (default: 50)')

    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Must specify --input or --input-dir")

    strict = not args.no_strict

    if args.input_dir:
        validate_directory(
            args.input_dir,
            args.output_dir,
            strict=strict,
            min_confidence=args.min_confidence,
            min_text_length=args.min_text_length
        )
    else:
        filter_results(
            args.input,
            args.output,
            strict=strict,
            min_confidence=args.min_confidence,
            min_text_length=args.min_text_length
        )


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    main()
