import os
import json
import argparse
import re
from typing import List, Dict, Any
import requests

def extract_text_from_transcript(transcript_content: str, start_time: str, end_time: str) -> str:
    """
    Extract exact text from transcript for a time range. No LLM involved.

    Args:
        transcript_content: Full transcript text
        start_time: Start time (e.g., "188.16s")
        end_time: End time (e.g., "207.84s")

    Returns:
        Concatenated exact text from matching segments
    """
    start = float(start_time.rstrip('s'))
    end = float(end_time.rstrip('s'))
    # Include a line if it overlaps the requested range
    return " ".join(text for text, s, e in transcript_lines(transcript_content) if e > start and s < end)


def transcript_lines(transcript_content: str) -> List[tuple]:
    """(text, start, end) for each "text: 1.00s - 9.60s" line, sorted, chunk overlaps de-duplicated."""
    lines = set()
    for line in transcript_content.strip().split('\n'):
        match = re.match(r'^(.+):\s*([\d.]+)s\s*-\s*([\d.]+)s\s*$', line)
        if match and not line.startswith('#'):
            lines.add((match.group(1).strip(), float(match.group(2)), float(match.group(3))))
    return sorted(lines, key=lambda l: (l[1], l[2]))


def overlapping_windows(input_dir: str, files: List[str], window: float = 300.0, overlap: float = 60.0) -> List[tuple]:
    """(name, transcript) per window over all files' lines, each sharing `overlap` s with the next,
    so a story that crosses a file boundary (every 5 min) is seen whole by one call."""
    if not 0 <= overlap < window:
        raise ValueError(f"window overlap must be 0 to under {window:.0f} s, got {overlap}")
    lines = set()
    for name in files:
        with open(os.path.join(input_dir, name), encoding='utf-8') as f:
            lines.update(transcript_lines(f.read()))
    lines = sorted(lines, key=lambda l: (l[1], l[2]))
    out, start, i = [], 0.0, 0
    while lines and start < lines[-1][2]:
        chunk = [l for l in lines if start <= l[1] < start + window]
        if chunk:
            out.append((f"window_{i}.txt", "\n".join(f"{t}: {s:.2f}s - {e:.2f}s" for t, s, e in chunk)))
        start += window - overlap
        i += 1
    return out


def parse_json_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse JSON array from LLM response.

    Args:
        response_text: Raw response text from LLM

    Returns:
        Parsed list of segments
    """
    try:
        # Try to extract JSON from response
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            results = json.loads(json_str)
            return results
        else:
            print("Warning: No JSON array found in response")
            print(f"Raw response: {response_text}")
            return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response_text}")
        return []


def validate_segment(segment: Dict[str, Any], min_text_length: int = 50) -> tuple[bool, List[str]]:
    """
    Validate if a segment's reason matches its extracted text.

    Args:
        segment: Segment dictionary with topic, reason, text, etc.
        min_text_length: Minimum text length in characters

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    text = segment.get("text", "").lower()
    reason = segment.get("reason", "")
    issues = []

    # Extract quoted terms from reason (these should be in the text).
    # An apostrophe inside a word ("speaker's") is not a quote mark.
    quoted_terms = re.findall(r"""(?<!\w)["'“‘]([^"'“”‘’]+)["'”’](?!\w)""", reason)

    # Check if quoted terms appear in the text
    missing_terms = []
    for term in quoted_terms:
        # Skip very short quoted terms (articles, pronouns, etc.)
        if len(term) > 3 and term.lower() not in text:
            missing_terms.append(term)

    if missing_terms:
        issues.append(f"Quoted terms not found in text: {', '.join(missing_terms)}")

    # Check minimum text length
    if len(text.strip()) < min_text_length:
        issues.append(f"Text too short ({len(text)} chars, minimum {min_text_length})")

    return len(issues) == 0, issues


class ContentAnalyzer:
    """Analyze transcript content using Ollama API to identify specific topics/moments."""

    edge_check = False  # --edge-check: a second call per clip moves its edges (see check_edges)

    def __init__(self, model_name: str = "gemma4:12b-it-q4_K_M", ollama_url: str = "http://localhost:11434"):
        """
        Initialize the content analyzer with Ollama.

        Args:
            model_name: Ollama model identifier
            ollama_url: URL of the Ollama server
        """
        print(f"Connecting to Ollama at {ollama_url}")
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip('/')

        # Test Ollama connection
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            print(f"Connected to Ollama. Using model: {model_name}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama at {ollama_url}: {e}")

    def _chat(self, messages, max_new_tokens, temperature=0.3, top_p=0.9):
        """One Ollama /api/chat call. Returns (content, done_reason)."""
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                # Hybrid models (qwen3:14b) think first otherwise; thinking counts toward
                # num_predict and cut answers off at 2048. Same picks without it, 5x faster.
                "think": False,
                # Ollama's default 4096 context is near-full for a 5-min chunk plus the clip list,
                # and it drops old tokens silently when full. One size for every call: a change reloads the model.
                "options": {"temperature": temperature, "top_p": top_p, "num_predict": max_new_tokens, "num_ctx": 8192},
            },
            timeout=300  # 5 minute timeout for long transcripts
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"], data.get("done_reason")

    def create_analysis_prompt(self, transcript: str, topics: List[str], language: str = "Indonesian") -> List[Dict[str, str]]:
        """
        Ask for whole stories, cut the way short-form editors cut them: hook -> payoff, 30-60 s.

        Lines are numbered and the model answers with line numbers. Asked for timestamps, it
        copied one ~10 s line per pick (the payoff), and no padding of that line found where the
        story starts. Line numbers fixed that.

        Args:
            transcript: The transcript text with timestamps
            topics: List of topics to find (e.g., ["funny", "controversial", "insightful"])
            language: Language of the content

        Returns:
            List of message dicts for Ollama API
        """
        topics_str = ", ".join(topics)
        numbered = "\n".join(f"[{i}] ({start:.0f}s) {text}" for i, (text, start, _) in enumerate(transcript_lines(transcript)))

        article = "an" if language[:1].lower() in "aeiou" else "a"
        system_prompt = f"""You are a short-form video editor. Below is {article} {language} podcast transcript; each line has a number and its start time.
Cut it into standalone clips for TikTok/Reels. Each clip is one complete story or exchange that makes sense on its own:
it starts at the hook (a question, a bold claim or the start of a story) and ends after the payoff (the answer, punchline or reaction).
Each clip is 30-60 seconds (usually 3-6 lines). Clips must not overlap. Skip filler, greetings, ads and small talk.
Score each clip 0-10:
- "hook": do the first seconds grab attention
- "flow": does it make sense alone, start to end
- "value": how strong the payoff is ({topics_str}, or useful)
Return ONLY a JSON array: [{{"start_line": <n>, "end_line": <m>, "topic": "<one of: {topics_str}>", "title": "<short title in {language}>", "hook": <0-10>, "flow": <0-10>, "value": <0-10>}}]"""

        # Keep this prompt as tested: asking for the transcript "about <topics>" pulled the model back to
        # topic-hunting, and asking for the badge here too cut worse clips
        user_prompt = numbered

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def badge(self, segment: Dict[str, Any], language: str = "Indonesian", max_chars: int = 28) -> Dict[str, str]:
        """Badge label + emoji for one clip, asked on its own (see create_analysis_prompt).

        No example label or emoji list here: asked about one clip at a time, the model copied
        them (6/34 labels were the example, 27/34 emojis the third one listed).
        """
        messages = [{"role": "system", "content": f"""Write the badge for a short clip cut from a podcast. The clip's working title and transcript are below.
- "badge_label": a headline of 3-5 words, at most {max_chars} characters (the badge cuts longer ones), saying what
  happens in THIS clip with its names and key words: who does what. Specific, never generic: it must not
  fit any other clip. Write it in {language}.
- "badge_emoji": the one emoji that best fits this clip's content
Output ONLY JSON: {{"badge_label": "<label>", "badge_emoji": "<emoji>"}}"""},
                    {"role": "user", "content": f"Working title: {segment.get('title', '')}\n\n{segment['text']}"}]
        title = str(segment.get("title", ""))
        try:
            content, _ = self._chat(messages, 60, temperature=0.3)
            badge = json.loads(re.search(r"\{[^{}]*\}", content).group(0))
            label, emoji = str(badge["badge_label"]).strip(), str(badge["badge_emoji"]).strip()
            if len(label) > max_chars:  # models can't count characters, but can count words: ask once
                messages += [{"role": "assistant", "content": content}, {"role": "user", "content":
                             f"Too long for the badge ({len(label)} of {max_chars} characters). Rewrite badge_label in at most 4 short words, keep the names. Same JSON."}]
                content, _ = self._chat(messages, 60, temperature=0.3)
                label = str(json.loads(re.search(r"\{[^{}]*\}", content).group(0))["badge_label"]).strip()
            if len(label) > max_chars and 0 < len(title) <= max_chars:
                label = title
            return {"badge_label": label, "badge_emoji": emoji}
        except (requests.exceptions.RequestException, AttributeError, KeyError, TypeError, ValueError):
            return {"badge_label": title, "badge_emoji": "🔥"}

    def judge(self, text: str, language: str = "Indonesian") -> Any:
        """
        Score one clip on its own, like OpusClip's judge (hook, coherence, connection, trend).

        The cutting call scores clips against the rest of their 5-minute chunk; this one
        compares across the whole video. Returns 0-1, or None if the reply is unusable.
        """
        messages = [{"role": "system", "content": f"""You judge short clips cut from {"an" if language[:1].lower() in "aeiou" else "a"} {language} podcast, for TikTok/Reels. The clip's transcript is below.
Score it 0-100 on each:
- "hook": do the first seconds make people stop scrolling?
- "coherence": does it make sense on its own, with a clear start and ending?
- "connection": does it make viewers feel something (shock, laughter, anger, inspiration) or care?
- "trend": is it about something people are talking about right now?
Most clips score 40-70. Give 85+ only to clips that would clearly go viral.
Output ONLY JSON: {{"hook": <n>, "coherence": <n>, "connection": <n>, "trend": <n>}}"""},
                    {"role": "user", "content": text}]
        try:
            content, _ = self._chat(messages, 80, temperature=0.1)
            scores = json.loads(re.search(r"\{[^{}]*\}", content).group(0))
            return sum(min(100.0, max(0.0, float(scores[k]))) for k in ("hook", "coherence", "connection", "trend")) / 400
        except (requests.exceptions.RequestException, AttributeError, KeyError, TypeError, ValueError):
            return None

    def check_edges(self, lines, first, last, language="Indonesian", reach=3):
        """Second look at one clip's edges: start where the setup starts, end right after the payoff.

        Shows `reach` lines either side. On the test podcast the cutting call picked the right
        moments but started on the reaction (14:00 instead of 13:30) or stopped one line before
        the payoff (15:59). Returns (first, last), unchanged when the reply is unusable.
        """
        lo, hi = max(0, first - reach), min(len(lines) - 1, last + reach)
        shown = "\n".join(f"[{i}] {'>' if first <= i <= last else ' '} {lines[i][0]}" for i in range(lo, hi + 1))
        messages = [{"role": "system", "content": f"""You check the edges of one clip cut from {"an" if language[:1].lower() in "aeiou" else "a"} {language} podcast transcript. Lines marked > are the clip; the lines around it are shown too.
- "start_line": the line where this story's setup starts (the question, claim or quote the rest answers). A viewer must understand the first line without anything before it.
- "end_line": the line with the payoff (answer, punchline or reaction). Stop there; leave out lines that move to a new topic.
Keep the clip 30-60 seconds. Output ONLY JSON: {{"start_line": <n>, "end_line": <m>}}"""},
                    {"role": "user", "content": shown}]
        try:
            content, _ = self._chat(messages, 40, temperature=0.1)
            reply = json.loads(re.search(r"\{[^{}]*\}", content).group(0))
            new_first, new_last = int(reply["start_line"]), int(reply["end_line"])
        except (requests.exceptions.RequestException, AttributeError, KeyError, TypeError, ValueError):
            return first, last
        return (new_first, new_last) if lo <= new_first <= new_last <= hi else (first, last)

    def analyze(
        self,
        transcript: str,
        topics: List[str],
        language: str = "Indonesian",
        max_new_tokens: int = 4096,
        temperature: float = 0.3,
        top_p: float = 0.9,
        validate: bool = True,
        min_confidence: float = 0.67,
        min_text_length: int = 50,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Cut the transcript into story clips, score each one, keep the best and give them badges.

        Args:
            transcript: The transcript text with timestamps
            topics: List of topics to find
            language: Language of the content
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            validate: Whether to validate and filter results (default: True)
            min_confidence: Minimum score to keep, 0-1 (default: 0.67, about the better half of the clips)
            min_text_length: Minimum text length in chars (default: 50)

        Returns:
            List of identified segments with metadata
        """
        messages = self.create_analysis_prompt(transcript, topics, language)

        try:
            content, done_reason = self._chat(messages, max_new_tokens, temperature, top_p)
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return []
        if done_reason == "length":
            print(f"  ⚠️  Reply cut off at {max_new_tokens} tokens; picks from this chunk may be lost (raise --max-new-tokens)")

        # Hybrid step: the model gives line numbers, the times and text come from the transcript
        lines = transcript_lines(transcript)
        results = []
        for clip in parse_json_response(content):
            try:
                first, last = int(clip["start_line"]), int(clip["end_line"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= first <= last < len(lines):
                continue
            if self.edge_check:
                first, last = self.check_edges(lines, first, last, language)
            start, end = f"{lines[first][1]:.2f}s", f"{lines[last][2]:.2f}s"
            results.append(dict(clip, start_time=start, end_time=end,
                                text=extract_text_from_transcript(transcript, start, end)))

        # Score: the cutting call's own scores (0-10) averaged with a judge call per clip
        removed_segments = []
        if validate:
            kept = []
            for segment in results:
                is_valid, issues = validate_segment(segment, min_text_length)
                if is_valid:
                    kept.append(segment)
                else:
                    removed_segments.append({"segment": segment, "reason": "; ".join(issues)})
            results = kept
        for segment in results:
            try:
                own = sum(min(10.0, max(0.0, float(segment.get(k, 0)))) for k in ("hook", "flow", "value")) / 30
            except (TypeError, ValueError):
                own = 0.0
            judged = self.judge(segment["text"], language)
            segment["confidence"] = round(own if judged is None else (own + judged) / 2, 3)
        if validate:
            removed_segments += [{"segment": seg, "reason": f"Low confidence: {seg['confidence']}"}
                                 for seg in results if seg["confidence"] < min_confidence]
            results = [seg for seg in results if seg["confidence"] >= min_confidence]
        for segment in results:
            segment.update(self.badge(segment, language))

        if removed_segments:
            low = sum(item["reason"].startswith("Low confidence") for item in removed_segments)
            print(f"  Kept {len(results)}/{len(results) + len(removed_segments)} clips "
                  f"({low} under --min-confidence {min_confidence}, {len(removed_segments) - low} failed checks)")
            if verbose:
                print("  Filtered clips:")
                for item in removed_segments:
                    seg = item["segment"]
                    print(f"    - {seg.get('start_time')} to {seg.get('end_time')}: {item['reason']}")
        return results

    def analyze_transcript_file(
        self,
        transcript_path: str,
        topics: List[str],
        language: str = "Indonesian",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Analyze a transcript file and identify segments.

        Args:
            transcript_path: Path to transcript file
            topics: List of topics to find
            language: Language of the content
            **kwargs: Additional arguments for analyze()

        Returns:
            List of identified segments with metadata
        """
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()

        return self.analyze(transcript, topics, language, **kwargs)

    def analyze_directory(
        self,
        input_dir: str,
        output_dir: str,
        topics: List[str],
        language: str = "Indonesian",
        window_overlap: float = 0.0,
        **kwargs
    ) -> None:
        """
        Analyze every transcript in a directory. Results replace earlier *_analysis.json there:
        the merge step reads every JSON in the directory, so another run's results would be merged too.

        window_overlap > 0: analyze 300 s windows over all files' lines, each sharing that many seconds
        with the next. A story copied into two windows is joined back into one by the merge step.
        """
        os.makedirs(output_dir, exist_ok=True)
        stale = [f for f in os.listdir(output_dir) if f.endswith('_analysis.json')]
        for name in stale:
            os.remove(os.path.join(output_dir, name))
        if stale:
            print(f"Removed {len(stale)} earlier *_analysis.json from {output_dir}")

        files = sorted(f for f in os.listdir(input_dir) if f.endswith('.txt'))
        if window_overlap > 0:
            jobs = overlapping_windows(input_dir, files, overlap=window_overlap)
        else:
            jobs = []
            for name in files:
                with open(os.path.join(input_dir, name), encoding='utf-8') as f:
                    jobs.append((name, f.read()))

        for name, transcript in jobs:
            print(f"\nAnalyzing: {name}")
            results = self.analyze(transcript, topics, language, **kwargs)
            output_file = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_analysis.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'source_file': name,
                    'topics_searched': topics,
                    'language': language,
                    'segments_found': len(results),
                    'segments': results
                }, f, ensure_ascii=False, indent=2)
            print(f"Found {len(results)} segments matching topics: {', '.join(topics)}")
            print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze transcripts to find specific topics/moments using Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find funny moments in a single transcript
  python analyze_content.py --input transcripts/podcast_merged_0.txt --topics funny --output results/

  # Find multiple types of content
  python analyze_content.py --input transcripts/ --topics funny controversial insightful --output results/

  # Analyze with custom parameters
  python analyze_content.py --input transcripts/ --topics motivational --language Indonesian --temperature 0.3
        """
    )

    parser.add_argument('--input', required=True, help='Input transcript file or directory')
    parser.add_argument('--output', required=True, help='Output directory for analysis results')
    parser.add_argument('--topics', nargs='+', required=True,
                       help='Topics to find (e.g., funny controversial insightful motivational dramatic)')
    parser.add_argument('--language', default='Indonesian', help='Language of the content (default: Indonesian)')
    parser.add_argument('--model', default='gemma4:12b-it-q4_K_M', help='Ollama model to use (default: gemma4:12b-it-q4_K_M)')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--max-new-tokens', type=int, default=4096, help='Max tokens to generate (default: 4096)')
    parser.add_argument('--temperature', type=float, default=0.3, help='Sampling temperature (default: 0.3)')
    parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling parameter (default: 0.9)')

    # Validation options
    parser.add_argument('--no-validate', action='store_true', help='Disable validation filtering (keep all results)')
    parser.add_argument('--min-confidence', type=float, default=0.67, help='Minimum clip score, 0-1 (default: 0.67)')
    parser.add_argument('--min-text-length', type=int, default=50, help='Minimum text length in chars (default: 50)')
    parser.add_argument('--verbose', action='store_true', help='Show details about filtered segments')
    parser.add_argument('--edge-check', action='store_true',
                        help='Second LLM call per clip: start at the setup, end at the payoff')
    parser.add_argument('--window-overlap', type=float, default=0.0,
                        help='Directory input: analyze 300 s windows overlapping by this many seconds (default: 0, one call per file)')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ContentAnalyzer(model_name=args.model, ollama_url=args.ollama_url)
    analyzer.edge_check = args.edge_check

    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Analyze single file
        results = analyzer.analyze_transcript_file(
            args.input,
            args.topics,
            args.language,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            validate=not args.no_validate,
            min_confidence=args.min_confidence,
            min_text_length=args.min_text_length,
            verbose=args.verbose
        )

        # Save results
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, f"{os.path.splitext(os.path.basename(args.input))[0]}_analysis.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'source_file': args.input,
                'topics_searched': args.topics,
                'language': args.language,
                'segments_found': len(results),
                'validation_enabled': not args.no_validate,
                'segments': results
            }, f, ensure_ascii=False, indent=2)

        print(f"\nFound {len(results)} valid segments matching topics: {', '.join(args.topics)}")
        print(f"Saved to: {output_file}")

    elif os.path.isdir(args.input):
        # Analyze directory
        analyzer.analyze_directory(
            args.input,
            args.output,
            args.topics,
            args.language,
            window_overlap=args.window_overlap,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            validate=not args.no_validate,
            min_confidence=args.min_confidence,
            min_text_length=args.min_text_length,
            verbose=args.verbose
        )

    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1

    print("\nAnalysis complete!")
    return 0


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows: piped/redirected output is cp1252 and crashes on ✓/emoji
    exit(main())
