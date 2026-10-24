import os
import json
import argparse
import requests

import analyze_content
# Same prompts, scoring and validation as the local analyzer; only the API call differs.
from analyze_content import validate_segment, extract_text_from_transcript, parse_json_response  # noqa: F401

DEFAULT_MODEL = "cogito-2.1:671b"


class ContentAnalyzer(analyze_content.ContentAnalyzer):
    """Analyze transcripts with Ollama cloud models (or ollama.com directly, with an API key)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, ollama_url: str = "http://localhost:11434",
                 api_key: str = None, debug: bool = False):
        """
        Initialize the content analyzer with Ollama (supports cloud models).

        Args:
            model_name: Ollama model identifier (e.g., "kimi-k2.5:cloud", "qwen3-coder-next:cloud")
            ollama_url: URL of the Ollama server (use https://ollama.com for direct cloud access)
            api_key: Ollama API key (required for direct cloud access at https://ollama.com)
            debug: Enable debug output for troubleshooting
        """
        print(f"Connecting to Ollama at {ollama_url}")
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip('/')
        self.api_key = api_key or os.environ.get('OLLAMA_API_KEY')
        self.debug = debug
        self.is_cloud_model = ":cloud" in model_name
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if not self.api_key and "ollama.com" in ollama_url:
            print(f"⚠️  Warning: Using ollama.com but no API key provided!")
            print(f"   Set OLLAMA_API_KEY environment variable or use --api-key")
            print(f"   Get API key from: https://ollama.com/cloud")

        # Test Ollama connection
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", headers=self.headers, timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            if not any(m.get("name") == model_name or m.get("model") == model_name for m in models):
                print(f"⚠️  Warning: Model '{model_name}' not found in Ollama.")
                print(f"Available models: {', '.join([m.get('name', m.get('model', 'unknown')) for m in models[:5]])}")
                print(f"The model will be pulled on first use.")
            print(f"Connected to Ollama. Using {'cloud' if self.is_cloud_model else 'local'} model: {model_name}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama at {ollama_url}: {e}")

    def _chat(self, messages, max_new_tokens, temperature=0.3, top_p=0.9):
        """One /api/chat call with the API key. Big cloud models may think first: they get
        a longer timeout, and a reply left only in "thinking" is used as the answer."""
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            headers=self.headers,
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "top_p": top_p, "num_predict": max_new_tokens},
            },
            timeout=600 if self.is_cloud_model else 300
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise requests.exceptions.RequestException(f"API error: {data['error']}")
        message = data.get("message", {})
        content = message.get("content") or message.get("thinking") or message.get("text") or data.get("response") or ""
        if self.debug:
            print(f"  📥 {len(content)} chars, done_reason={data.get('done_reason')}")
        return content, data.get("done_reason")


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
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help=f'Ollama model to use (default: {DEFAULT_MODEL}). Examples: kimi-k2.5:cloud, qwen3-coder-next:cloud, gemini-3-flash-preview:latest')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                       help='Ollama server URL. Use http://localhost:11434 for local proxy or https://ollama.com for direct cloud access')
    parser.add_argument('--api-key', help='Ollama API key (required for https://ollama.com). Get from https://ollama.com/cloud or set OLLAMA_API_KEY env var')
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
    parser.add_argument('--debug', action='store_true', help='Enable debug output for troubleshooting cloud models')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ContentAnalyzer(
        model_name=args.model,
        ollama_url=args.ollama_url,
        api_key=args.api_key,
        debug=args.debug or ":cloud" in args.model  # Auto-enable debug for cloud models
    )
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
