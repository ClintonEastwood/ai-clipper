# AI Clipper

Turns a long podcast video into short vertical clips (30-60 s) that follow the speaker, with
captions and a hook title. One command, and everything runs locally on a single 12 GB GPU.

```bash
python scripts/run_pipeline.py --input input/podcast.mp4 --topics funny controversial motivational
```

Output: `output/clips_final/*.mp4`, 1080x1920, ready for Shorts, Reels or TikTok.

## How it works

| Step | What happens | Script |
|---|---|---|
| 1-2 | Split the audio into 30 s chunks and transcribe with timestamps (Qwen3-ASR + forced aligner) | `chunk_audio.py`, `transcribe.py` |
| 3 | An LLM reads the numbered transcript lines and cuts whole stories (hook to payoff). Python copies the text from the transcript, so quotes are always verbatim. A second call scores each story (hook, coherence, connection, trend); weak clips are dropped | `analyze_content.py` |
| 4 | Join duplicates, move cut points into the nearest pause, fit stories longer than 60 s into a 30-60 s run of their own lines (one LLM call), drop the show's opening, rank by score (`--audio-rank` also weighs laughter and loudness) | `merge_rulebased.py`, `fit_window.py`, `rank_clips.py` |
| 5 | Cut the clips with FFmpeg | `video_cutter.py` |
| 6 | Reframe to 9:16 on the active speaker with TalkNet, per camera shot: one speaker, split screen for two people, or a blurred-background fallback | `auto_reframe.py` |
| 7 | Word-timed captions; the spoken word turns yellow | `add_subtitles.py` |
| 8 | An LLM writes a hook title per clip, shown on a white card for the first 4.5 s | `add_badge.py` |

Finished steps are skipped on a rerun, so an interrupted run resumes where it stopped.

## Result

A blind test against OpusClip on 2 Indonesian podcasts (46 and 41 min), judged by 3 friends, each
voting separately and blind, 20 clip pairs each:

| | AI Clipper | OpusClip | Tie |
|---|---|---|---|
| Podcast 1 | 13 | 15 | 2 |
| Podcast 2 | 14 | 12 | 4 |
| **Total (60 votes)** | **27** | **27** | **6** |

- Rank against rank: each tool's #1 clip against the other's #1, down to #10.
- Viewers did not know which tool made which clip, and the A/B side was random per pair.
- Both clips in a pair were cut from the same video and rendered by the same plain FFmpeg
  command (no captions or titles), so only the choice of moment was compared.
- The pass rule was fixed before voting: AI Clipper gets at least as many votes as OpusClip.

The test is small; read it as on par, not better.

## Requirements

- An NVIDIA GPU with CUDA. The speech model runs on `cuda:0`, so the pipeline does not run on CPU
  or Apple Silicon. 12 GB is enough for the default model.
- Python 3.10+, FFmpeg and git on the PATH.
- [Ollama](https://ollama.com) 0.30.9 or newer for the local LLM.

## Install

```bash
git clone https://github.com/Rembr4nt/ai-clipper.git
cd ai-clipper
python -m venv .venv --system-site-packages   # keeps a preinstalled CUDA PyTorch visible
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                # install the CUDA build of torch first (pytorch.org)
ollama pull gemma4:12b-it-q4_K_M               # default model, 8 GB
```

The first run downloads Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B (~6.5 GB) from Hugging Face,
and the first reframe sets up TalkNet-ASD on its own (clone, weights, NumPy 2 patch; see
`scripts/setup_talknet.py`). Keep the project on a path without spaces: TalkNet's FFmpeg calls
don't quote paths.

## Usage

```bash
# Everything, local model
python scripts/run_pipeline.py --input input/podcast.mp4 --topics funny controversial

# Cloud model through Ollama
python scripts/run_pipeline.py --input input/podcast.mp4 --topics funny --cloud --model cogito-2.1:671b

# Settings from a file: copy pipeline.yaml.example to pipeline.yaml first
python scripts/run_pipeline.py --config pipeline.yaml
```

| Flag | Effect |
|---|---|
| `--language` | Content language (default: Indonesian) |
| `--min-confidence` | Clip score threshold, 0-1 (default: 0.67) |
| `--max-duration` | Longest clip in seconds (default: 60) |
| `--vertical-mode pad` | Whole frame on a blurred background instead of following the speaker |
| `--skip-vertical` | Landscape clips only (no reframe, captions or title) |
| `--caption-style normal` | Plain white subtitles instead of the yellow highlight |
| `--skip-subtitles`, `--skip-badge` | Leave out captions or the hook title |
| `--from N`, `--force` | Start at step N, or rerun every step |
| `--dry-run` | Print the commands without running them |

Each step is also a script you can run on its own; `python scripts/<script>.py --help` lists its options.

## Project layout

```
scripts/      the pipeline (run_pipeline.py runs steps 1-8)
tests/        unit tests (pytest)
assets/fonts/ Luckiest Guy (captions) and Montserrat Bold (titles), with their licenses
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

The tests run on CPU and mock the LLM, Ollama and git. Two tests in `test_audio_cues.py` call the
real `ffmpeg`.

## Limitations

- Tested on Indonesian podcasts only. The speech model is multilingual, but the prompts were
  tuned on Indonesian.
- Transcripts are made from 30 s chunks without overlap, so a clip sometimes starts at a chunk
  edge, mid-sentence.
- Clip choice comes from an LLM, so results vary between runs and models.

## Credits

- [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) (Tao et al., ACM MM 2021) and the S3FD
  face detector, downloaded at setup and not redistributed here.
- [Qwen3-ASR and Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) for
  speech-to-text and word timestamps.
- [AST AudioSet classifier](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) for laughter.
- Gemma via [Ollama](https://ollama.com), and [FFmpeg](https://ffmpeg.org).
- Fonts: Luckiest Guy (Apache-2.0) and Montserrat (SIL OFL 1.1).

Model weights and fonts keep their own licenses.

## License

MIT, see [LICENSE](LICENSE).
