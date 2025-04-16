# Usage

Verbatim can be used through its command-line interface, as a Python library, or via Docker.

```{toctree}
---
maxdepth: 2
---
cli
python_api
```

## Basic Usage Examples

### Command Line

Simple usage:

```bash
verbatim audio_file.mp3
```

With verbose output:

```bash
verbatim audio_file.mp3 -v
```

Force CPU only:

```bash
verbatim audio_file.mp3 --cpu
```

Save file in a specific directory:

```bash
verbatim audio_file.mp3 -o ./output/
```

### Docker

With GPU support:

```bash
docker run --network none --shm-size 8G --gpus all \
    -v "/local/path/to/out/:/data/out/" \
    -v "/local/path/to/audio.mp3:/data/audio.mp3" ghcr.io/gaspardpetit/verbatim:latest \
    verbatim /data/audio.mp3 -o /data/out --languages en fr
```

Without GPU support:

```bash
docker run --network none \
    -v "/local/path/to/out/:/data/out/" \
    -v "/local/path/to/audio.mp3:/data/audio.mp3" ghcr.io/gaspardpetit/verbatim:latest \
    verbatim /data/audio.mp3 -o /data/out --languages en fr
```

### Python API

Basic usage:

```python
from verbatim import Context, Pipeline

context = Context(
    languages=["en", "fr"],
    nb_speakers=2,
    source_file="audio.mp3",
    out_dir="out"
)

pipeline = Pipeline(context=context)
pipeline.execute()
```
```

### usage/cli.md

```markdown
# Command Line Interface

## Overview

Verbatim's command-line interface provides a flexible way to transcribe audio files with various options for
controlling the transcription process, speaker diarization, and output formats.

## Basic Syntax

```bash
verbatim [input] [options]
```

If no input file is provided, Verbatim expects input from stdin or can use the microphone.

## Arguments

### Positional Arguments

- `input` (optional): Path to the input audio file.

  - Use `-` to read from stdin (16-bit 16kHz mono PCM stream)
  - Use `>` to use the microphone

### File Processing

| Option | Description |
| ------ | ----------- |
| `-f, --from <timestamp>` | Start time within the file (hh:mm:ss.ms or mm:ss.ms) |
| `-t, --to <timestamp>` | Stop time within the file (hh:mm:ss.ms or mm:ss.ms) |
| `-o, --outdir <path>` | Path to the output directory |
| `-w, --workdir <path>` | Set the working directory for temporary files |
| `-e, --eval <file>` | Path to a reference JSON file for evaluation |

### Diarization & Speaker Handling

| Option | Description |
| ------ | ----------- |
| `--diarization-strategy <strategy>` | Select diarization strategy (`pyannote` or `stereo`, default: `pyannote`) |
| `-d, --diarization <file>` | Identify speakers using an RTTM file |
| `--separate` | Enable speaker voice separation |
| `-n, --diarize <num>` | Number of speakers in the audio file |

### Language & Transcription Settings

| Option | Description |
| ------ | ----------- |
| `-l, --languages <lang1 lang2 ...>` | Specify languages for speech recognition |
| `--format-timestamp <style>` | Set timestamp format (`none`, `start`, `range`, `minute`) |
| `--format-speaker <style>` | Set speaker format (`none`, `change`, `always`) |
| `--format-probability <style>` | Set probability format (`none`, `line`, `line_75`, `line_50`, `line_25`, `word`, `word_75`, `word_50`, `word_25`) |
| `--format-language <style>` | Set language format (`none`, `change`, `always`) |
| `-b, --nb_beams <num>` | Number of parallel processing beams (default: 9, range: 1-15) |

### Output Format Options

| Option | Description |
| ------ | ----------- |
| `--ass` | Enable ASS subtitle file output |
| `--docx` | Enable Microsoft Word DOCX output |
| `--txt` | Enable plain text output |
| `--json` | Enable JSON file output |
| `--md` | Enable Markdown output |
| `--stdout` | Enable stdout output (default: enabled) |
| `--stdout-nocolor` | Enable stdout output without colors |

### Performance & Debugging

| Option | Description |
| ------ | ----------- |
| `--cpu` | Force CPU usage |
| `-s, --stream` | Enable low latency streaming mode |
| `-v, --verbose` | Increase verbosity (use multiple times for more detail) |
| `-i, --isolate` | Extract voices from background noise |

### Version & Help

| Option | Description |
| ------ | ----------- |
| `--version` | Display the version and exit |
| `-h, --help` | Show help message and exit |

## Examples

### Basic Transcription

```bash
verbatim input.wav --txt --json -o output/
```

### Transcribe with Speaker Diarization

```bash
verbatim input.wav --diarization diarization.rttm --json
```

### Transcribe with Language Selection

```bash
verbatim input.wav -l en fr --txt
```

### Stream Audio from Microphone

```bash
verbatim '>' --stream --stdout
```

### Save Output in Multiple Formats

```bash
verbatim input.wav --md --json --docx
```

### Adjusting Processing Performance

```bash
verbatim input.wav -b 12 --cpu
```
