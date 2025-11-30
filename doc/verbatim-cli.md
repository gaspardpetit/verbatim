# Verbatim Command Line Interface Documentation

## Overview
Verbatim is a command-line tool for transcribing and processing audio files. It supports various output formats, speaker diarization, language recognition, and noise isolation.

## Usage
```
verbatim [input] [options]
```

If no input file is provided, Verbatim expects input from stdin or can use the microphone.

## Arguments

### Positional Arguments
- `input` (optional): Path to the input audio file.
  - Use `-` to read from stdin (16-bit 16kHz mono PCM stream)
  - Use `>` to use the microphone

### Optional Arguments

#### File Processing
- `-f, --from <timestamp>`: Start time within the file (hh:mm:ss.ms or mm:ss.ms)
- `-t, --to <timestamp>`: Stop time within the file (hh:mm:ss.ms or mm:ss.ms)
- `-o, --outdir <path>`: Path to the output directory
- `-w, --workdir <path>`: Set the working directory for temporary files
- `-e, --eval <file>`: Path to a reference JSON file for evaluation

#### Diarization & Speaker Handling
- `--diarization-strategy <strategy>`: Select diarization strategy (`pyannote` or `stereo`, default: `pyannote`)
- `--vttm <file>`: VTTM diarization manifest to consume (preferred). If omitted, a minimal VTTM is created and filled if diarization runs.
- `-d, --diarization <file>`: (Deprecated) RTTM file; wrapped into VTTM for processing.
- `--separate`: Enable speaker voice separation
- `-n, --diarize <num>`: Number of speakers in the audio file

#### Language & Transcription Settings
- `-l, --languages <lang1 lang2 ...>`: Specify languages for speech recognition
- `--format-timestamp <style>`: Set timestamp format (`none`, `start`, `range`, `minute`)
- `--format-speaker <style>`: Set speaker format (`none`, `change`, `always`)
- `--format-probability <style>`: Set probability format (`none`, `line`, `line_75`, `line_50`, `line_25`, `word`, `word_75`, `word_50`, `word_25`)
- `--format-language <style>`: Set language format (`none`, `change`, `always`)
- `-b, --nb_beams <num>`: Number of parallel processing beams (default: 9, range: 1-15)

#### Output Format Options
- `--ass`: Enable ASS subtitle file output
- `--docx`: Enable Microsoft Word DOCX output
- `--txt`: Enable plain text output
- `--json`: Enable JSON file output
- `--md`: Enable Markdown output
- `--stdout`: Enable stdout output (default: enabled)
- `--stdout-nocolor`: Enable stdout output without colors

#### Performance & Debugging
- `--cpu`: Force CPU usage
- `-s, --stream`: Enable low latency streaming mode
- `-v, --verbose`: Increase verbosity (use multiple times for more detail)
- `-i, --isolate`: Extract voices from background noise

#### Server
- `--serve`: Start an HTTP server with OpenAI-compatible `/audio/transcriptions` and `/models` endpoints
  - Use `-F stream=true` in your request to receive Server-Sent Events

#### Version & Help
- `--version`: Display the version and exit
- `-h, --help`: Show help message and exit

## Examples

### Basic Transcription
```
verbatim input.wav --txt --json -o output/
```

### Transcribe with Speaker Diarization (VTTM)
```
verbatim input.wav --vttm diarization.vttm --json
```

### Transcribe with Language Selection
```
verbatim input.wav -l en fr --txt
```

### Stream Audio from Microphone
```
verbatim '>' --stream --stdout
```

### Save Output in Multiple Formats
```
verbatim input.wav --md --json --docx
```

### Adjusting Processing Performance
```
verbatim input.wav -b 12 --cpu
```

## Exit Codes
- `0`: Success

