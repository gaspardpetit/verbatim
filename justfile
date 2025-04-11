# Just file for verbatim
outdir := "./out"

# Default recipe
default:
	@just --list

# Transcribe all files at path
transcribe-dir inpath outpath lang diarization_strategy num_speakers:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Transcribing audio files in '{{inpath}}'..."
    source .venv/bin/activate

    shopt -s nullglob
    for file in {{inpath}}/*.wav {{inpath}}/*.m4a; do
        echo "Transcribing $file..."
        verbatim  --outdir {{outpath}} \
            --languages {{lang}} \
            --diarize {{num_speakers}} \
            --diarization-strategy {{diarization_strategy}} \
            --json \
            "$file" \
            --eval
    done

transcribe-dir-both-dia inpath outpath lang num_speakers:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Transcribing audio files in '{{inpath}}'..."
    source .venv/bin/activate

    # Enable nullglob option to expand to empty string if no matches found
    shopt -s nullglob

    # Tell me which files are being processed
    # for file in {{inpath}}/*.wav {{inpath}}/*.m4a; do
    for file in {{inpath}}/*.wav; do
        echo "Processing $file with stereo diarization"
    done
    # for file in {{inpath}}/*.wav {{inpath}}/*.m4a; do
    #     echo "Processing $file with pyannote diarization"
    # done

    # Process files with stereo diarization
    # for file in {{inpath}}/*.wav {{inpath}}/*.m4a; do
    for file in {{inpath}}/*.wav; do
        echo "Transcribing $file... with stereo diarization"
        verbatim --outdir {{outpath}} \
            --languages {{lang}} \
            --diarize {{num_speakers}} \
            --diarization-strategy stereo \
            --json \
            "$file" \
            --eval
    done

    # Process files with pyannote diarization
    # for file in {{inpath}}/*.wav {{inpath}}/*.m4a; do
    #     echo "Transcribing $file... with pyannote diarization"
    #     verbatim --outdir {{outpath}} \
    #         --languages {{lang}} \
    #         --diarize {{num_speakers}} \
    #         --diarization-strategy pyannote \
    #         --json \
    #         "$file" \
    #         --eval
    # done


eval hyppath:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob

    # Use the find command to locate all JSON files including in subdirectories
    # Sort them alphabetically
    find "{{hyppath}}" -type f -name "*.json" | sort | while read -r hypfile; do
        # Extract the filename from the path
        filename=$(basename "$hypfile")
        echo
        echo
        echo "Caclulating metrics for $filename..."

        # Extract the pattern part (e.g., 2spk_de_108_01) using regex
        # This pattern looks for digits followed by "spk_" followed by language code
        # followed by underscore, digits, underscore, and digits
        pattern_part=$(echo "$filename" | grep -o "[0-9]\+spk_[a-z]\+_[0-9]\+_[0-9]\+")

        # Construct refpath
        reffile="tests/data/ground_truth/private/$pattern_part.utt.json"

        echo "Reference file ---- $reffile"
        echo "Hypothesis file --- $hypfile"

        source .venv/bin/activate
        uv run python verbatim/eval/cli.py $reffile $hypfile
    done
