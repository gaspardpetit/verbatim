import sys

from verbatim.main import main

# Use caller-provided arguments when present; otherwise run the default demo with pyannote diarization.
if len(sys.argv) == 1:
    sys.argv = [
        "run.py",
        "ext/samples/audio/1ch_2spk_en-fr_AirFrance_00h03m54s.wav",
        "--eval",
        "ext/samples/truth/2spk_en-fr_AirFrance.json",
        "-v",
        "-w",
        "out",
        "--language",
        "en",
        "fr",
        "--diarize",
        "--diarization-strategy",
        "pyannote",
        "--txt",
        "--md",
        "--json",
        "--docx",
        "--ass",
    ]

# Call the main function
main()
