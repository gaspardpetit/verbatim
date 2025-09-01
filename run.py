import sys

from verbatim.main import main

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
    "--txt",
    "--md",
    "--json",
    "--docx",
    "--ass",
]

# Call the main function
main()
