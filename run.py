import sys
from verbatim.main import main

sys.argv = [
    "run.py",
    "samples/1ch_2spk_en-fr_AirFrance_00h03m54s.m4a", "--eval", "tests/data/ground_truth/2spk_en-fr_AirFrance.json",
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
