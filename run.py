import sys
from verbatim.main import main

sys.argv = [
    "run.py",
    "samples/Airfrance - Bienvenue à bord.wav",
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
