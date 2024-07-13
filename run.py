import sys
from verbatim.__main__ import main

# Simulate command-line arguments
sys.argv = [
    'run.py',                # The name of the script (ignored by argparse)
    'tests/data/init.mp3',   # The input argument
    '-o', 'out/',            # The output argument
    '-l', 'en', 'fr',        # The languages argument
    '-n', '2',               # The number of speakers argument
    '-b', '12',              # The number of beams argument
    '-v',                    # Increase verbosity
    #'--cpu',                 # Use CPU
    '--transcribe_only',     # Transcribe only
]

# Call the main function
main()
