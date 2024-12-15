import sys
from verbatim.__main__ import main

sys.argv = ['run.py', 'samples/voices.wav', '-w', 'out', '--language', 'en', 'fr', '--txt', '--md', '--json', '--docx', '--ass']

# Call the main function
main()
