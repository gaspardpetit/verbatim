import os
import logging
from typing import Optional

LOG = logging.getLogger(__name__)


def find_reference_file(audio_filename: str, reference_dir: str = "tests/data/ground_truth") -> Optional[str]:
    """
    Recursively search for a matching reference file in the reference directory.

    Args:
        audio_filename: The name of the audio file being transcribed
        reference_dir: Directory to search for reference files

    Returns:
        Path to the best matching reference file, or None if no match found
    """
    if not os.path.exists(reference_dir):
        LOG.warning(f"Reference directory {reference_dir} does not exist")
        return None

    # Extract base name without extension
    audio_base = os.path.splitext(os.path.basename(audio_filename))[0]

    # Dictionary to store matches and their match length
    matches = {}

    # Recursively find all JSON files in the reference directory
    for root, _, files in os.walk(reference_dir):
        for file in files:
            if file.endswith(".json"):
                ref_base = os.path.splitext(file)[0]

                # Find the longest common substring
                common_substring = ""
                for i in range(len(audio_base)):
                    for j in range(i + 5, len(audio_base) + 1):  # At least 5 characters
                        substr = audio_base[i:j]
                        if substr in ref_base:
                            if len(substr) > len(common_substring):
                                common_substring = substr

                if len(common_substring) >= 5:
                    matches[os.path.join(root, file)] = len(common_substring)

    # Get the file with the longest match
    if matches:
        best_match = max(matches.items(), key=lambda x: x[1])
        LOG.info(f"Found best matching reference file: {best_match[0]} with {best_match[1]} matching characters")
        return best_match[0]

    LOG.warning(f"No matching reference file found for {audio_filename}")
    return None
