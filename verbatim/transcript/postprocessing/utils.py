from typing import List, Dict

import termcolor


def color_diff_line(line: str) -> str:
    """Color diff lines based on their prefix"""
    if line.startswith("^"):
        return termcolor.colored(line, "yellow")  # or any color you prefer for structural changes
    return line


def format_chunk_for_display(utterances: List[Dict]) -> str:
    """Format utterances as raw diarized text"""
    return " ".join([f"<speaker:{utt['hyp_spk']}> {utt['hyp_text']}" for utt in utterances])
