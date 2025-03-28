import datetime
import json
import logging
import os
from typing import Any, Optional, List
from dataclasses import asdict, is_dataclass

from verbatim.eval.compare import Metrics

# Configure logger
LOG = logging.getLogger(__name__)


class MetricsEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles dataclass objects."""

    def default(self, o):
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        return super().default(o)


def log_transcription(
    source_path: str,
    output_prefix_no_ext: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    languages: Optional[List[str]] = None,
    diarization_strategy: Optional[str] = None,
    num_speakers: Optional[int] = None,
    metrics: Optional[Metrics] = None,
):
    """
    Log transcription details to a log file.

    Args:
        source_path: Path to the audio file that was transcribed
        output_prefix_no_ext: Prefix for output files
        start_time: Time when transcription started
        end_time: Time when transcription ended
        languages: List of languages used for transcription
        diarization_strategy: Strategy used for diarization
        num_speakers: Number of speakers if specified
        metrics: Optional evaluation metrics if --eval was provided
    """
    log_dir = os.path.dirname(output_prefix_no_ext)
    log_file = os.path.join(log_dir, "transcription_log.json")

    # Format the log entry
    duration = (end_time - start_time).total_seconds()
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "audio_file": source_path,
        "output_prefix": output_prefix_no_ext,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "languages": languages,
        "diarization_strategy": diarization_strategy,
    }

    # Add number of speakers if provided
    if num_speakers is not None:
        log_entry["num_speakers"] = num_speakers

    # Add metrics if available
    if metrics:
        log_entry["metrics"] = metrics

    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Read existing entries or initialize new array
    entries = []
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except json.JSONDecodeError:
            LOG.warning(f"Could not parse existing log file {log_file}, creating new file")
            entries = []

    # Add new entry
    entries.append(log_entry)

    # Write the complete array back to the file
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, cls=MetricsEncoder, indent=2)

    LOG.info(f"Transcription logged to {log_file}")
