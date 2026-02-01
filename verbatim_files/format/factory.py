import sys
from typing import List

from .file import FileFormatter, MultiFileFormatter
from .writer import TranscriptWriterConfig


def configure_writers(
    write_config: TranscriptWriterConfig,
    output_formats: List[str],
    original_audio_file: str,
    output_prefix_no_ext: str,
) -> MultiFileFormatter:
    # pylint: disable=import-outside-toplevel
    formatters: List[FileFormatter] = []
    if "txt" in output_formats:
        from .txt import TextTranscriptWriter

        formatters.append(FileFormatter(writer=TextTranscriptWriter(config=write_config), output_path=f"{output_prefix_no_ext}.txt"))

    if "ass" in output_formats:
        from .ass import AssTranscriptWriter

        formatters.append(
            FileFormatter(
                writer=AssTranscriptWriter(config=write_config, original_audio_file=original_audio_file),
                output_path=f"{output_prefix_no_ext}.ass",
            )
        )

    if "docx" in output_formats:
        from .docx import DocxTranscriptWriter

        formatters.append(FileFormatter(writer=DocxTranscriptWriter(config=write_config), output_path=f"{output_prefix_no_ext}.docx"))

    if "md" in output_formats:
        from .md import MarkdownTranscriptWriter

        formatters.append(FileFormatter(writer=MarkdownTranscriptWriter(config=write_config), output_path=f"{output_prefix_no_ext}.md"))

    if "json" in output_formats:
        from .json import JsonTranscriptWriter

        formatters.append(FileFormatter(writer=JsonTranscriptWriter(config=write_config), output_path=f"{output_prefix_no_ext}.json"))

    if "jsonl" in output_formats:
        from .json import JsonlTranscriptWriter

        formatters.append(FileFormatter(writer=JsonlTranscriptWriter(config=write_config), output_path=f"{output_prefix_no_ext}.jsonl"))

    if "stdout" in output_formats and "stdout-nocolor" not in output_formats:
        from .stdout import StdoutTranscriptWriter

        formatters.append(
            FileFormatter(writer=StdoutTranscriptWriter(config=write_config, with_colours=True), output=sys.stdout.buffer, close_output=False)
        )

    if "stdout-nocolor" in output_formats:
        from .stdout import StdoutTranscriptWriter

        formatters.append(
            FileFormatter(writer=StdoutTranscriptWriter(config=write_config, with_colours=False), output=sys.stdout.buffer, close_output=False)
        )

    return MultiFileFormatter(formatters=formatters)
