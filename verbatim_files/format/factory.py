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
    use_stdout = "stdout" in output_formats
    if use_stdout:
        selected_formats = [fmt for fmt in output_formats if fmt != "stdout"]
        if len(selected_formats) != 1:
            raise ValueError("stdout output requires exactly one selected output format")
        stdout_output = sys.stdout.buffer

    if "txt" in output_formats:
        from .txt import TextTranscriptWriter

        if use_stdout:
            formatters.append(FileFormatter(writer=TextTranscriptWriter(config=write_config), output=stdout_output, close_output=False))
        else:
            formatters.append(FileFormatter(writer=TextTranscriptWriter(config=write_config), output_path=f"{output_prefix_no_ext}.txt"))

    if "ass" in output_formats:
        from .ass import AssTranscriptWriter

        if use_stdout:
            formatters.append(
                FileFormatter(
                    writer=AssTranscriptWriter(config=write_config, original_audio_file=original_audio_file),
                    output=stdout_output,
                    close_output=False,
                )
            )
        else:
            formatters.append(
                FileFormatter(
                    writer=AssTranscriptWriter(config=write_config, original_audio_file=original_audio_file),
                    output_path=f"{output_prefix_no_ext}.ass",
                )
            )

    if "docx" in output_formats:
        from .docx import DocxTranscriptWriter

        if use_stdout:
            formatters.append(FileFormatter(writer=DocxTranscriptWriter(config=write_config), output=stdout_output, close_output=False))
        else:
            formatters.append(FileFormatter(writer=DocxTranscriptWriter(config=write_config), output_path=f"{output_prefix_no_ext}.docx"))

    if "md" in output_formats:
        from .md import MarkdownTranscriptWriter

        if use_stdout:
            formatters.append(FileFormatter(writer=MarkdownTranscriptWriter(config=write_config), output=stdout_output, close_output=False))
        else:
            formatters.append(FileFormatter(writer=MarkdownTranscriptWriter(config=write_config), output_path=f"{output_prefix_no_ext}.md"))

    if "json" in output_formats:
        from .json import JsonTranscriptWriter

        if use_stdout:
            formatters.append(FileFormatter(writer=JsonTranscriptWriter(config=write_config), output=stdout_output, close_output=False))
        else:
            formatters.append(FileFormatter(writer=JsonTranscriptWriter(config=write_config), output_path=f"{output_prefix_no_ext}.json"))

    if "jsonl" in output_formats:
        from .json import JsonlTranscriptWriter

        if use_stdout:
            formatters.append(FileFormatter(writer=JsonlTranscriptWriter(config=write_config), output=stdout_output, close_output=False))
        else:
            formatters.append(FileFormatter(writer=JsonlTranscriptWriter(config=write_config), output_path=f"{output_prefix_no_ext}.jsonl"))

    return MultiFileFormatter(formatters=formatters)
