from typing import List

from .writer import TranscriptWriterConfig, TranscriptWriter


def configure_writers(
    write_config: TranscriptWriterConfig,
    output_formats: List[str],
    original_audio_file: str,
) -> TranscriptWriter:
    # pylint: disable=import-outside-toplevel
    from .multi import MultiTranscriptWriter

    multi_writer: MultiTranscriptWriter = MultiTranscriptWriter()
    if "txt" in output_formats:
        from .txt import TextTranscriptWriter

        multi_writer.add_writer(TextTranscriptWriter(config=write_config))
    if "ass" in output_formats:
        from .ass import AssTranscriptWriter

        multi_writer.add_writer(AssTranscriptWriter(config=write_config, original_audio_file=original_audio_file))
    if "docx" in output_formats:
        from .docx import DocxTranscriptWriter

        multi_writer.add_writer(DocxTranscriptWriter(config=write_config))
    if "md" in output_formats:
        from .md import MarkdownTranscriptWriter

        multi_writer.add_writer(MarkdownTranscriptWriter(config=write_config))
    if "json" in output_formats:
        from .json import JsonTranscriptWriter

        multi_writer.add_writer(JsonTranscriptWriter(config=write_config))
    if "json_dlm" in output_formats:
        from .json_dlm import JsonDiarizationLMTranscriptWriter

        multi_writer.add_writer(JsonDiarizationLMTranscriptWriter(config=write_config))
    if "stdout" in output_formats and "stdout-nocolor" not in output_formats:
        from .stdout import StdoutTranscriptWriter

        multi_writer.add_writer(StdoutTranscriptWriter(config=write_config, with_colours=True))
    if "stdout-nocolor" in output_formats:
        from .stdout import StdoutTranscriptWriter

        multi_writer.add_writer(StdoutTranscriptWriter(config=write_config, with_colours=False))
    return multi_writer
