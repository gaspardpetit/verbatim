"""Single-file execution helpers for the CLI."""

from typing import List

from verbatim.verbatim import execute
from verbatim_audio.sources.audiosource import AudioSource


def build_audio_sources(
    *,
    args,
    config,
    source_config,
    source_path: str,
    working_prefix_no_ext: str,
    output_prefix_no_ext: str,
) -> List[AudioSource]:
    audio_sources: List[AudioSource] = []
    if args.separate:
        from verbatim_audio.sources.factory import create_separate_speaker_sources

        audio_sources += create_separate_speaker_sources(
            strategy=args.separate or "pyannote",
            source_config=source_config,
            device=config.device,
            input_source=source_path,
            start_time=args.start_time,
            stop_time=args.stop_time,
            working_prefix_no_ext=working_prefix_no_ext,
            output_prefix_no_ext=output_prefix_no_ext,
        )
    else:
        from verbatim_audio.sources.factory import create_audio_sources

        audio_sources.extend(
            create_audio_sources(
                source_config=source_config,
                device=config.device,
                input_source=source_path,
                start_time=args.start_time,
                stop_time=args.stop_time,
                working_prefix_no_ext=working_prefix_no_ext,
                output_prefix_no_ext=output_prefix_no_ext,
                stream=config.stream,
            )
        )
    return audio_sources


def run_execute(
    *,
    source_path: str,
    config,
    write_config,
    audio_sources: List[AudioSource],
    output_formats,
    output_prefix_no_ext: str,
    working_prefix_no_ext: str,
    eval_file,
):
    execute(
        source_path=source_path,
        config=config,
        write_config=write_config,
        audio_sources=audio_sources,
        output_formats=output_formats,
        output_prefix_no_ext=output_prefix_no_ext,
        working_prefix_no_ext=working_prefix_no_ext,
        eval_file=eval_file,
    )
