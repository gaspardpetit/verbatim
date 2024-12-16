"""
Originally from https://github.com/jianfch/stable-ts/blob/main/stable_whisper/text_output.py
"""
import logging
# pylint: disable=logging-fstring-interpolation
import os
import warnings
from itertools import chain
from typing import List, Tuple, Union, Callable

from .writer import TranscriptWriter, TranscriptWriterConfig
from ..words import VerbatimUtterance

LOG = logging.getLogger(__name__)

def is_ascending_sequence(seq: List[Union[int, float]], verbose: bool = True) -> bool:
    """
    Check if a sequence is in ascending order.

    Args:
        seq (List[Union[int, float]]): The input sequence to check.
        verbose (bool, optional): If True, log details of the first descending pair.
            Defaults to True.

    Returns:
        bool: True if the sequence is in ascending order, False otherwise.
    """
    is_ascending = all(i <= j for i, j in zip(seq, seq[1:]))

    if not is_ascending and verbose:
        first_descending_idx = next((idx for idx, (i, j)
                                     in enumerate(zip(seq[:-1], seq[1:])) if i > j), None)
        LOG.info(f"[Index{first_descending_idx}]:{seq[first_descending_idx]} > " +
                 f"[Index{first_descending_idx + 1}]:{seq[first_descending_idx + 1]}")

    return is_ascending


def valid_ts(ts: List[dict], warn: bool = True) -> bool:
    """
    Check if a list of timestamp dictionaries represents a valid time sequence.

    Args:
        ts (List[dict]): List of timestamp dictionaries, each containing 'start' and 'end'.
        warn (bool, optional): If True, issue a warning for backward timestamp jumps.
            Defaults to True.

    Returns:
        bool: True if the timestamps form a valid ascending sequence, False otherwise.
    """
    time_points = list(chain.from_iterable([s['start'], s['end']] for s in ts))
    valid = is_ascending_sequence(time_points, False)

    if warn and not valid:
        warnings.warn("Found timestamp(s) jumping backwards in time. "
                      "Use word_timestamps=True to avoid the issue.")

    return valid


__all__ = ['result_to_srt_vtt', 'result_to_ass', 'result_to_tsv', 'result_to_txt', 'AssTranscriptWriter']
SUPPORTED_FORMATS = ('srt', 'vtt', 'ass', 'tsv', 'txt')


def _save_as_file(content: str, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    LOG.info(f'Saved: {os.path.abspath(path)}')


def _get_segments(result: (dict, list), min_dur: float, reverse_text: Union[bool, tuple] = False):
    if isinstance(result, dict):
        if reverse_text:
            warnings.warn(f'[reverse_text]=True only applies to WhisperResult but result is {type(result)}')
        return result.get('segments')
    elif not isinstance(result, list) and callable(getattr(result, 'segments_to_dicts', None)):
        return result.apply_min_dur(min_dur, inplace=False).segments_to_dicts(reverse_text=reverse_text)
    return result


def finalize_text(text: str, strip: bool = True):
    if not strip:
        return text
    return text.strip().replace('\n ', '\n')


def sec2hhmmss(seconds: (float, int)):
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    return hh, mm, ss


def sec2milliseconds(seconds: (float, int)) -> int:
    return round(seconds * 1000)


def sec2centiseconds(seconds: (float, int)) -> int:
    return round(seconds * 100)


def sec2vtt(seconds: (float, int)) -> str:
    hh, mm, ss = sec2hhmmss(seconds)
    return f'{hh:0>2.0f}:{mm:0>2.0f}:{ss:0>6.3f}'


def sec2srt(seconds: (float, int)) -> str:
    return sec2vtt(seconds).replace(".", ",")


def sec2ass(seconds: (float, int)) -> str:
    hh, mm, ss = sec2hhmmss(seconds)
    return f'{hh:0>1.0f}:{mm:0>2.0f}:{ss:0>2.2f}'


def segment2vttblock(segment: dict, strip=True) -> str:
    return f'{sec2vtt(segment["start"])} --> {sec2vtt(segment["end"])}\n' \
           f'{finalize_text(segment["text"], strip)}'


def segment2srtblock(segment: dict, idx: int, strip=True) -> str:
    return f'{idx}\n{sec2srt(segment["start"])} --> {sec2srt(segment["end"])}\n' \
           f'{finalize_text(segment["text"], strip)}'


def segment2assblock(segment: dict, idx: int, strip=True) -> str:
    return f'Dialogue: {idx},{sec2ass(segment["start"])},{sec2ass(segment["end"])},Default,,0,0,0,,' \
           f'{finalize_text(segment["text"], strip)}'


def segment2tsvblock(segment: dict, strip=True) -> str:
    return f'{sec2milliseconds(segment["start"])}' \
           f'\t{sec2milliseconds(segment["end"])}' \
           f'\t{segment["text"].strip() if strip else segment["text"]}'


def words2segments(words: List[dict], tag: Tuple[str, str], reverse_text: bool = False) -> List[dict]:
    def add_tag(idx: int):
        return ''.join(
            (
                f" {tag[0]}{w['word'][1:]}{tag[1]}"
                if w['word'].startswith(' ') else
                f"{tag[0]}{w['word']}{tag[1]}"
            )
            if w['word'] not in ('', ' ') and idx_ == idx else
            w['word']
            for idx_, w in idx_filled_words
        )

    filled_words = []
    for i, word in enumerate(words):
        curr_end = round(word['end'], 3)
        filled_words.append({"word": word['word'], "start": round(word['start'], 3), "end": curr_end })
        if word != words[-1]:
            next_start = round(words[i + 1]['start'], 3)
            if next_start - curr_end != 0:
                filled_words.append({ "word": '', "start": curr_end, "end": next_start })
    idx_filled_words = list(enumerate(filled_words))
    if reverse_text:
        idx_filled_words = list(reversed(idx_filled_words))

    segments = [{
        "text": add_tag(i),
        "start": filled_words[i]['start'],
        "end": filled_words[i]['end']
        } for i in range(len(filled_words))]
    return segments


def to_word_level_segments(segments: List[dict], tag: Tuple[str, str]) -> List[dict]:
    return list(
        chain.from_iterable(
            words2segments(s['words'], tag, reverse_text=s.get('reversed_text'))
            for s in segments
        )
    )


def to_vtt_word_level_segments(segments: List[dict]) -> List[dict]:
    def to_segment_string(segment: dict):
        segment_string = ''
        prev_end = 0
        for i, word in enumerate(segment['words']):
            if i != 0:
                curr_start = word['start']
                if prev_end == curr_start:
                    segment_string += f"<{sec2vtt(curr_start)}>"
                else:
                    if segment_string.endswith(' '):
                        segment_string = segment_string[:-1]
                    elif segment['words'][i]['word'].startswith(' '):
                        segment['words'][i]['word'] = segment['words'][i]['word'][1:]
                    segment_string += f"<{sec2vtt(prev_end)}> <{sec2vtt(curr_start)}>"
            segment_string += word['word']
            prev_end = word['end']
        return segment_string

    return [{
            "text": to_segment_string(s),
            "start": s['start'],
            "end": s['end']
        } for s in segments]


def to_ass_word_level_segments(segments: List[dict], karaoke: bool = False) -> List[dict]:
    def to_segment_string(segment: dict):
        segment_string = ''
        for _, word in enumerate(segment['words']):
            curr_word, space = (word['word'][1:], " ") if word['word'].startswith(" ") else (word['word'], "")
            segment_string += (
                    space +
                    r"{\k" +
                    ("f" if karaoke else "") +
                    f"{sec2centiseconds(word['end'] - word['start'])}" +
                    r"}" +
                    curr_word
            )
        return segment_string

    return [
        {
            "text": to_segment_string(s),
            "start": s['start'],
            "end": s['end']
        } for s in segments
    ]


def to_word_level(segments: List[dict]) -> List[dict]:
    return [{"text": w['word'], "start": w['start'], "end": w['end']} for s in segments for w in s['words']]



def _confirm_word_level(segments: List[dict]) -> bool:
    if not all(bool(s.get('words')) for s in segments):
        warnings.warn('Result is missing word timestamps. Word-level timing cannot be exported. '
                      'Use "word_level=False" to avoid this warning')
        return False
    return True


def _preprocess_args(result: (dict, list),
                     segment_level: bool,
                     word_level: bool,
                     min_dur: float,
                     reverse_text: Union[bool, tuple] = False):
    if not segment_level and not word_level:
        raise ValueError('`segment_level` or `word_level` must be True')
    segments = _get_segments(result, min_dur, reverse_text=reverse_text)
    if word_level:
        word_level = _confirm_word_level(segments)
    return segments, segment_level, word_level


def result_to_any(*, result: (dict, list),
                  filepath: str = None,
                  filetype: str = None,
                  segments2blocks: Callable = None,
                  segment_level=True,
                  word_level=True,
                  min_dur: float = 0.02,
                  tag: Tuple[str, str] = None,
                  default_tag: Tuple[str, str] = None,
                  strip=True,
                  reverse_text: Union[bool, tuple] = False,
                  to_word_level_string_callback: Callable = None):

    segments, segment_level, word_level = _preprocess_args(
        result, segment_level, word_level, min_dur, reverse_text=reverse_text
    )

    if filetype is None:
        filetype = os.path.splitext(filepath)[-1][1:] or 'srt'
    if filetype.lower() not in SUPPORTED_FORMATS:
        raise NotImplementedError(f'{filetype} not supported')
    if filepath and not filepath.lower().endswith(f'.{filetype}'):
        filepath += f'.{filetype}'

    if word_level and segment_level:
        if tag is None:
            if default_tag is None:
                tag = ('<font color="#00ff00">', '</font>') if filetype == 'srt' else ('<u>', '</u>')
            else:
                tag = default_tag
        if to_word_level_string_callback is None:
            to_word_level_string_callback = to_word_level_segments
        segments = to_word_level_string_callback(segments, tag)
    elif word_level:
        segments = to_word_level(segments)

    valid_ts(segments)

    if segments2blocks is None:
        sub_str = '\n\n'.join(segment2srtblock(s, i, strip=strip) for i, s in enumerate(segments))
    else:
        sub_str = segments2blocks(segments)

    if filepath:
        _save_as_file(sub_str, filepath)
        return None
    else:
        return sub_str


def result_to_srt_vtt(*, result: (dict, list),
                      filepath: str = None,
                      segment_level=True,
                      word_level=True,
                      min_dur: float = 0.02,
                      tag: Tuple[str, str] = None,
                      vtt: bool = None,
                      strip=True,
                      reverse_text: Union[bool, tuple] = False):
    is_srt = (filepath is None or not filepath.lower().endswith('.vtt')) if vtt is None else not vtt
    if is_srt:
        segments2blocks = None
        to_word_level_string_callback = None
    else:
        def segments2blocks(segments):
            return 'WEBVTT\n\n' + '\n\n'.join(segment2vttblock(s, strip=strip) for i, s in enumerate(segments))

        to_word_level_string_callback = to_vtt_word_level_segments if tag is None else tag

    return result_to_any(
        result=result,
        filepath=filepath,
        filetype=('vtt', 'srt')[is_srt],
        segments2blocks=segments2blocks,
        segment_level=segment_level,
        word_level=word_level,
        min_dur=min_dur,
        tag=tag,
        strip=strip,
        reverse_text=reverse_text,
        to_word_level_string_callback=to_word_level_string_callback
    )


def result_to_tsv(*, result: (dict, list),
                  filepath: str = None,
                  segment_level: bool = None,
                  word_level: bool = None,
                  min_dur: float = 0.02,
                  strip=True,
                  reverse_text: Union[bool, tuple] = False):
    if segment_level is None and word_level is None:
        segment_level = True
    if word_level is segment_level:
        raise ValueError('[word_level] and [segment_level] cannot be the same '
                                    'since [tag] is not support for this format')

    def segments2blocks(segments):
        return '\n\n'.join(segment2tsvblock(s, strip=strip) for i, s in enumerate(segments))

    return result_to_any(
        result=result,
        filepath=filepath,
        filetype='tsv',
        segments2blocks=segments2blocks,
        segment_level=segment_level,
        word_level=word_level,
        min_dur=min_dur,
        strip=strip,
        reverse_text=reverse_text
    )


def result_to_ass(*, result: (dict, list),
                  filepath: str = None,
                  segment_level=True,
                  word_level=True,
                  min_dur: float = 0.02,
                  tag: Union[Tuple[str, str], int] = None,
                  font: str = None,
                  font_size: int = 24,
                  strip=True,
                  highlight_color: str = None,
                  karaoke=False,
                  reverse_text: Union[bool, tuple] = False,
                  **kwargs):
    if tag == ['-1']:  # CLI
        tag = -1
    if highlight_color is None:
        highlight_color = '00ff00'

    def segments2blocks(segments):
        fmt_style_dict = {'Name': 'Default', 'Fontname': 'Arial', 'Fontsize': '48', 'PrimaryColour': '&Hffffff',
                          'SecondaryColour': '&Hffffff', 'OutlineColour': '&H0', 'BackColour': '&H0', 'Bold': '0',
                          'Italic': '0', 'Underline': '0', 'StrikeOut': '0', 'ScaleX': '100', 'ScaleY': '100',
                          'Spacing': '0', 'Angle': '0', 'BorderStyle': '1', 'Outline': '1', 'Shadow': '0',
                          'Alignment': '2', 'MarginL': '10', 'MarginR': '10', 'MarginV': '10', 'Encoding': '0'}

        for k, v in filter(lambda x: 'colour' in x[0].lower() and not str(x[1]).startswith('&H'), kwargs.items()):
            kwargs[k] = f'&H{kwargs[k]}'

        fmt_style_dict.update((k, v) for k, v in kwargs.items() if k in fmt_style_dict)

        if tag is None and 'PrimaryColour' not in kwargs:
            fmt_style_dict['PrimaryColour'] = \
                highlight_color if highlight_color.startswith('&H') else f'&H{highlight_color}'

        if font:
            fmt_style_dict.update(Fontname=font)
        if font_size:
            fmt_style_dict.update(Fontsize=font_size)

        fmts = f'Format: {", ".join(map(str, fmt_style_dict.keys()))}'

        styles = f'Style: {",".join(map(str, fmt_style_dict.values()))}'

        sub_str = f'[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\nScaledBorderAndShadow: yes\n\n' \
                  f'[V4+ Styles]\n{fmts}\n{styles}\n\n' \
                  f'[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\n'

        sub_str += '\n'.join(segment2assblock(s, i, strip=strip) for i, s in enumerate(segments))

        return sub_str

    if tag is not None and karaoke:
        warnings.warn('[tag] is not support for [karaoke]=True; [tag] will be ignored.')

    return result_to_any(
        result=result,
        filepath=filepath,
        filetype='ass',
        segments2blocks=segments2blocks,
        segment_level=segment_level,
        word_level=word_level,
        min_dur=min_dur,
        tag=None if tag == -1 else tag,
        default_tag=(r'{\1c' + f'{highlight_color}&' + '}', r'{\r}'),
        strip=strip,
        reverse_text=reverse_text,
        to_word_level_string_callback=(
            (lambda s, _: to_ass_word_level_segments(segments=s, karaoke=karaoke))
            if karaoke or (word_level and segment_level and tag is None)
            else None
        )
    )


def result_to_txt(
        result: (dict, list),
        filepath: str = None,
        min_dur: float = 0.02,
        strip=True,
        reverse_text: Union[bool, tuple] = False
):
    def segments2blocks(segments: dict, _strip=True) -> str:
        return '\n'.join(f'{segment["text"].strip() if _strip else segment["text"]}' for segment in segments)

    return result_to_any(
        result=result,
        filepath=filepath,
        filetype='txt',
        segments2blocks=segments2blocks,
        segment_level=True,
        word_level=False,
        min_dur=min_dur,
        strip=strip,
        reverse_text=reverse_text
    )


class AssTranscriptWriter(TranscriptWriter):
    def __init__(self, config: TranscriptWriterConfig, original_audio_file:str):
        super().__init__(config)
        self.utterances:List[VerbatimUtterance] = []
        self.output_file = None
        self.original_audio_file = original_audio_file

    def open(self, path_no_ext:str):
        self.output_file = f"{path_no_ext}.ass"

    def write(self, utterance:VerbatimUtterance):
        self.utterances.append(utterance)

    def close(self):
        result_to_ass(result={"segments": [{
            "start": u.get_start(),
            "end": u.get_end(),
            "text": f"{u.speaker}: {''.join([w.word for w in u.words])}",
            "words": [{
                "start": u.get_start(),
                "end": u.get_start(),
                "word": f"{u.speaker}: "
            }] + [{
                "start": w.start_ts / 16000,
                "end": w.end_ts / 16000,
                "word": w.word
            } for w in u.words]}
            for u in self.utterances]}, filepath=self.output_file)
        LOG.info("To combine the subtitles with the original file:")
        LOG.info(
            f"""ffmpeg -f lavfi -i color=size=720x120:rate=25:color=black -i "{self.original_audio_file}" """ +
                f"""-vf "subtitles={self.output_file}:force_style='Fontsize=70'" """ +
                f"""-shortest "{self.output_file}.mp4" """)
