import re
from typing import List, Optional, Tuple

from ...transcript.words import Word

AlignedUnit = Tuple[int, int, str, float]


def split_transcript_text(transcript_text: str) -> List[str]:
    tokens: List[str] = []
    cursor = 0
    length = len(transcript_text)

    while cursor < length:
        token_start = cursor
        if transcript_text[cursor].isspace():
            while cursor < length and transcript_text[cursor].isspace():
                cursor += 1
        while cursor < length and not transcript_text[cursor].isspace():
            cursor += 1
        token = transcript_text[token_start:cursor]
        if token:
            tokens.append(token)

    return tokens


def normalize_for_alignment(text: str) -> str:
    return "".join(char for char in text if char.isalnum())


def find_matching_span(transcript_text: str, cursor: int, normalized_text: str) -> Optional[Tuple[int, int]]:
    if not normalized_text:
        return None

    start: Optional[int] = None
    norm_index = 0
    first_char = normalized_text[0].lower()

    for index in range(cursor, len(transcript_text)):
        char = transcript_text[index]
        char_norm = char.lower() if char.isalnum() else ""

        if start is None:
            if char_norm == first_char:
                start = index
                norm_index = 1
                if norm_index == len(normalized_text):
                    return start, index + 1
            continue

        if not char_norm:
            continue

        if norm_index < len(normalized_text) and char_norm == normalized_text[norm_index].lower():
            norm_index += 1
            if norm_index == len(normalized_text):
                return start, index + 1
            continue

        if char_norm == first_char:
            start = index
            norm_index = 1
            if norm_index == len(normalized_text):
                return start, index + 1
        else:
            start = None
            norm_index = 0

    return None


def split_leading_nonword(chunk: str) -> Tuple[str, str]:
    match = re.search(r"\w", chunk, re.UNICODE)
    if match is None:
        return chunk, ""
    if match.start() == 0:
        return "", chunk
    return chunk[: match.start()], chunk[match.start() :]


def project_timestamps_onto_transcript(
    *,
    transcript_text: str,
    aligned_units: List[AlignedUnit],
    lang: str,
    window_ts: int,
    audio_ts: int,
) -> List[Word]:
    if not transcript_text:
        return []

    words: List[Word] = []
    last_end_ts = window_ts
    cursor = 0
    truncated_at_audio_end = False

    for unit_start_ts, unit_end_ts, unit_text, unit_probability in aligned_units:
        normalized_unit = normalize_for_alignment(unit_text)
        if not normalized_unit:
            continue
        if unit_end_ts > audio_ts:
            truncated_at_audio_end = True
            break

        matching_span = find_matching_span(transcript_text=transcript_text, cursor=cursor, normalized_text=normalized_unit)
        if matching_span is None:
            continue

        _match_start, match_end = matching_span
        raw_chunk = transcript_text[cursor:match_end]
        leading_nonword, lexical_chunk = split_leading_nonword(raw_chunk)

        if leading_nonword and words:
            words[-1].word += leading_nonword
            words[-1].end_ts = max(words[-1].end_ts, unit_start_ts)
        elif leading_nonword:
            lexical_chunk = leading_nonword + lexical_chunk

        if not lexical_chunk:
            cursor = match_end
            continue

        start_ts = unit_start_ts
        end_ts = unit_end_ts
        probability = unit_probability
        if start_ts < last_end_ts:
            start_ts = last_end_ts
            end_ts = max(end_ts, start_ts)

        words.append(
            Word(
                start_ts=start_ts,
                end_ts=end_ts,
                word=lexical_chunk,
                probability=probability,
                lang=lang,
            )
        )
        last_end_ts = end_ts
        cursor = match_end

    if cursor < len(transcript_text) and words and not truncated_at_audio_end:
        tail = transcript_text[cursor:]
        words[-1].word += tail

    return words
