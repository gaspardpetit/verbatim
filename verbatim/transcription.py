from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, LetterCase, DataClassJsonMixin

import json


def red(str):
    return f"\033[91m{str}\033[00m"


def green(str):
    return f"\033[92m{str}\033[00m"


def yellow(str):
    return f"\033[93m{str}\033[00m"


def light_purple(str):
    return f"\033[94m{str}\033[00m"


def purple(str):
    return f"\033[95m{str}\033[00m"


def cyan(str):
    return f"\033[96m{str}\033[00m"


def light_gray(str):
    return f"\033[97m{str}\033[00m"


def black(str):
    return f"\033[0m{str}\033[00m"


def colorize(text, confidence):
    if confidence > 0.8:
        return green(text)
    elif confidence < 0.2:
        return light_purple(text)
    elif confidence < 0.4:
        return red(text)
    elif confidence < 0.6:
        return yellow(text)
    return text


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(eq=False, unsafe_hash=True)
class Word(DataClassJsonMixin):
    start: float
    end: float
    confidence: float
    text: str

    def __init__(self, text: str, start: float, end: float, confidence: float):
        self.start: float = start
        self.end: float = end
        self.confidence: float = confidence
        self.text: str = text

    def __json__(self):
        return self.__dict__

    def get_colour_text(self):
        return colorize(self.text, self.confidence)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(eq=False, unsafe_hash=True)
class Utterance(DataClassJsonMixin):
    words: list[Word]
    language: str
    speaker: str
    start: float
    end: float
    confidence: float
    silence_prob: float

    def __init__(self, speaker: str, words: [Word], language: str, start: float = None, end: float = None,
                 confidence: float = None, silence_prob: float = None):
        self.words: [Word] = words
        self.language: str = language
        self.speaker: str = speaker
        self.start: float = start or min((w.start for w in words), default=start)
        self.end: float = end or max((w.end for w in words), default=end)
        self.confidence: float = confidence or min((w.confidence for w in words), default=confidence)
        self.silence_prob: float = silence_prob

    def __str__(self):
        return ''.join(w.text for w in self.words)

    def __json__(self):
        return self.__dict__

    def get_colour_text(self):
        transcription = ''.join(word.get_colour_text() for word in self.words)
        return f"[{self.start:.2f}->{self.end:.2f} {self.end - self.start:.2f}][{self.language}][{self.confidence:.2%}] {self.speaker}: {transcription}"

    def get_text(self):
        transcription = ''.join(word.text for word in self.words)
        return f"[{self.start:.2f}] {self.speaker}: {transcription}"

    def append(self, word: Word):
        self.words.append(word)
        self.start: float = min(self.start, word.start)
        self.end: float = max(self.end, word.end)
        self.confidence: float = min(self.confidence, word.confidence)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(eq=False, unsafe_hash=True)
class Transcription(DataClassJsonMixin):
    start: float
    end: float
    confidence: float
    utterances: list[Utterance]

    def __init__(self, start: float = None, end: float = None, confidence: float = None,
                 utterances: list[Utterance] = None):
        self.start = start
        self.end = end
        self.confidence = confidence
        self.utterances = utterances or []

    def append(self, utterance):
        self.utterances.append(utterance)
        self.utterances = sorted(self.utterances, key=lambda u: u.start)
        self.start = min(self.start or utterance.start, utterance.start)
        self.end = max(self.end or utterance.end, utterance.end)
        self.confidence = min(self.confidence or utterance.confidence, utterance.confidence)

    @staticmethod
    def from_json(jsonstr: str):
        parsed: dict = json.loads(jsonstr)
        return Transcription.from_dict(parsed)

    @staticmethod
    def load(json_file_path: str):
        with open(json_file_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
            return Transcription.from_dict(parsed)

    def save(self, json_file_path: str):
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(obj=self.to_dict(), fp=f, skipkeys=True, indent=2)

    def get_colour_text(self):
        return '\n'.join(utterance.get_colour_text() for utterance in self.utterances)

    def get_text(self):
        return '\n'.join(utterance.get_text() for utterance in self.utterances)

    def regroup_by_words(self):
        uttered_words: [Utterance] = []
        for utterance in self.utterances:
            for word in utterance.words:
                uttered_words.append(
                    Utterance(utterance.speaker, words=[word], language=utterance.language, start=word.start,
                              end=word.end, confidence=word.confidence, silence_prob=utterance.silence_prob))

        uttered_words = sorted(uttered_words, key=lambda u: u.start)

        grouped: Transcription = Transcription()
        state: Utterance = None
        for uttered_word in uttered_words:
            if state is not None:
                if state.language == uttered_word.language and state.speaker == uttered_word.speaker and uttered_word.start - state.end < 0.75:
                    state.append(uttered_word.words[0])
                else:
                    grouped.append(state)
                    state = uttered_word
            else:
                state = uttered_word
        if state is not None:
            grouped.append(state)
        return grouped
