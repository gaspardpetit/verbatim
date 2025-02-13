import logging
from abc import ABC, abstractmethod
import math
from typing import List
import re
from wtpsplit import SaT

from verbatim.transcript.words import Word
from verbatim.audio.audio import samples_to_seconds

# Configure logger
LOG = logging.getLogger(__name__)


class SentenceTokenizer(ABC):
    @abstractmethod
    def split(self, words:List[Word]) -> List[str]:
        pass

class FastSentenceTokenizer(SentenceTokenizer):
    def split(self, words:List[Word]) -> List[str]:
        text = ''.join(w.word for w in words)
        # List of punctuation marks to split on
        split_punctuations = r".。;!！?？"
        punctuations = "\"'.。,;，!！?？:：”)]}、\"'“¿([{-"
        # Create a regex pattern to match any punctuation character
        regex_punctuation = f"([{re.escape(split_punctuations)}])"
        regex_not_punctuation = f"([^{re.escape(punctuations)}])"
        # Split the sentence, keeping the punctuation in the result
        parts = re.split(regex_punctuation, text)
        # Merge punctuation with the preceding part, preserving spaces
        result = []
        text_words = ""
        for part in parts:
            text_words += part
            if text_words and re.match(regex_not_punctuation, text_words):
                result.append(text_words)
                text_words = ""
            elif len(text_words) > 0:
                result[-1] += text_words
                text_words = ""

        if text_words:
            if re.match(regex_not_punctuation, text_words):
                result.append(text_words)
            elif len(result) > 0:
                result[-1] += text_words
            else:
                result.append(text_words)

        return result


class SaTSentenceTokenizer(SentenceTokenizer):
    def __init__(self, device: str, model="sat-3l-sm"):
        LOG.info(f"Initializing SaT Sentence Tokenizer with model {model}")
        self.sat_sm = SaT(model)
        self.sat_sm.half().to(device)

    def split(self, words:List[Word]) -> List[str]:
        text = ''.join(w.word for w in words)
        return self.sat_sm.split(text)  # pyright: ignore[reportReturnType]

class SilenceSentenceTokenizer(SentenceTokenizer):
    def __init__(self):
        # parameters can be tuned as needed
        self.target_duration = 10.0    # target sentence length in seconds
        self.min_duration = 2.0        # below this, the penalty is huge
        self.max_duration = 25.0       # above this, also huge penalty
        # Weight for how much a silence gap improves the candidate break.
        self.gap_weight = 1.0
        self.distance_weight = 1.0
        self.silence_weight = 2.0
        self.word_weight = 1.0

    def split(self, words: List[Word]) -> List[str]:
        if not words:
            return []

        def distance_energy(val:float, min_val:float, max_val:float, ideal_val:float) -> float:
            value:float = max(0, min(1, (val - min_val) / (ideal_val - min_val) if val < ideal_val else (max_val-val) / (max_val-ideal_val)))
            return 1 - value

        def silence_energy(duration:float, max_duration:float) -> float:
            value:float = max(0, min(1, duration / max_duration))
            return 1 - value

        def word_energy(left_word:str, right_word:str) -> float:
            primary_suffix_punctuations = tuple([".", "。", ";", "!", "！", "?", "？"])
            secondary_prefix_punctuations = tuple(["\"", "“", "¿", "¡", "(", "[", "{", "«"])
            secondary_suffix_punctuations = tuple([",", "、"])

            value:float = 0
            if left_word.rstrip().endswith(primary_suffix_punctuations):
                value += 1
            if left_word.rstrip().endswith(secondary_suffix_punctuations):
                value += 0.25
            if right_word.lstrip().startswith(secondary_prefix_punctuations):
                value += 0.25
            return 1 - max(0, min(1, value))

        def _energy(first_word:Word, left_word:Word, right_word:Word):
            seconds_from_start = samples_to_seconds(right_word.start_ts - first_word.start_ts)
            silence_duration = samples_to_seconds(right_word.start_ts - left_word.end_ts)

            distance_cost = distance_energy(val=seconds_from_start, min_val=self.min_duration, max_val=self.max_duration, ideal_val=self.target_duration)
            silence_cost = silence_energy(duration=silence_duration, max_duration=self.max_duration)
            word_cost = word_energy(left_word=left_word.word, right_word=right_word.word)
            energy = self.distance_weight * distance_cost + self.silence_weight * silence_cost + self.word_weight * word_cost
            return energy

        sentences = []
        weights = []
        current_start = 0  # index of the first word in the current sentence

        best_break_index = None
        best_energy = math.inf

        for i in range(current_start, len(words) - 1):
            energy = _energy(first_word=words[0], left_word=words[i], right_word=words[i+1])
            weights.append(energy)
            if energy < best_energy:
                best_energy = energy
                best_break_index = i

        LOG.debug(f"split weight:\n{''.join(w.word for w in words)}\n{weights}")
        # If no candidate was found (e.g. because the duration grows too large)
        # then take all remaining words.
        if best_break_index is None:
            best_break_index = len(words) - 1

        # Build the sentence from current_start to best_break_index (inclusive)
        sentence = "".join(word.word for word in words[current_start: best_break_index+1])
        if sentence:
            sentences.append(sentence)

        sentence = "".join(word.word for word in words[best_break_index+1:])
        if sentence:
            sentences.append(sentence)

        # Move to the next segment
        current_start = best_break_index + 1

        return sentences

class BoundedSentenceTokenizer(SentenceTokenizer):
    def __init__(self, other_tokenizer:SentenceTokenizer, bounding_tokenizer:SentenceTokenizer=SilenceSentenceTokenizer(), max_duration:float=25):
        self.other_tokenizer = other_tokenizer
        self.bounding_tokenizer = bounding_tokenizer
        self.max_duration = max_duration

    def split(self, words:List[Word]) -> List[str]:
        tok = self.other_tokenizer.split(words=words)
        if len(tok) != 1:
            return tok

        duration:float = samples_to_seconds(words[-1].end_ts - words[0].start_ts)
        if duration <= self.max_duration:
            return tok

        return self.bounding_tokenizer.split(words=words)
