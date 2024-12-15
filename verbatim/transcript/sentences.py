import logging
from abc import abstractmethod
from typing import List

# Configure logger
LOG = logging.getLogger(__name__)

class SentenceTokenizer:
    @abstractmethod
    def split(self, text:str) -> List[str]:
        pass

class FastSentenceTokenizer(SentenceTokenizer):
    def split(self, text:str) -> List[str]:
        import re
        # List of punctuation marks to split on
        SPLIT_PUNCTUATIONS = r".。;!！?？"
        PUNCTUATIONS = "\"'.。,;，!！?？:：”)]}、\"'“¿([{-"
        # Create a regex pattern to match any punctuation character
        regex_punctuation = f"([{re.escape(SPLIT_PUNCTUATIONS)}])"
        regex_not_punctuation = f"([^{re.escape(PUNCTUATIONS)}])"
        # Split the sentence, keeping the punctuation in the result
        parts = re.split(regex_punctuation, text)
        # Merge punctuation with the preceding part, preserving spaces
        result = []
        words = ""
        for part in parts:
            words += part
            if words and re.match(regex_not_punctuation, words):
                result.append(words)
                words = ""

        if words:
            if re.match(regex_not_punctuation, words):
                result.append(words)
            elif len(result) > 0:
                result[-1] += words
            else:
                result.append(words)

        return result

class SaTSentenceTokenizer(SentenceTokenizer):
    def __init__(self, device:str, model="sat-12l-sm"):
        from wtpsplit import SaT
        self.sat_sm = SaT(model)
        self.sat_sm.half().to(device)

    def split(self, text:str) -> List[str]:
        return self.sat_sm.split(text)

