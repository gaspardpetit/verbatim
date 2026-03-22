import logging
from typing import List, Optional, Protocol, Tuple

import torch
from langcodes import Language
from numpy.typing import NDArray

from .config import Config
from .models import Models

LOG = logging.getLogger(__name__)


class LanguageIdentifierProtocol(Protocol):
    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]: ...


class TranscriberLanguageIdentifier:
    """Default language identifier that delegates to the selected transcriber backend."""

    def __init__(self, models: Models):
        self._models = models

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        return self._models.transcriber.guess_language(audio=audio, lang=lang)


class MmsLanguageIdentifier:
    """Language identifier using facebook/mms-lid-126 independent of the transcription backend."""

    def __init__(self, *, model_size_or_path: str, device: str):
        try:
            from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
        except ImportError as exc:
            raise RuntimeError("MMS language identification requires `transformers`. Install the optional dependency set that includes it.") from exc

        if device == "cuda":
            torch_device = "cuda:0"
        elif device in ("cpu", "mps"):
            torch_device = device
        else:
            torch_device = "cpu"

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_size_or_path)
        self._model = Wav2Vec2ForSequenceClassification.from_pretrained(model_size_or_path)
        self._model.to(torch_device)
        self._model.eval()
        self._device = torch_device

    @staticmethod
    def _normalize_label(label: str) -> Optional[str]:
        candidate = str(label).strip().lower()
        if not candidate:
            return None
        try:
            normalized = Language.get(candidate).language
            if normalized and normalized != "und":
                return normalized
        except Exception:  # pylint: disable=broad-exception-caught
            LOG.debug("Failed to normalize MMS label %r", label, exc_info=True)
        return None

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        if len(lang) == 0:
            return "en", 1.0
        if len(lang) == 1:
            return lang[0], 1.0

        inputs = self._feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

        best_lang = lang[0]
        best_prob = 0.0
        id2label = getattr(self._model.config, "id2label", {})

        for index, probability in enumerate(probabilities.tolist()):
            label = id2label.get(index)
            if label is None:
                continue
            normalized = self._normalize_label(label)
            if normalized in lang and probability > best_prob:
                best_lang = normalized
                best_prob = float(probability)

        return best_lang, best_prob


def create_language_identifier(config: Config, models: Models) -> LanguageIdentifierProtocol:
    backend = (config.language_identifier_backend or "transcriber").lower()
    if backend == "transcriber":
        return TranscriberLanguageIdentifier(models=models)
    if backend == "mms":
        return MmsLanguageIdentifier(model_size_or_path=config.mms_lid_model_size, device=config.device)
    raise RuntimeError(f"Unsupported language_identifier_backend: {config.language_identifier_backend}")
