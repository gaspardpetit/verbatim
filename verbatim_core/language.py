import logging

from numpy.typing import NDArray

from .interfaces import GuessLanguageFn, LanguageDetectionRequest, LanguageDetectionResult

LOG = logging.getLogger(__name__)


def _slice_audio(audio: NDArray, start: int, duration: int) -> NDArray:
    return audio[start : start + duration]


def detect_language(request: LanguageDetectionRequest, guess_fn: GuessLanguageFn) -> LanguageDetectionResult:
    """Stateless language detection helper that can be shared by verbatim-core."""
    if len(request.lang) == 0:
        LOG.warning("Language is not set - defaulting to english")
        return LanguageDetectionResult(language="en", probability=1.0, samples_used=0)

    if len(request.lang) == 1:
        return LanguageDetectionResult(language=request.lang[0], probability=1.0, samples_used=0)

    lang_sample_start = max(0, request.timestamp - request.window_ts)
    available_samples = request.audio_ts - request.window_ts - lang_sample_start
    lang_samples_size = min(2 * 16000, available_samples)

    while True:
        lang_samples = _slice_audio(request.audio, lang_sample_start, lang_samples_size)
        lang, prob = guess_fn(lang_samples, request.lang)
        LOG.info(
            "Detecting language using samples %s(%s) to %s(%s)",
            lang_sample_start,
            lang_sample_start / 16000.0,
            lang_sample_start + lang_samples_size,
            (lang_sample_start + lang_samples_size) / 16000.0,
        )
        if prob > 0.5 or lang_samples_size == available_samples:
            break
        # retry with larger sample
        lang_samples_size = min(2 * lang_samples_size, available_samples)

    return LanguageDetectionResult(language=lang, probability=prob, samples_used=lang_samples_size)
