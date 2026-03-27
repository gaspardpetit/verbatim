import logging
from time import perf_counter

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
    initial_samples = max(1, request.initial_samples)
    increment_samples = max(0, request.increment_samples)
    growth_factor = max(1.0, float(request.factor))
    lang_samples_size = min(initial_samples, available_samples)
    detect_start = perf_counter()

    while True:
        lang_samples = _slice_audio(request.audio, lang_sample_start, lang_samples_size)
        lang, prob = guess_fn(lang_samples, request.lang)
        LOG.debug(
            "Language detection attempt: samples=%s seconds=%.3f start=%s lang=%s prob=%.3f allowed=%s",
            lang_samples_size,
            lang_samples_size / 16000.0,
            lang_sample_start,
            lang,
            prob,
            request.lang,
        )
        if prob > 0.5 or lang_samples_size == available_samples:
            break
        # retry with larger sample
        next_size = int(lang_samples_size * growth_factor) + increment_samples
        next_size = min(next_size, available_samples)
        if next_size <= lang_samples_size:
            LOG.debug(
                "Language detection growth stalled: current=%s next=%s factor=%.3f increment=%s",
                lang_samples_size,
                next_size,
                growth_factor,
                increment_samples,
            )
            break
        lang_samples_size = next_size

    elapsed_ms = (perf_counter() - detect_start) * 1000.0
    if lang not in request.lang and request.lang:
        LOG.debug(
            "Language detection did not resolve to an allowed language after %s samples; falling back to %s",
            lang_samples_size,
            request.lang[0],
        )
        lang = request.lang[0]
        prob = 0.0
    LOG.info(
        "Detected language using %s samples (%.3fs): lang=%s prob=%.3f elapsed=%.1fms",
        lang_samples_size,
        lang_samples_size / 16000.0,
        lang,
        prob,
        elapsed_ms,
    )

    return LanguageDetectionResult(language=lang, probability=prob, samples_used=lang_samples_size)
