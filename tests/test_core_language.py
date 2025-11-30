import numpy as np

from verbatim_core import (
    LanguageDetectionRequest,
    TranscriptionWindowResult,
    detect_language,
)


def test_detect_language_defaults_to_en_when_lang_empty():
    req = LanguageDetectionRequest(audio=np.zeros(16000, dtype=np.float32), lang=[], timestamp=0, window_ts=0, audio_ts=16000)
    result = detect_language(request=req, guess_fn=lambda _audio, _langs: ("xx", 0.0))
    assert result.language == "en"
    assert result.probability == 1.0
    assert result.samples_used == 0


def test_detect_language_respects_single_language_hint():
    req = LanguageDetectionRequest(audio=np.zeros(16000, dtype=np.float32), lang=["fr"], timestamp=0, window_ts=0, audio_ts=16000)
    result = detect_language(request=req, guess_fn=lambda _audio, _langs: ("en", 0.9))
    assert result.language == "fr"
    assert result.probability == 1.0
    assert result.samples_used == 0


def test_detect_language_expands_window_until_confident():
    # Prepare audio longer than initial 2s window
    total_samples = 16000 * 8
    audio = np.arange(total_samples, dtype=np.float32)
    calls = []

    def guess_fn(chunk, langs):
        calls.append((len(chunk), tuple(langs)))
        # First call low confidence, second call confident
        if len(calls) == 1:
            return ("en", 0.2)
        return ("es", 0.8)

    req = LanguageDetectionRequest(audio=audio, lang=["en", "es"], timestamp=0, window_ts=0, audio_ts=total_samples)
    result = detect_language(request=req, guess_fn=guess_fn)

    # First try uses 2s (32k samples), second uses 4s (64k)
    assert calls[0][0] == 32000
    assert calls[1][0] == 64000
    assert result.language == "es"
    assert result.samples_used == 64000


def test_transcription_window_result_as_tuple_lists_sequences():
    result = TranscriptionWindowResult(utterance="u", unacknowledged=("a1", "a2"), unconfirmed_words=("w1",))
    utterance, unack, unconfirmed = result.as_tuple()
    assert utterance == "u"
    assert isinstance(unack, list) and unack == ["a1", "a2"]
    assert isinstance(unconfirmed, list) and unconfirmed == ["w1"]
