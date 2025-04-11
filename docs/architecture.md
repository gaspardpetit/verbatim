# Architecture

`verbatim` transforms speech to text through a pipeline of specialized components:

1. Voice isolation
2. Speaker diarization and isolation (using both `pyannote.audio` and a specialized [stereo approach](./api/verbatim.voices.diarize.stereo.rst))
3. Language detection and transcription
4. Post-processing functinos (WIP) of the transcripts

## Introduction

Whisper, OpenAI's powerful automatic speech recognition (ASR) model, has set new standards in transcription accuracy across multiple languages. However, most projects based on Whisper (including OpenAI Whisper, WhisperX, and Faster Whisper) expect a single language setting or detect language based on a short audio sample.

In multilingual conversations, these approaches lead to speech being dropped, translated incorrectly, or hallucinated.

`verbatim` implements multiple strategies to handle rapid language changes:

1. Dynamically adjusting the boundaries of the audio segment used for language detection
2. Adjusting the attention window provided to Whisper to avoid skipping speech that doesn't match the current language
3. Detecting when transcription boundaries don't match language detection boundaries

## Dynamic Adjustment of Language Detection Boundaries

Most Whisper-based transcription projects use a fixed-length window for language detection. `verbatim` instead uses an adaptive approach, continuously refining the detected language by adjusting sample size and confidence scoring.

## Incremental Language Detection

Rather than making a one-time decision, `verbatim` iteratively expands the subset of speech samples until it reaches high-confidence language classification:

```python
# Start with a small sample and expand if confidence is low
lang_sample_start = max(0, timestamp - self.state.window_ts)
available_samples = self.state.audio_ts - self.state.window_ts - lang_sample_start
lang_samples_size = min(2 * 16000, available_samples)  # Start with 2 seconds of audio

while True:
    lang, prob = self._guess_language(
        audio=self.state.rolling_window.array,
        sample_offset=lang_sample_start,
        sample_duration=lang_samples_size,
        lang=self.config.lang,
    )
    if prob > 0.5 or lang_samples_size == available_samples:
        break  # Stop expanding when confidence is high

    # Double the sample size if needed
    lang_samples_size = min(2 * lang_samples_size, available_samples)
```

## Handling Code-Switching and Language Transitions

`verbatim` ensures that the remainder of the transcription is within the language detection window:

```python
if transcript_words[0].start_ts >= self.state.window_ts + used_samples_for_language:
    best_transcript = transcript_words
    best_first_word_ts = self.state.audio_ts
    best_lang = lang

    # Re-test transcription in alternative languages
    for test_lang in self.config.lang:
        alt_transcript_words = self.models.transcriber.transcribe(...)

        if len(alt_transcript_words) > 0:
            first_word_ts = alt_transcript_words[0].start_ts
            confirmed_lang, _, _ = self.guess_language(timestamp=max(0, first_word_ts))

            if confirmed_lang == test_lang and first_word_ts < best_first_word_ts:
                best_first_word_ts = first_word_ts
                best_transcript = alt_transcript_words
                best_lang = test_lang

    # Update to best detected language and transcript
    lang = best_lang
    transcript_words = best_transcript
```

## Minimizing Transcription Attention Span

`verbatim` minimizes Whisper's attention span to ensure that each transcription window contains a **single utterance**, avoiding skipping multilingual content and preventing hallucinations.

### Growing Speech Windows

```python
while True:
    self.capture_audio(audio_source=audio_stream)
    self.flush_overflowing_utterances(diarization=audio_stream.diarization)
    self.process_audio_window(audio_stream=audio_stream)
```

### Transcribing Only the Necessary Window

```python
def transcribe_window(self) -> Tuple[List[Word], List[Word]]:
    self.guess_language(timestamp=max(0, self.state.acknowledged_ts))
    prefix_text = # Build from previous confirmed words
    transcript_words = self.models.transcriber.transcribe(prefix=prefix_text, ...)
    confirmed_words = self.state.transcript_candidate_history.confirm(...)
    self.state.transcript_candidate_history.add(transcription=transcript_words)
    return confirmed_words, unconfirmed_words
```

## Conclusion

In summary, `verbatim` achieves multilingual support by:

1. Reducing the length of each transcription to the minimum necessary
2. Detecting language using the shortest audio length that provides sufficient confidence

This approach transforms Whisper into a truly multilingual ASR system capable of handling real-world multilingual conversations with precision.
