# Making Whisper Truly Multilingual

## Introduction
Whisper, OpenAI’s powerful automatic speech recognition (ASR) model, has set new standards in transcription accuracy across multiple languages. The model can transform speech from a hundred different languages into text with error rates rivaling the best commercial solutions.

All the projects we could find based on Whisper, including [OpenAI Whisper](https://github.com/openai/whisper), [WhisperX](https://github.com/m-bain/whisperX), and [Faster Whisper](https://github.com/SYSTRAN/faster-whisper), either expect the language setting to be provided by the user or detect the language based on a short (30s) audio segment.

In situations where multiple languages may be spoken within the same conversation, results with these implementations vary—from speech being dropped to being translated incorrectly or hallucinated.

Achieving true multilingual transcription is a complex challenge that goes beyond merely repeating language detection. This project implements multiple strategies to handle rapid language changes, including:

- Dynamically adjusting the boundaries of the audio segment used for language detection.
- Adjusting the attention window provided to Whisper when transcribing to avoid skipping over speech that does not match the model's current language setting.
- Detecting when the transcription's boundaries do not match the language detection boundaries.

By combining these techniques, Verbatim turns Whisper into a robust multilingual transcription system.

The rolling audio buffer exposes `active_audio()` for ingested samples in the current window, `window_audio()` for consumers that require the padded window, and `audio_between(start_ts, end_ts)` for an absolute sample range. `State` keeps the active start and audio end timestamps; appending and advancing use the buffer's write and advance methods. The buffer retains up to `Config.backward_window_duration` seconds (30 by default) before the active start, available from `retained_start_ts` through `audio_between()`. This backward duration is independent of the active transcription window. The active attention window and the audio supplied to transcription and language detection remain unchanged. This retained range makes a later second-opinion transcription possible without changing acknowledgement behavior here.

The backward duration must be finite and non-negative; zero disables retained history.

Session creation rejects a `Config.sampling_rate` that differs from the runtime `VERBATIM_SAMPLE_RATE` (`AUDIO_PARAMS.sample_rate`). This ensures that the retained history and active window represent their configured durations in the audio stream.

The post-acknowledgement advance uses `Config.post_acknowledgement_skip_duration` (0.1 seconds by default). The actual advance is capped at the next pending or unconfirmed word and at the end of available audio.
The duration must be finite and non-negative; zero disables this extra advance.

The gap-language check requires at least `Config.language_gap_min_duration` seconds between the current anchor and a candidate utterance (0.25 seconds by default). When a language mismatch blocks the candidate, `Config.language_mismatch_skip_duration` controls the advance before retrying (0.1 seconds by default). Both durations belong to the session and are converted using its sample rate.

## Dynamic Adjustment of Language Detection Boundaries

Most Whisper-based transcription projects rely on a fixed-length language detection window—typically 30 seconds—to determine the transcription language. While this works well for monolingual recordings, it fails in multilingual conversations where speakers rapidly switch languages mid-sentence.

Instead of using a rigid 30-second window for language detection, Verbatim employs an adaptive approach, continuously refining the detected language by adjusting the sample size and confidence scoring.

### 1. Incremental Language Detection

Rather than making a one-time decision, Verbatim iteratively expands the subset of speech samples until it reaches a high-confidence classification.

#### Example: Incrementally Refining Detected Language

```python
# Start with a small sample and expand if confidence is low.
lang_sample_start = max(0, timestamp - self.state.window_ts)
available_samples = self.state.audio_ts - self.state.window_ts - lang_sample_start
lang_samples_size = min(2 * 16000, available_samples)  # Start with 2 seconds of audio

while True:
    lang, prob = self._guess_language(
        audio=self.state.window_audio(),
        sample_offset=lang_sample_start,
        sample_duration=lang_samples_size,
        lang=self.config.lang,
    )
    if prob > 0.5 or lang_samples_size == available_samples:
        break  # Stop expanding when confidence is high.
    
    # Double the sample size if needed.
    lang_samples_size = min(2 * lang_samples_size, available_samples)  
```

**How It Works:**
- The function starts with a small language sample and increases it iteratively.
- If the confidence remains low, the function expands the window until a confident language decision is made.
- This prevents premature misclassification, which is a major problem in traditional Whisper implementations.

However, a drawback of this approach is incorrect detection on hesitation markers such as "hmmmm [pause] je vois ce que tu veux dire..." since Whisper tends to detect English with high confidence on short hesitation markers ("ah...", "huh..." etc.).

### 2. Handling Code-Switching and Language Transitions

A way to detect these situations is to ensure that the remainder of the transcription is within the language detection window. Whisper often skips hesitation markers in the transcription, meaning that language detection might be based on these hesitation markers rather than the actual speech.

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

    # Update to best detected language and transcript.
    lang = best_lang
    transcript_words = best_transcript
```

**How It Works:**
- If Whisper starts transcribing words outside the original language detection window, Verbatim re-tests transcription in each possible language.
- It selects the earliest correctly aligned transcript, ensuring the detected language matches the actual spoken words.
- This prevents speech from being ignored or misclassified, improving transcription reliability.

## Minimizing Transcription Attention Span

Most ASR systems, including Whisper, process fixed-length audio segments (e.g., 30 seconds) before outputting transcription results. While this is relatively short, it is long enough to contain multiple utterances and possibly multiple languages. Typically, when Whisper is configured for a given language (e.g., English), utterances from other languages will be ignored within the window.

Verbatim minimizes Whisper's attention span to ensure that each transcription window contains a **single utterance**. This avoids skipping multilingual content and prevents hallucinations.

### 1. Growing Speech Windows

```python
while True:
    self.capture_audio(audio_source=audio_stream)
    self.flush_overflowing_utterances(diarization=audio_stream.diarization)
    self.process_audio_window(audio_stream=audio_stream)
```

**How It Works:**
- Captures audio in an **adaptively growing window**.
- Uses **Voice Activity Detection (VAD)** to skip silence and detect speech.
- Expands the speech window until a **complete utterance** is detected.

### 2. Transcribing Only the Necessary Window

```python
def transcribe_window(self) -> Tuple[List[Word], List[Word]]:
    self.guess_language(timestamp=max(0, self.state.acknowledged_ts))
    prefix_text = # Build from previous confirmed words
    transcript_words = self.models.transcriber.transcribe(prefix=prefix_text, ...)
    confirmed_words = self.state.transcript_candidate_history.confirm(...)
    self.state.transcript_candidate_history.add(transcription=transcript_words)
    return confirmed_words, unconfirmed_words
```

**How It Works:**
- Transcription is performed only on the detected speech segment.
- Prefix text ensures continuity with previous confirmed words.
- Avoids transcribing unnecessary silence or incomplete utterances.

## Transcriber capability contract

The selected transcriber exposes `get_capabilities()` through `TranscriberProtocol`. The capability record also declares `window_duration` in seconds; every current backend reports `30.0`. The orchestrator still uses its configured window duration in this change. The returned immutable `TranscriberCapabilities` describes whether the backend accepts a language hint, can detect a language separately, reports a language in its transcription output, and uses prompt or prefix context. When the selected backend does not accept language hints, both rolling and full-audio transcription skip pre-transcription language detection and pass `lang=None`; rolling transcription also skips language-specific retries and gap-language probes. The backend must label returned words with its reported language or `und` when unknown. A backend may override the method for a particular model later.

`reports_language` means the adapter supplies a language determined from the transcription result. A word labeled with the language hint that Verbatim supplied does not count as reporting language. `supports_prompt` and `supports_prefix` describe distinct inputs; a backend may use them differently.

Parakeet uses NeMo with a configurable `.nemo` checkpoint (default `nvidia/parakeet-tdt-0.6b-v3`). It consumes in-memory mono 16 kHz audio and returns timestamped words in the original sample timeline. Parakeet chooses the transcription language itself and does not return per-word language labels, so its words are labeled `und`. The orchestrator therefore does not request a language hint or perform language-specific retries for this backend.

`verbastream` is a process adapter for live audio. A reader thread queues raw PCM16 mono audio from stdin while the pipeline consumes available samples through `AudioStream`. Senko's streaming session updates speaker segments after each consumed block; transcript snapshots reproject all visible words and utterances onto the latest segments. Stdout carries one `data: {JSON}` event per line, with `status`, `snapshot`, and `error` event types; diagnostic output uses stderr. Low latency mode requests three seconds at a time, with lighter transcription settings; when its input queue accumulates at least 30 seconds, it requests a larger block bounded by free attention-window space. Quality mode uses regular transcription settings and waits for enough audio to fill the 30-second rolling window before each pass.

## Conclusion

In short, multilingual support is achieved by:
- Reducing the length of each transcription to the minimum necessary to produce complete utterances.
- Detecting language using the shortest audio length that provides sufficient confidence.

By dynamically adapting transcription windows and refining language detection, Verbatim transforms Whisper into a truly multilingual ASR system, capable of handling real-world multilingual conversations with precision.
