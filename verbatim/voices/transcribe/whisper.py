import logging
from typing import List, Tuple, Union

import numpy as np
import whisper
from whisper.model import Whisper

from .transcribe import Transcriber, WhisperConfig
from ...transcript.words import VerbatimWord

LOG = logging.getLogger(__name__)


class WhisperTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model_size_or_path: str,
        device: str,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: List[float] = None,
    ):
        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        self.device = device
        self.whisper_beam_size = whisper_beam_size
        self.whisper_best_of = whisper_best_of
        self.whisper_patience = whisper_patience
        self.whisper_temperatures = whisper_temperatures
        self.model: Whisper = whisper.load_model(model_size_or_path, device=device)

    def guess_language(self, audio: np.array, lang: List[str]) -> Tuple[str, float]:
        audio = whisper.pad_or_trim(audio)
        mel_spectrogram = whisper.log_mel_spectrogram(
            audio, n_mels=self.model.dims.n_mels
        ).to(self.model.device)

        _, lang_probs = self.model.detect_language(mel=mel_spectrogram)
        best_lang = max((k for k in lang_probs if k in lang), key=lang_probs.get)

        return best_lang, lang_probs[best_lang]

    def transcribe(
        self,
        *,
        audio: np.array,
        lang: str,
        prompt: str,
        prefix: str,
        window_ts: int,
        audio_ts: int,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: List[float] = None,
    ) -> List[VerbatimWord]:
        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        use_fp16 = self.device == "cuda"
        verbose: Union[None, bool] = None
        if LOG.getEffectiveLevel() < logging.INFO:
            verbose = True
        elif LOG.getEffectiveLevel() < logging.WARN:
            verbose = False  # will still display a progress bar
        else:
            verbose = None

        whisper_config: WhisperConfig = WhisperConfig()

        options = whisper.DecodingOptions(
            task=whisper_config.task,
            language=lang,
            temperature=tuple(whisper_temperatures),
            sample_len=None,
            best_of=whisper_best_of,
            beam_size=whisper_beam_size,
            patience=whisper_patience,
            length_penalty=whisper_config.length_penalty,
            prompt=prompt,
            prefix=prefix,
            suppress_tokens=whisper_config.suppress_tokens,
            suppress_blank=whisper_config.suppress_blank,
            without_timestamps=False,
            max_initial_timestamp=1.0,
            fp16=use_fp16,
        )

        transcript = self.model.transcribe(
            audio=audio,
            word_timestamps=True,
            verbose=verbose,
            compression_ratio_threshold=whisper_config.compression_ratio_threshold,
            logprob_threshold=whisper_config.logprob_threshold,
            no_speech_threshold=whisper_config.no_speech_threshold,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            prepend_punctuations=whisper_config.prepend_punctuations,
            append_punctuations=whisper_config.append_punctuations,
            clip_timestamps=[0.0],
            hallucination_silence_threshold=None,
            **options.__dict__,
        )
        words: List[VerbatimWord] = []
        for segment in transcript["segments"]:
            # pylint: disable=unused-variable
            segment_id: int = segment.get("id")
            segment_seek: int = segment.get("seek")
            segment_start: str = segment.get("start")
            segment_end: str = segment.get("end")
            segment_text: str = segment.get("text")
            segment_temperature: float = segment.get("temperature")
            segment_avg_logprob: float = segment.get("avg_logprob")
            segment_compression_ratio: float = segment.get("compression_ratio")
            segment_no_speech_prob: float = segment.get("no_speech_prob")
            segment_words: List[str, object] = segment.get("words")
            for word in segment_words:
                word_start: float = word.get("start")
                word_end: float = word.get("end")
                word_text: str = word.get("word")
                word_probability: float = word.get("probability")

                start_ts = int(word_start * 16000) + window_ts
                end_ts = int(word_end * 16000) + window_ts
                words.append(
                    VerbatimWord(
                        start_ts=start_ts,
                        end_ts=end_ts,
                        word=word_text,
                        probability=word_probability,
                        lang=lang,
                    )
                )

        return words
