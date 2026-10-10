"""Transcribe raw 16 kHz mono PCM from stdin into revisable JSON snapshots."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import queue
import sys
import threading
from typing import Any, BinaryIO, Callable, Iterator, Optional, TextIO

import numpy as np

from verbatim.config import Config
from verbatim.transcript.words import Utterance, Word
from verbatim.verbatim import Verbatim
from verbatim_audio.settings import AUDIO_PARAMS
from verbatim_audio.sources.audiosource import AudioStream

SAMPLE_RATE = 16000
LOW_LATENCY_BLOCK_SECONDS = 3
QUALITY_WINDOW_SECONDS = 30
BLOCK_SECONDS = LOW_LATENCY_BLOCK_SECONDS
BLOCK_SAMPLES = BLOCK_SECONDS * SAMPLE_RATE
BLOCK_BYTES = BLOCK_SAMPLES * 2
LOG = logging.getLogger(__name__)


def write_event(output: TextIO, payload: dict[str, Any]) -> None:
    """Write one protocol event without buffering it behind model work."""
    output.write("data: " + json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")
    output.flush()


def write_status(output: TextIO, stage: str, **details: Any) -> None:
    write_event(output, {"type": "status", "stage": stage, **details})


class QueuedAudioStream(AudioStream):
    """Wait for the recognizer's requested sample count or input EOF."""

    def __init__(self, blocks: "queue.Queue[Optional[bytes]]", on_audio: Callable[[np.ndarray], None], block_seconds: int = BLOCK_SECONDS):
        super().__init__(start_offset=0, diarization=None)
        self.blocks = blocks
        self.on_audio = on_audio
        self.block_seconds = block_seconds
        self.pending = bytearray()
        self.ended = False
        self.total_samples = 0

    def has_more(self) -> bool:
        if self.pending:
            return True
        if self.ended:
            return False
        next_bytes = self.blocks.get()
        if next_bytes is None:
            self.ended = True
            return False
        self.pending.extend(next_bytes)
        return True

    def next_chunk(self, chunk_length=None) -> np.ndarray:
        if not self.has_more():
            return np.empty(0, dtype=np.float32)
        requested_bytes = int((chunk_length or self.block_seconds) * SAMPLE_RATE * 2)
        while len(self.pending) < requested_bytes and not self.ended:
            next_bytes = self.blocks.get()
            if next_bytes is None:
                self.ended = True
            else:
                self.pending.extend(next_bytes)
        part = bytes(self.pending[:requested_bytes])
        del self.pending[:requested_bytes]
        audio = np.frombuffer(part, dtype="<i2").astype(np.float32) / 32768.0
        self.total_samples += len(audio)
        self.on_audio(audio)
        return audio

    def get_nchannels(self) -> int:
        return 1

    def get_rate(self) -> int:
        return SAMPLE_RATE

    def close(self) -> None:
        pass

    def buffered_seconds(self) -> float:
        """Approximate audio queued behind the current transcription window."""
        return (len(self.pending) + self.blocks.qsize() * (SAMPLE_RATE * 2 // 10)) / (SAMPLE_RATE * 2)


def read_blocks(source: BinaryIO, blocks: "queue.Queue[Optional[bytes]]") -> None:
    """Queue PCM as it arrives; the consumer controls window boundaries."""
    pending = bytearray()
    try:
        while chunk := source.read(SAMPLE_RATE * 2 // 10):
            pending.extend(chunk)
            complete_bytes = len(pending) & ~1
            if complete_bytes:
                blocks.put(bytes(pending[:complete_bytes]))
                del pending[:complete_bytes]
        if len(pending) % 2:
            raise ValueError("stdin ended halfway through a PCM16 sample")
    finally:
        blocks.put(None)


def speaker_for(start: int, end: int, segments: list[dict[str, Any]]) -> Optional[str]:
    """Choose the speaker with the most temporal overlap."""
    votes: dict[str, float] = {}
    for segment in segments:
        overlap = max(0.0, min(end / SAMPLE_RATE, float(segment["end"])) - max(start / SAMPLE_RATE, float(segment["start"])))
        if overlap:
            speaker = str(segment["speaker"])
            votes[speaker] = votes.get(speaker, 0.0) + overlap
    return max(votes, key=votes.get) if votes else None  # type: ignore[arg-type]


def word_payload(word: Word) -> dict[str, Any]:
    return {
        "start": word.start_ts / SAMPLE_RATE,
        "end": word.end_ts / SAMPLE_RATE,
        "text": word.word,
        "probability": word.probability,
        "language": word.lang,
    }


class SnapshotPublisher:
    """Project each complete transcript state onto the latest Senko result."""

    def __init__(self, output: TextIO):
        self.output = output
        self.revision = 0
        self.segments: list[dict[str, Any]] = []
        self.processed_samples = 0
        self.provisional_words: list[Word] = []

    def set_diarization(self, audio: np.ndarray, session: Any) -> None:
        result = session.add_samples(audio, sample_rate=SAMPLE_RATE)
        self.segments = result.get("merged_segments", []) if result else []
        self.processed_samples += len(audio)

    def publish(self, transcriber: Verbatim, *, final: bool = False, final_utterances: Optional[list[Utterance]] = None) -> None:
        state = transcriber.state
        self.revision += 1

        def utterance_payload(utterance: Utterance, status: str) -> dict[str, Any]:
            return {
                "id": utterance.utterance_id,
                "status": status,
                "start": utterance.start_ts / SAMPLE_RATE,
                "end": utterance.end_ts / SAMPLE_RATE,
                "speaker": speaker_for(utterance.start_ts, utterance.end_ts, self.segments),
                "text": utterance.text,
                "words": [word_payload(word) for word in utterance.words],
            }

        confirmed = final_utterances if final_utterances is not None else state.acknowledged_utterances
        visible_utterances = confirmed + ([] if final else state.unacknowledged_utterances)
        current_words = state.unconfirmed_words
        if current_words:
            window_start = getattr(state, "window_ts", 0)
            self.provisional_words = [word for word in self.provisional_words if word.end_ts <= window_start]
            for word in current_words:
                if word not in self.provisional_words:
                    self.provisional_words.append(word)
        for utterance in visible_utterances:
            self.provisional_words = [
                word for word in self.provisional_words
                if word.end_ts <= utterance.start_ts or word.start_ts >= utterance.end_ts
            ]
        payload = {
            "type": "snapshot",
            "revision": self.revision,
            "audio_seconds": self.processed_samples / SAMPLE_RATE,
            "final": final,
            "utterances": [utterance_payload(item, "confirmed") for item in confirmed]
            + ([] if final else [utterance_payload(item, "pending") for item in state.unacknowledged_utterances]),
            "unconfirmed_words": [
                {**word_payload(word), "status": "unconfirmed", "speaker": speaker_for(word.start_ts, word.end_ts, self.segments)}
                for word in self.provisional_words
            ],
        }
        write_event(self.output, payload)


class StreamingVerbatim(Verbatim):
    """Publish the complete state after each available audio block."""

    def __init__(self, config: Config, publisher: SnapshotPublisher, *, quality: bool = False, transcriber: Any = None):
        super().__init__(config, transcriber=transcriber)
        self.publisher = publisher
        self.quality_initial_window = quality

    def capture_audio(self, audio_source: AudioStream):
        if isinstance(audio_source, QueuedAudioStream) and audio_source.buffered_seconds() >= QUALITY_WINDOW_SECONDS:
            occupied_seconds = (self.state.audio_ts - self.state.window_ts) / self.config.sampling_rate
            available_seconds = max(1, self.config.window_duration - occupied_seconds)
            self.config.chunk_table = [(0, min(QUALITY_WINDOW_SECONDS, available_seconds) / self.config.window_duration)]
        elif not self.quality_initial_window:
            self.config.chunk_table = [(0, LOW_LATENCY_BLOCK_SECONDS / self.config.window_duration)]
        captured = super().capture_audio(audio_source)
        if captured and self.quality_initial_window:
            self.quality_initial_window = False
            self.config.chunk_table = [(0, LOW_LATENCY_BLOCK_SECONDS / self.config.window_duration)]
        return captured

    def process_audio_window(self, audio_stream: AudioStream) -> Iterator[tuple[Utterance, list[Utterance], list[Word]]]:
        yield from super().process_audio_window(audio_stream)
        self.publisher.publish(self)


def run(source: BinaryIO, output: TextIO, *, device: str = "auto", model: str = "large-v3",
        languages: Optional[list[str]] = None, mode: str = "low-latency", backend: str = "whisper") -> None:
    write_status(output, "initializing")
    if AUDIO_PARAMS.sample_rate != SAMPLE_RATE:
        raise RuntimeError(f"verbastream requires VERBATIM_SAMPLE_RATE={SAMPLE_RATE}; configured rate is {AUDIO_PARAMS.sample_rate} Hz")
    try:
        import senko  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError('Install the Senko backend with `uv pip install ".[senko]"`') from exc

    config = Config(device=device, stream=mode != "quality", working_dir=None, transcriber_backend=backend)
    config.configure_device(device)
    config.configure_latency(mode != "quality")
    if backend == "parakeet":
        config.parakeet_model_size = model if model != "large-v3" else "nvidia/parakeet-tdt-0.6b-v3"
    elif backend == "qwen":
        if model != "large-v3":
            config.qwen_asr_model_size = model
    else:
        config.whisper_model_size = model
    config.configure_languages(languages or ["en", "fr"])
    # Quality mode starts with a full window, then revises it as new audio arrives.
    block_seconds = LOW_LATENCY_BLOCK_SECONDS
    if mode == "quality":
        config.window_duration = QUALITY_WINDOW_SECONDS
    config.chunk_table = [(0, (QUALITY_WINDOW_SECONDS if mode == "quality" else block_seconds)
                           / config.window_duration)]
    if mode == "quality":
        config.whisper_best_of = 9
        config.whisper_beam_size = 9
        config.whisper_patience = 2.0
    publisher = SnapshotPublisher(output)
    write_status(output, "loading_diarizer")
    diarizer = senko.Diarizer(device="coreml" if config.device == "mps" else config.device, warmup=True, quiet=True)
    session = senko.DiarizationSession(diarizer)
    blocks: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=1024)
    started = False

    def on_audio(audio: np.ndarray) -> None:
        nonlocal started
        if not started:
            write_status(output, "transcribing")
            started = True
        publisher.set_diarization(audio, session)

    stream = QueuedAudioStream(blocks, on_audio, QUALITY_WINDOW_SECONDS if mode == "quality" else block_seconds)
    selected_model = config.parakeet_model_size if backend == "parakeet" else (config.qwen_asr_model_size if backend == "qwen" else model)
    write_status(output, "loading_transcriber", backend=backend, model=selected_model)
    transcriber = StreamingVerbatim(config, publisher, quality=mode == "quality")
    # Load the ASR model while the desktop app is preparing playback, before
    # the first three seconds of audio are ready to transcribe.
    _ = transcriber.models.transcriber
    write_status(output, "ready")
    producer_error: list[BaseException] = []

    def produce() -> None:
        try:
            read_blocks(source, blocks)
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            producer_error.append(exc)

    producer = threading.Thread(target=produce, name="verbastream-input", daemon=True)
    producer.start()
    write_status(output, "waiting_for_audio")
    finalized: list[Utterance] = []
    for utterance, _pending, _unconfirmed in transcriber.transcribe(stream):
        finalized.append(utterance)
    if not stream.ended:
        # Verbatim currently logs and swallows backend exceptions. Its input
        # producer may then block forever on a full queue if we wait for it.
        raise RuntimeError("Transcription stopped before consuming all input; check the preceding model error in stderr")
    producer.join()
    if producer_error:
        raise producer_error[0]
    write_status(output, "finalizing")
    publisher.publish(transcriber, final=True, final_utterances=finalized)
    write_status(output, "complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream raw 16 kHz mono signed 16-bit little-endian PCM to JSON transcript snapshots")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--backend", choices=("whisper", "qwen", "parakeet"), default="whisper")
    parser.add_argument("--language", action="append", choices=("en", "fr"), dest="languages")
    parser.add_argument("--mode", choices=("low-latency", "quality"), default="low-latency")
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    try:
        protocol_stdout = sys.stdout
        with contextlib.redirect_stdout(sys.stderr):
            run(sys.stdin.buffer, protocol_stdout, device=args.device, model=args.model,
                languages=args.languages, mode=args.mode, backend=args.backend)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        write_event(protocol_stdout, {"type": "error", "message": str(exc)})
        LOG.exception("verbastream failed")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
