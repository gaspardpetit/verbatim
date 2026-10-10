"""Protocol and scheduling checks for the stdin/stdout stream app."""

import io
import json
import queue
import sys
import threading
import types
import unittest
import wave
from pathlib import Path
from typing import cast
from unittest.mock import patch

from verbatim.config import Config
from verbatim.transcript.words import Utterance, Word
from verbatim.verbatim import Verbatim
from verbatim_audio.sources.audiosource import AudioStream
from verbatim_stream.main import (
    BLOCK_BYTES,
    BLOCK_SAMPLES,
    BLOCK_SECONDS,
    QueuedAudioStream,
    SnapshotPublisher,
    StreamingVerbatim,
    read_blocks,
    run,
    speaker_for,
    write_event,
)


class StreamTests(unittest.TestCase):
    def test_blocking_stream_and_final_partial_block(self):
        source = io.BytesIO(b"\x00\x00" * (BLOCK_SAMPLES + 123))
        blocks = queue.Queue()
        read_blocks(source, blocks)
        sizes = []
        stream = QueuedAudioStream(blocks, lambda audio: sizes.append(len(audio)))
        self.assertTrue(stream.has_more())
        self.assertEqual(len(stream.next_chunk()), BLOCK_SAMPLES)
        self.assertTrue(stream.has_more())
        self.assertEqual(len(stream.next_chunk()), 123)
        self.assertFalse(stream.has_more())
        self.assertEqual(sizes, [BLOCK_SAMPLES, 123])
        self.assertEqual(BLOCK_SECONDS, 3)
        self.assertEqual(BLOCK_BYTES, 96000)

    def test_speaker_projection_and_unconfirmed_status(self):
        output = io.StringIO()
        publisher = SnapshotPublisher(output)
        publisher.segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"}]
        word = Word(start_ts=0, end_ts=16000, word="hello", probability=0.9, lang="en")
        unconfirmed = Word(start_ts=16000, end_ts=32000, word="there", probability=0.8, lang="en")
        utterance = Utterance.from_words("u1", [word])
        state = types.SimpleNamespace(window_ts=0, acknowledged_utterances=[utterance], unacknowledged_utterances=[], unconfirmed_words=[unconfirmed])
        publisher.publish(cast(Verbatim, types.SimpleNamespace(state=state)))
        payload = json.loads(output.getvalue().removeprefix("data: "))
        self.assertEqual(payload["utterances"][0]["speaker"], "SPEAKER_01")
        self.assertEqual(payload["unconfirmed_words"][0]["status"], "unconfirmed")
        self.assertEqual(speaker_for(0, 16000, publisher.segments), "SPEAKER_01")

    def test_snapshot_keeps_words_that_leave_the_rolling_window(self):
        output = io.StringIO()
        publisher = SnapshotPublisher(output)
        first = Word(start_ts=0, end_ts=16000, word="first", probability=0.9, lang="en")
        second = Word(start_ts=30 * 16000, end_ts=31 * 16000, word="second", probability=0.9, lang="en")
        state = types.SimpleNamespace(window_ts=0, acknowledged_utterances=[], unacknowledged_utterances=[], unconfirmed_words=[first])
        transcriber = cast(Verbatim, types.SimpleNamespace(state=state))
        publisher.publish(transcriber)
        state.window_ts = 30 * 16000
        state.unconfirmed_words = [second]
        publisher.publish(transcriber)
        state.window_ts = 32 * 16000
        publisher.publish(transcriber)
        publisher.publish(transcriber, final=True, final_utterances=[])
        snapshots = [json.loads(line.removeprefix("data: ")) for line in output.getvalue().splitlines()]
        self.assertEqual([word["text"] for word in snapshots[1]["unconfirmed_words"]], ["first", "second"])
        self.assertEqual([word["text"] for word in snapshots[2]["unconfirmed_words"]], ["first", "second"])
        self.assertEqual([word["text"] for word in snapshots[3]["unconfirmed_words"]], ["first", "second"])

    def test_snapshot_removes_retracted_current_window_word(self):
        output = io.StringIO()
        publisher = SnapshotPublisher(output)
        word = Word(start_ts=16000, end_ts=32000, word="withdrawn", probability=0.8, lang="en")
        state = types.SimpleNamespace(window_ts=0, acknowledged_utterances=[], unacknowledged_utterances=[], unconfirmed_words=[word])
        transcriber = cast(Verbatim, types.SimpleNamespace(state=state))
        publisher.publish(transcriber)
        state.unconfirmed_words = []
        publisher.publish(transcriber)
        snapshots = [json.loads(line.removeprefix("data: ")) for line in output.getvalue().splitlines()]
        self.assertEqual([item["text"] for item in snapshots[0]["unconfirmed_words"]], ["withdrawn"])
        self.assertEqual(snapshots[1]["unconfirmed_words"], [])

    def test_sample_wav_drives_full_block_scheduler(self):
        sample = Path(__file__).resolve().parents[1] / "ext/samples/audio/1ch_2spk_en-fr_AirFrance_00h03m54s.wav"
        if not sample.exists():
            self.skipTest("optional sample checkout is absent")

        class FakeSession:
            def __init__(self, _diarizer):
                self.seconds = 0.0

            def add_samples(self, audio, *, sample_rate):
                self.seconds += len(audio) / sample_rate
                return {"merged_segments": [{"start": 0.0, "end": self.seconds, "speaker": "SPEAKER_01"}]}

        class FakeVerbatim:
            def __init__(self, _config, publisher, *, quality=False):
                _ = quality
                if _config.lang != ["en", "fr"]:
                    raise AssertionError("verbastream must accept English and French")
                if _config.transcriber_backend != "whisper":
                    raise AssertionError("Whisper should remain the default model")
                if not _config.stream or _config.whisper_beam_size != 3 or _config.whisper_best_of != 3:
                    raise AssertionError("Low latency mode must use streaming transcription settings")
                self.publisher = publisher
                self.models = types.SimpleNamespace(transcriber=object())
                self.state = types.SimpleNamespace(window_ts=0, acknowledged_utterances=[], unacknowledged_utterances=[], unconfirmed_words=[])

            def transcribe(self, stream):
                while stream.has_more():
                    stream.next_chunk()
                    self.publisher.publish(self)
                yield from ()

        fake_senko = types.SimpleNamespace(Diarizer=lambda **_kwargs: object(), DiarizationSession=FakeSession)
        output = io.StringIO()
        with wave.open(str(sample), "rb") as wav_file:
            self.assertEqual((wav_file.getnchannels(), wav_file.getframerate(), wav_file.getsampwidth()), (1, 16000, 2))
            raw = wav_file.readframes(wav_file.getnframes())
        with patch.dict(sys.modules, {"senko": fake_senko}), patch("verbatim_stream.main.StreamingVerbatim", FakeVerbatim):
            run(io.BytesIO(raw), output, device="cpu")
        messages = [json.loads(line.removeprefix("data: ")) for line in output.getvalue().splitlines()]
        snapshots = [message for message in messages if message["type"] == "snapshot"]
        statuses = [message["stage"] for message in messages if message["type"] == "status"]
        expected_blocks = (len(raw) + BLOCK_BYTES - 1) // BLOCK_BYTES
        self.assertEqual(len(snapshots), expected_blocks + 1)
        self.assertEqual(snapshots[0]["audio_seconds"], 3.0)
        self.assertEqual(snapshots[-1]["audio_seconds"], len(raw) / 32000)
        self.assertTrue(snapshots[-1]["final"])
        self.assertEqual(statuses, ["initializing", "loading_diarizer", "loading_transcriber", "ready",
                                    "waiting_for_audio", "transcribing", "finalizing", "complete"])
        self.assertLess(messages.index(next(message for message in messages if message.get("stage") == "ready")),
                        messages.index(snapshots[0]))
        self.assertEqual(messages[-1]["stage"], "complete")

    def test_quality_mode_requests_full_window_then_waits_for_remaining_audio(self):
        seen = []

        class FakeVerbatim:
            def __init__(self, config, publisher, *, quality=False):
                _ = quality
                seen.append((config.lang, config.transcriber_backend, config.stream, config.window_duration,
                             config.chunk_table, config.whisper_beam_size))
                self.publisher = publisher
                self.models = types.SimpleNamespace(transcriber=object())
                self.state = types.SimpleNamespace(window_ts=0, acknowledged_utterances=[], unacknowledged_utterances=[], unconfirmed_words=[])

            def transcribe(self, stream):
                while stream.has_more():
                    stream.next_chunk(30)
                    self.publisher.publish(self)
                yield from ()

        fake_senko = types.SimpleNamespace(Diarizer=lambda **_kwargs: object(),
                                           DiarizationSession=lambda _diarizer: types.SimpleNamespace(
                                               add_samples=lambda audio, sample_rate: {"merged_segments": []}))
        output = io.StringIO()
        with patch.dict(sys.modules, {"senko": fake_senko}), patch("verbatim_stream.main.StreamingVerbatim", FakeVerbatim):
            run(io.BytesIO(b"\x00\x00" * (37 * 16000)), output, device="cpu",
                languages=["fr"], mode="quality", backend="qwen")
        messages = [json.loads(line.removeprefix("data: ")) for line in output.getvalue().splitlines()]
        self.assertEqual([message["audio_seconds"] for message in messages if message["type"] == "snapshot"],
                         [30.0, 37.0, 37.0])
        self.assertEqual(seen, [(["fr"], "qwen", False, 30, [(0, 1.0)], 9)])

    def test_quality_mode_releases_short_recording_at_eof(self):
        blocks = queue.Queue()
        read_blocks(io.BytesIO(b"\x00\x00" * (7 * 16000)), blocks)
        stream = QueuedAudioStream(blocks, lambda _audio: None, 30)
        self.assertEqual(len(stream.next_chunk(30)), 7 * 16000)
        self.assertFalse(stream.has_more())

    def test_protocol_error_event(self):
        output = io.StringIO()
        write_event(output, {"type": "error", "message": "model failed"})
        self.assertEqual(json.loads(output.getvalue().removeprefix("data: ")),
                         {"type": "error", "message": "model failed"})

    def test_rejects_non_16khz_internal_sample_rate(self):
        with patch("verbatim_stream.main.AUDIO_PARAMS", types.SimpleNamespace(sample_rate=8000)):
            with self.assertRaisesRegex(RuntimeError, "VERBATIM_SAMPLE_RATE=16000"):
                run(io.BytesIO(), io.StringIO(), device="cpu")

    def test_stream_waits_for_requested_window_while_input_continues(self):
        blocks = queue.Queue()
        stream = QueuedAudioStream(blocks, lambda _audio: None, 30)
        result = []
        finished = threading.Event()

        def consume():
            result.append(len(stream.next_chunk(30)))
            finished.set()

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        blocks.put(b"\x00\x00" * (10 * 16000))
        self.assertFalse(finished.wait(0.05))
        blocks.put(b"\x00\x00" * (20 * 16000))
        self.assertTrue(finished.wait(1))
        self.assertEqual(result, [30 * 16000])
        blocks.put(None)

    def test_backend_failure_does_not_wait_for_full_input_queue(self):
        class FailingVerbatim:
            def __init__(self, _config, _publisher, *, quality=False):
                _ = quality
                self.models = types.SimpleNamespace(transcriber=object())

            def transcribe(self, stream):
                stream.next_chunk()
                yield from ()

        fake_senko = types.SimpleNamespace(Diarizer=lambda **_kwargs: object(),
                                           DiarizationSession=lambda _diarizer: types.SimpleNamespace(
                                               add_samples=lambda audio, sample_rate: {"merged_segments": []}))
        with patch.dict(sys.modules, {"senko": fake_senko}), patch("verbatim_stream.main.StreamingVerbatim", FailingVerbatim):
            with self.assertRaisesRegex(RuntimeError, "stopped before consuming all input"):
                run(io.BytesIO(b"\x00\x00" * (105 * 16000)), io.StringIO(), device="cpu")

    def test_streaming_transcriber_propagates_error_after_input_eof(self):
        transcriber = object.__new__(StreamingVerbatim)
        transcriber.config = Config(device="cpu")
        transcriber.raise_on_transcription_error = True
        transcriber.status_hook = None
        blocks = queue.Queue()
        read_blocks(io.BytesIO(b"\x00\x00" * 16000), blocks)
        stream = QueuedAudioStream(blocks, lambda _audio: None)

        def fail_after_eof(audio_source):
            self.assertIs(audio_source, stream)
            stream.next_chunk()
            raise RuntimeError("backend failed on final partial block")

        with patch.object(StreamingVerbatim, "capture_audio", side_effect=fail_after_eof):
            with self.assertRaisesRegex(RuntimeError, "backend failed on final partial block"):
                list(transcriber.transcribe(stream))
        self.assertTrue(stream.ended)

    def test_quality_mode_refills_the_rolling_window(self):
        transcriber = object.__new__(StreamingVerbatim)
        transcriber.config = cast(Config, types.SimpleNamespace(sampling_rate=16000, window_duration=30, chunk_table=[(0, 1.0)]))
        transcriber.state = types.SimpleNamespace(window_ts=0, audio_ts=0)
        transcriber.quality = True
        with patch.object(Verbatim, "capture_audio", return_value=True):
            self.assertTrue(transcriber.capture_audio(cast(AudioStream, None)))
        self.assertEqual(transcriber.get_next_number_of_chunks(), 30)
        transcriber.state.audio_ts = 10 * 16000
        with patch.object(Verbatim, "capture_audio", return_value=True):
            self.assertTrue(transcriber.capture_audio(cast(AudioStream, None)))
        self.assertEqual(transcriber.get_next_number_of_chunks(), 20)
        transcriber.state.audio_ts = 30 * 16000
        with patch.object(Verbatim, "capture_audio", return_value=True):
            self.assertTrue(transcriber.capture_audio(cast(AudioStream, None)))
        self.assertEqual(transcriber.get_next_number_of_chunks(), 1)

    def test_backlog_uses_free_space_in_attention_window(self):
        blocks = queue.Queue()
        for _ in range(300):
            blocks.put(b"\x00\x00" * 1600)
        audio_stream = QueuedAudioStream(blocks, lambda _audio: None)
        transcriber = object.__new__(StreamingVerbatim)
        transcriber.config = cast(Config, types.SimpleNamespace(sampling_rate=16000, window_duration=30, chunk_table=[(0, 0.1)]))
        transcriber.state = types.SimpleNamespace(window_ts=0, audio_ts=6 * 16000)
        transcriber.quality = False

        with patch.object(Verbatim, "capture_audio", return_value=True):
            self.assertTrue(transcriber.capture_audio(audio_stream))

        self.assertEqual(transcriber.get_next_number_of_chunks(), 24)

    def test_parakeet_routes_through_core_backend(self):
        selected = []

        class FakeVerbatim:
            def __init__(self, config, publisher, *, quality=False):
                _ = quality
                selected.append((config.transcriber_backend, config.parakeet_model_size))
                self.publisher = publisher
                self.models = types.SimpleNamespace(transcriber=object())
                self.state = types.SimpleNamespace(window_ts=0, acknowledged_utterances=[], unacknowledged_utterances=[], unconfirmed_words=[])

            def transcribe(self, stream):
                while stream.has_more():
                    stream.next_chunk()
                    self.publisher.publish(self)
                yield from ()

        fake_senko = types.SimpleNamespace(Diarizer=lambda **_kwargs: object(),
                                           DiarizationSession=lambda _diarizer: types.SimpleNamespace(
                                               add_samples=lambda audio, sample_rate: {"merged_segments": []}))
        with patch.dict(sys.modules, {"senko": fake_senko}), patch("verbatim_stream.main.StreamingVerbatim", FakeVerbatim):
            run(io.BytesIO(b"\x00\x00" * 16000), io.StringIO(), device="cpu", backend="parakeet", model="nvidia/parakeet-tdt-0.6b-v2")

        self.assertEqual(selected, [("parakeet", "nvidia/parakeet-tdt-0.6b-v2")])


if __name__ == "__main__":
    unittest.main()
