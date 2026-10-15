"""Parakeet OpenVINO conversion and session behavior without a model download."""

import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from verbatim.voices.transcribe.parakeet_openvino import (
    BUCKET_FRAMES,
    ParakeetOpenVinoTranscriber,
    _compiled_dir,
    _OpenVinoSession,
    _word_results,
    prefetch_parakeet_openvino_model,
)


class TestParakeetOpenVino(unittest.TestCase):
    def test_subword_timestamps_become_offset_words(self):
        result = SimpleNamespace(tokens=[" Hello", " there", "!"], timestamps=[0.1, 0.5, 0.7])

        words = _word_results(result, window_ts=32000, audio_ts=48000)

        self.assertEqual("Hello there!", "".join(word.word for word in words))
        self.assertEqual([(33600, 40000), (40000, 48000)], [(word.start_ts, word.end_ts) for word in words])

    def test_session_casts_decoder_inputs_and_maps_named_outputs(self):
        outputs = {
            Mock(any_name="joint"): np.zeros((1, 1, 1, 4), dtype=np.float32),
            Mock(any_name="state_a"): np.zeros((2, 1, 640), dtype=np.float32),
            Mock(any_name="state_b"): np.ones((2, 1, 640), dtype=np.float32),
        }
        request = Mock()
        request.infer.return_value = outputs
        compiled = Mock()
        compiled.create_infer_request.return_value = request
        session = _OpenVinoSession(compiled)

        results = session.run(
            ["outputs", "output_states_1", "output_states_2"],
            {"targets": [[1]], "target_length": [1], "encoder_outputs": np.zeros((1, 1024, 1), dtype=np.float32)},
        )

        self.assertEqual(3, len(results))
        self.assertEqual(np.int32, request.infer.call_args.args[0]["targets"].dtype)
        self.assertEqual(1, results[2][0, 0, 0])

    def test_prepare_exports_all_shapes_to_configured_cache(self):
        core = Mock(available_devices=["NPU"])
        core.compile_model.return_value.export_model.return_value = io.BytesIO(b"compiled")
        openvino = SimpleNamespace(Core=Mock(return_value=core), __version__="test")
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root) / "model"
            model_dir.mkdir()
            compiled_dir = Path(root) / "configured-cache" / "parakeet-openvino"
            with patch.dict(sys.modules, {"openvino": openvino}):
                with patch("verbatim.voices.transcribe.parakeet_openvino._compiled_dir", return_value=compiled_dir):
                    prefetch_parakeet_openvino_model(str(model_dir))
                    self.assertEqual(b"compiled", (compiled_dir / "decoder.blob").read_bytes())
                    for frames in BUCKET_FRAMES:
                        self.assertEqual(b"compiled", (compiled_dir / f"encoder-{frames}.blob").read_bytes())
                    prefetch_parakeet_openvino_model(str(model_dir))
        self.assertEqual(1 + len(BUCKET_FRAMES), core.compile_model.call_count)

    def test_prepared_blobs_use_model_cache_directory(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root) / "model"
            model_dir.mkdir()
            for name in ("encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"):
                (model_dir / name).write_bytes(b"model")
            cache_dir = Path(root) / "configured-cache"
            with patch.dict("os.environ", {"VERBATIM_MODELDIR": str(cache_dir)}):
                compiled_dir = _compiled_dir(model_dir, "test-version")
            self.assertEqual(cache_dir / "parakeet-openvino", compiled_dir.parent)

    def test_runtime_imports_prepared_blobs_without_compilation_or_writes(self):
        core = Mock(available_devices=["NPU"])
        core.import_model.return_value.create_infer_request.return_value = Mock()
        openvino = SimpleNamespace(Core=Mock(return_value=core), __version__="test")
        adapter = Mock()
        adapter.with_timestamps.return_value = adapter
        onnx_asr = SimpleNamespace(load_model=Mock(return_value=adapter))
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root) / "model"
            model_dir.mkdir()
            for name in ("encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"):
                (model_dir / name).write_bytes(b"model")
            compiled_dir = Path(root) / "configured-cache" / "parakeet-openvino"
            compiled_dir.mkdir(parents=True)
            for name in ("decoder.blob", *(f"encoder-{frames}.blob" for frames in BUCKET_FRAMES)):
                (compiled_dir / name).write_bytes(b"compiled")
            with patch.dict(sys.modules, {"openvino": openvino, "onnx_asr": onnx_asr}):
                with patch("verbatim.voices.transcribe.parakeet_openvino._compiled_dir", return_value=compiled_dir):
                    with patch("verbatim.voices.transcribe.parakeet_openvino.os.makedirs", side_effect=AssertionError("disk write")):
                        transcriber = ParakeetOpenVinoTranscriber(model_size_or_path=str(model_dir), device="cpu")
                        transcriber._model.asr._encoder._load(BUCKET_FRAMES[1])
        core.compile_model.assert_not_called()
        self.assertEqual(3, core.import_model.call_count)

    def test_runtime_requires_all_prepared_shapes(self):
        core = Mock(available_devices=["NPU"])
        openvino = SimpleNamespace(Core=Mock(return_value=core), __version__="test")
        with tempfile.TemporaryDirectory() as root:
            model_dir = Path(root) / "model"
            model_dir.mkdir()
            for name in ("encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"):
                (model_dir / name).write_bytes(b"model")
            with patch.dict(sys.modules, {"openvino": openvino, "onnx_asr": SimpleNamespace()}):
                with patch("verbatim.voices.transcribe.parakeet_openvino._compiled_dir", return_value=Path(root) / "missing"):
                    with self.assertRaisesRegex(RuntimeError, "--install"):
                        ParakeetOpenVinoTranscriber(model_size_or_path=str(model_dir), device="cpu")
        core.import_model.assert_not_called()

    def test_empty_window_skips_recognition(self):
        transcriber = ParakeetOpenVinoTranscriber.__new__(ParakeetOpenVinoTranscriber)
        transcriber._model = Mock()

        words = transcriber.transcribe(audio=np.array([], dtype=np.float32), lang=None, prompt="", prefix="", window_ts=0, audio_ts=0)

        self.assertEqual([], words)
        transcriber._model.recognize.assert_not_called()
