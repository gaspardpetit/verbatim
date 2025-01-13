import os
import sys
import json
import unittest

from verbatim.config import Config
from verbatim.verbatim import Verbatim
from verbatim.audio.sources.audiosource import AudioSource
from verbatim.audio.sources.sourceconfig import SourceConfig
from verbatim.audio.sources.factory import create_audio_source
from verbatim.eval.metrics import compute_metrics_on_json_dict
from verbatim.transcript.format.writer import TranscriptWriterConfig
from verbatim.transcript.format.json_dlm import JsonDiarizationLMTranscriptWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Set CUDA_VISIBLE_DEVICES to -1 to use CPU


class TestPipeline(unittest.TestCase):
    def test_diarization_metrics_short(self):
        # Setup paths
        test_file = "tests/data/init.mp3"
        ref_file = "tests/data/init.utt.ref.json"
        out_file = "tests/data/init.out"

        # Load reference data
        with open(ref_file, "r", encoding="utf-8") as f:
            ref_data = json.load(f)

        # Run verbatim pipeline
        config: Config = Config(device="cpu").configure_languages(["en"])
        audio_source: AudioSource = create_audio_source(input_source=test_file, device=config.device)
        verbatim: Verbatim = Verbatim(config=config)

        # Setup DLM JSON writer
        writer = JsonDiarizationLMTranscriptWriter(config=TranscriptWriterConfig())
        writer.open(out_file)

        # Process audio
        with audio_source.open() as audio_stream:
            for utterance, _unack_utterances, _unconfirmed_words in verbatim.transcribe(audio_stream=audio_stream):
                writer.write(utterance=utterance)

        writer.close()

        # Load output data
        with open(f"{out_file}.utt.json", "r", encoding="utf-8") as f:
            hyp_data = json.load(f)

        # Merge reference data into hypothesis data
        for h_utt, r_utt in zip(hyp_data["utterances"], ref_data["utterances"]):
            h_utt["ref_text"] = r_utt["ref_text"]
            h_utt["ref_spk"] = r_utt["ref_spk"]

        # Calculate metrics
        result = compute_metrics_on_json_dict(hyp_data)

        # Check that all error rates are 0.0
        self.assertEqual(result["WER"], 0.0)
        self.assertEqual(result["WDER"], 0.0)
        self.assertEqual(result["cpWER"], 0.0)
        self.assertEqual(result["SpkCntMAE"], 0.0)

        # Cleanup
        os.remove(f"{out_file}.utt.json")

    def test_diarization_metrics_long(self):
        # Setup paths
        test_file = "tests/data/test.mp3"
        ref_file = "tests/data/test.utt.ref.json"
        out_file = "tests/data/test.mp3.out"

        # Load reference data
        with open(ref_file, "r", encoding="utf-8") as f:
            ref_data = json.load(f)

        # Run verbatim pipeline
        config: Config = Config(device="cpu").configure_languages(["fr", "en"])
        source_config = SourceConfig(diarize=2)
        audio_source: AudioSource = create_audio_source(input_source=test_file, device=config.device, source_config=source_config)
        verbatim: Verbatim = Verbatim(config=config)

        # Setup DLM JSON writer
        writer = JsonDiarizationLMTranscriptWriter(config=TranscriptWriterConfig())
        writer.open(out_file)

        # Process audio
        with audio_source.open() as audio_stream:
            for acknowledged_utterance, _confirmed_utterance, _unconfirmed_words in verbatim.transcribe(audio_stream=audio_stream):
                writer.write(utterance=acknowledged_utterance)

        writer.close()

        # Load output data
        with open(f"{out_file}.utt.json", "r", encoding="utf-8") as f:
            hyp_data = json.load(f)

        # Merge reference data into hypothesis data
        for h_utt, r_utt in zip(hyp_data["utterances"], ref_data["utterances"]):
            h_utt["ref_text"] = r_utt["ref_text"]
            h_utt["ref_spk"] = r_utt["ref_spk"]

        # Squash everything into a single utterance
        hyp_data["utterances"] = [
            {
                "utterance_id": "utt0",
                "hyp_text": " ".join([utt["hyp_text"] for utt in hyp_data["utterances"]]),
                "hyp_spk": " ".join([utt["hyp_spk"] for utt in hyp_data["utterances"]]),
                "ref_text": " ".join([utt["ref_text"] for utt in hyp_data["utterances"]]),
                "ref_spk": " ".join([utt["ref_spk"] for utt in hyp_data["utterances"]]),
            }
        ]

        # Calculate metrics
        result = compute_metrics_on_json_dict(hyp_data)

        # Check that all error rates are below 10%
        self.assertLess(result["WER"], 0.1)
        self.assertLess(result["WDER"], 0.1)
        self.assertLess(result["cpWER"], 0.1)
        self.assertLess(result["SpkCntMAE"], 0.1)

        # Cleanup
        os.remove(f"{out_file}.utt.json")


if __name__ == "__main__":
    unittest.main()
    #config: Config = Config(device="cpu").configure_languages(["fr", "en"])
    #audio_source: AudioSource = create_audio_source(input_source="tests/data/init.mp3", device=config.device)
    #verbatim: Verbatim = Verbatim(config=config)
    #with audio_source.open() as audio_stream:
    #    for utterance, unack_utterances, unconfirmed_words in verbatim.transcribe(audio_stream=audio_stream):
    #        print(utterance.text)
