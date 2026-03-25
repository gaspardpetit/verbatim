from __future__ import annotations

import logging
from typing import List, Protocol

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)

DEFAULT_AST_AUDIO_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SILENCE_RMS_THRESHOLD = 0.01
SCORE_THRESHOLD = 0.2
TARGET_SR = 16000
CHUNK_SECONDS = 5.0
HOP_SECONDS = 2.5

AST_LABEL_MAP = {
    "Applause": "applause",
    "Clapping": "applause",
    "Cheering": "crowd",
    "Crowd": "crowd",
    "Laughter": "laughter",
    "Cough": "human_noise",
    "Throat clearing": "human_noise",
    "Breathing": "human_noise",
    "Sigh": "human_noise",
    "Music": "music",
    "Singing": "music",
    "Musical instrument": "music",
    "Typing": "mechanical_noise",
    "Computer keyboard": "mechanical_noise",
    "Door": "mechanical_noise",
    "Footsteps": "mechanical_noise",
    "Wind": "environmental_noise",
    "Rain": "environmental_noise",
    "Thunder": "environmental_noise",
}


class NonSpeechClassifierProtocol(Protocol):
    def classify(self, segment: NDArray[np.float32], sample_rate: int) -> List[str]:
        ...


def _rms(audio: NDArray[np.float32]) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


class AstNonSpeechClassifier:
    def __init__(self, *, model_name: str = DEFAULT_AST_AUDIO_MODEL, device: str = "cpu"):
        try:
            # pylint: disable=import-outside-toplevel
            import torch
            import torchaudio
            from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor
            from transformers.models.auto.modeling_auto import AutoModelForAudioClassification
        except ImportError as exc:  # pragma: no cover - exercised via runtime error path
            raise RuntimeError(
                "AST non-speech classification requires optional dependencies. Install `transformers`, `torch`, and `torchaudio`, "
                "or install the existing MMS extra."
            ) from exc

        self._torch = torch
        self._torchaudio = torchaudio
        self._device = device
        self._model_name = model_name
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)  # nosec B615 - model id is operator-configured
        self._model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)  # nosec B615 - model id is operator-configured
        self._model.eval()
        self._problem_type = getattr(self._model.config, "problem_type", None)
        LOG.info("Loaded AST non-speech classifier model=%s device=%s", model_name, device)

    def _resample(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        if sample_rate == TARGET_SR:
            return audio.astype(np.float32, copy=False)

        waveform = self._torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
        resampled = self._torchaudio.functional.resample(waveform, sample_rate, TARGET_SR)
        return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _chunk_audio(audio: NDArray[np.float32]) -> List[NDArray[np.float32]]:
        chunk_size = int(CHUNK_SECONDS * TARGET_SR)
        hop_size = int(HOP_SECONDS * TARGET_SR)
        if len(audio) <= chunk_size:
            return [audio]

        chunks: List[NDArray[np.float32]] = []
        for start in range(0, max(1, len(audio) - chunk_size + 1), hop_size):
            chunk = audio[start : start + chunk_size]
            if len(chunk) == 0:
                continue
            chunks.append(chunk)
            if start + chunk_size >= len(audio):
                break
        return chunks or [audio]

    def _scores_to_labels(self, probabilities: NDArray[np.float32]) -> List[str]:
        labels: List[str] = []
        for index, score in enumerate(probabilities):
            if score < SCORE_THRESHOLD:
                continue
            class_name = self._model.config.id2label[index]
            mapped = AST_LABEL_MAP.get(class_name)
            if mapped and mapped not in labels:
                labels.append(mapped)
        return labels or ["other"]

    def classify(self, segment: NDArray[np.float32], sample_rate: int) -> List[str]:
        segment = segment.astype(np.float32, copy=False)
        if _rms(segment) < SILENCE_RMS_THRESHOLD:
            return ["silence"]

        segment = self._resample(segment, sample_rate)
        all_probabilities: List[NDArray[np.float32]] = []
        for chunk in self._chunk_audio(segment):
            inputs = self._feature_extractor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.no_grad():
                logits = self._model(**inputs).logits
            if self._problem_type == "multi_label_classification":
                probs = self._torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                probs = self._torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float32, copy=False)
            all_probabilities.append(probs)

        mean_probabilities = np.mean(np.vstack(all_probabilities), axis=0)
        return self._scores_to_labels(mean_probabilities.astype(np.float32, copy=False))


def create_non_speech_classifier(*, backend: str, device: str, model_name: str) -> NonSpeechClassifierProtocol:
    normalized = (backend or "energy").lower()
    if normalized == "ast":
        return AstNonSpeechClassifier(model_name=model_name, device=device)
    raise RuntimeError(f"Unsupported non_speech_backend: {backend}")
