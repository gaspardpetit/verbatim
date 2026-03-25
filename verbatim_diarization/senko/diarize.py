import logging
import os
from typing import Any, Dict, List, Optional

import soundfile as sf

from verbatim.cache import ArtifactCache, FileBackedArtifactCache
from verbatim_audio.convert import convert_bytes_to_wav
from verbatim_diarization.diarize.base import DiarizationStrategy
from verbatim_diarization.utils import sanitize_uri_component
from verbatim_files.rttm import Annotation, Segment, dumps_rttm
from verbatim_files.vttm import AudioRef, dumps_vttm

LOG = logging.getLogger(__name__)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
    return bool(value)


class SenkoDiarization(DiarizationStrategy):
    def __init__(
        self,
        *,
        cache: ArtifactCache,
        device: str,
        warmup: bool = True,
        quiet: bool = True,
        vad: str = "auto",
        clustering: str = "auto",
        accurate: Optional[bool] = None,
        mer_cos: Optional[float] = None,
    ):
        super().__init__(cache=cache)
        self.device = self._resolve_device(device)
        self.warmup = warmup
        self.quiet = quiet
        self.vad = vad
        self.clustering = clustering
        self.accurate = None if accurate is None else _parse_bool(accurate)
        self.mer_cos = mer_cos
        self._diarizer: Any = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "mps":
            return "coreml"
        if device in ("cpu", "cuda"):
            return device
        return "auto"

    def _get_diarizer(self) -> Any:
        if self._diarizer is not None:
            return self._diarizer

        try:
            import senko  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "Senko diarization requires the optional dependency. Install it with `pip install \"verbatim[senko]\"` "
                "or `uv pip install \"git+https://github.com/narcotic-sh/senko.git\"`."
            ) from exc

        self._diarizer = senko.Diarizer(
            device=self.device,
            vad=self.vad,
            clustering=self.clustering,
            warmup=self.warmup,
            quiet=self.quiet,
            mer_cos=self.mer_cos,
        )
        return self._diarizer

    @staticmethod
    def _is_compatible_wav(path: str) -> bool:
        if not path.lower().endswith(".wav") or not os.path.exists(path):
            return False

        info = sf.info(path)
        subtype = (info.subtype or "").upper()
        return info.samplerate == 16000 and info.channels == 1 and subtype == "PCM_16"

    def _prepare_wav_path(self, file_path: str) -> tuple[str, Optional[str]]:
        if self._is_compatible_wav(file_path):
            return file_path, None

        if not isinstance(self.cache, FileBackedArtifactCache) or not self.cache.base_dir:
            raise RuntimeError(
                "Senko diarization requires an on-disk 16kHz mono PCM WAV input. "
                "Use an already-normalized WAV file or provide --workdir so Verbatim can materialize a temporary WAV."
            )

        input_bytes = self.cache.get_bytes(file_path)
        if not input_bytes:
            raise RuntimeError(f"Cached bytes missing for input '{file_path}'. Populate the artifact cache before invoking Senko.")

        base_name = sanitize_uri_component(os.path.splitext(os.path.basename(file_path))[0], fallback="senko")
        temp_path = f"{base_name}-senko.wav"
        materialized_path = convert_bytes_to_wav(
            input_bytes=input_bytes,
            input_label=file_path,
            working_prefix_no_ext=os.path.join(self.cache.base_dir, f"{base_name}-senko"),
            preserve_channels=False,
            cache=self.cache,
        )
        return materialized_path or temp_path, materialized_path

    @staticmethod
    def _segments_to_annotation(segments: List[Dict[str, Any]], *, file_id: str) -> Annotation:
        annotation_segments = []
        for segment in segments:
            start = float(segment["start"])
            end = float(segment["end"])
            speaker = str(segment["speaker"])
            annotation_segments.append(Segment(start=start, end=end, speaker=speaker, file_id=file_id))
        return Annotation(segments=annotation_segments, file_id=file_id)

    # pylint: disable=too-many-positional-arguments
    def compute_diarization(
        self,
        file_path: str,
        out_rttm_file: Optional[str] = None,
        out_vttm_file: Optional[str] = None,
        status_hook=None,
        nb_speakers: Optional[int] = None,
        **kwargs,
    ) -> Annotation:
        _ = status_hook
        if nb_speakers is not None:
            LOG.warning("Senko diarization does not accept a fixed speaker-count hint; ignoring nb_speakers=%s", nb_speakers)

        accurate = self.accurate
        if "accurate" in kwargs and kwargs["accurate"] is not None:
            accurate = _parse_bool(kwargs["accurate"])

        wav_path, temp_path = self._prepare_wav_path(file_path)
        try:
            diarizer = self._get_diarizer()
            result = diarizer.diarize(wav_path, accurate=accurate, generate_colors=False)
        except Exception as exc:
            raise RuntimeError(f"Senko diarization failed for '{file_path}': {exc}") from exc
        finally:
            if temp_path:
                self.cache.delete(temp_path)

        if result is None:
            return Annotation()

        merged_segments = result.get("merged_segments", [])
        file_id = sanitize_uri_component(os.path.splitext(os.path.basename(file_path))[0])
        annotation = self._segments_to_annotation(merged_segments, file_id=file_id)

        timing_stats = result.get("timing_stats")
        if timing_stats:
            LOG.info("Senko timing stats for %s: %s", file_path, timing_stats)

        if out_rttm_file:
            self.cache.set_text(out_rttm_file, dumps_rttm(annotation))

        if out_vttm_file:
            audio_refs = [AudioRef(id=file_id, path=file_path, channels=None)]
            self.cache.set_text(out_vttm_file, dumps_vttm(audio=audio_refs, annotation=annotation))

        return annotation
