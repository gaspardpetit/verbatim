from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
from numpy.typing import NDArray

from verbatim.transcript.idprovider import CounterIdProvider
from verbatim.transcript.sentences import FastSentenceTokenizer
from verbatim.transcript.words import Utterance, Word
from verbatim.verbatim import Verbatim
from verbatim_audio.sources.ffmpegfileaudiosource import PyAVAudioSource
from verbatim_files.format.json import save_utterances

LOG = logging.getLogger(__name__)


def configure_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def choose_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch  # pylint: disable=import-outside-toplevel

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return "cpu"


def load_audio_mono(path: Path, *, chunk_seconds: int = 30) -> NDArray[np.float32]:
    source = PyAVAudioSource(file_path=str(path), target_sample_rate=16000, preserve_channels=False)
    stream = source.open()
    chunks: List[NDArray[np.float32]] = []
    try:
        while stream.has_more():
            chunk = stream.next_chunk(chunk_length=chunk_seconds)
            if chunk.size == 0:
                continue
            if len(chunk.shape) > 1:
                chunk = np.mean(chunk, axis=1)
            chunks.append(chunk.astype(np.float32, copy=False))
    finally:
        stream.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def build_utterances(words: List[Word]) -> List[Utterance]:
    if not words:
        return []
    try:
        return Verbatim.words_to_sentences(
            word_tokenizer=FastSentenceTokenizer(),
            window_words=words,
            id_provider=CounterIdProvider(prefix="utt"),
        )
    except Exception:  # pylint: disable=broad-exception-caught
        LOG.debug("Sentence alignment failed; falling back to a single utterance.", exc_info=True)
        return [Utterance.from_words(utterance_id="utt1", words=words)]


def _normalize_text_for_txt(utterances: Iterable[Utterance]) -> str:
    return "\n".join(utterance.text.strip() for utterance in utterances if utterance.text and utterance.text.strip())


def write_outputs(*, outdir: Path, stem: str, utterances: List[Utterance], metadata: dict | None = None) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{stem}.json"
    txt_path = outdir / f"{stem}.txt"

    save_utterances(str(json_path), utterances, None)
    if metadata:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload.update(metadata)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    txt_path.write_text(_normalize_text_for_txt(utterances), encoding="utf-8")
    return json_path, txt_path
