import logging

import numpy as np
import scipy.io.wavfile
import torch
from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio.utils.multi_task import map_with_specifications
from pyannote.audio.utils.reproducibility import fix_reproducibility
from pyannote.core.annotation import Annotation
from pyannote.database.util import load_rttm
from scipy.spatial.distance import cdist

from ..audio.audio import wav_to_int16
from ..transcript.words import VerbatimUtterance

# Configure logger
LOG = logging.getLogger(__name__)

class Diarization:
    def __init__(self, device:str, huggingface_token:str, use_ami:bool= False):
        LOG.info("Initializing Diarization Pipeline.")
        self.huggingface_token = huggingface_token
        self._use_ami = use_ami
        if self._use_ami:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speech-separation-ami-1.0",
                use_auth_token=self.huggingface_token
            )
            hyper_parameters =         {
                    "segmentation": {
                    "min_duration_off": 0.0,
                    "threshold": 0.82
                    },
                    "clustering": {
                        "method": "centroid",
                        "min_cluster_size": 15,
                        "threshold": 0.68,
                    },
                    "separation": {
                        "leakage_removal": True,
                        "asr_collar": 0.32,
                    }
                }

            self.pipeline.instantiate(hyper_parameters)
        else:
            self.pipeline = Pipeline.from_pretrained(
                checkpoint_path="pyannote/speaker-diarization-3.1",
                use_auth_token=self.huggingface_token
            )
            self.pipeline.instantiate({})

        self.pipeline.to(torch.device(device))

    @staticmethod
    def load_diarization(rttm_file:str):
        rttms = load_rttm(file_rttm=rttm_file)
        annotation:Annotation = next(iter(rttms.values()))
        return annotation

    # pylint: disable=unused-argument
    def compute_diarization(self, file_path:str, out_rttm_file:str = None, nb_speakers:int=None) -> Annotation:
        if not out_rttm_file:
            out_rttm_file = "out.rttm"

        sources = None
        with ProgressHook() as hook:
            if self._use_ami:
                diarization, sources = self.pipeline(file_path, hook=hook)
            else:
                diarization = self.pipeline(file_path, hook=hook)

        # dump the diarization output to disk using RTTM format
        with open(out_rttm_file, "w", encoding="utf-8") as rttm:
            diarization.write_rttm(rttm)

        if sources:
            # dump sources to disk as SPEAKER_XX.wav files
            for s, speaker in enumerate(diarization.labels()):
                if s < sources.data.shape[1]:
                    speaker_data = sources.data[:, s]
                    if speaker_data.dtype != np.int16:
                        speaker_data = wav_to_int16(speaker_data)
                    scipy.io.wavfile.write(f'{speaker}.wav', 16000, speaker_data)
                else:
                    LOG.debug(f"Skipping speaker {s} as it is out of bounds.")
        return diarization

    def diarize_utterance(self, u:VerbatimUtterance, window_ts:int, audio:np.array, speaker_embeddings):
        rel_start = max(0, u.start_ts - window_ts)
        rel_end = max(0, u.end_ts - window_ts)
        # compute embedding and compare
        model = Model.from_pretrained(checkpoint="pyannote/embedding",
                                      use_auth_token=self.huggingface_token)

        inference = Inference(model, window="whole")
        fix_reproducibility(inference.device)
        inference.to(torch.device("cuda"))
        audio_segment = torch.tensor(audio[rel_start:rel_end].reshape(1, 1, -1),
                                     device='cuda')

        output = inference.infer(audio_segment)

        # pylint: disable=unused-argument
        def __first_sample(outputs: np.ndarray, **kwargs) -> np.ndarray:
            return outputs[0]

        embedding = map_with_specifications(inference.model.specifications, __first_sample, output)
        embedding = embedding.reshape(1, -1)

        best_i = -1
        best_distance = 1
        for i, speaker_embedding in enumerate(speaker_embeddings):
            distance = cdist(embedding, speaker_embedding, metric="cosine")[0, 0]
            if distance < best_distance:
                best_distance = distance
                best_i = i
        if best_i == -1:
            speaker_embeddings.append(embedding)
            print("Added first speaker")
        else:
            if best_distance < 0.15:
                print(f"Detected speaker {best_i}")
            else:
                print(f"Added speaker {best_i}")
                speaker_embeddings.append(embedding)
