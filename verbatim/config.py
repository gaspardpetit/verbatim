import logging
import os
import re
import errno

from dataclasses import dataclass, field
from typing import Dict, List, Union

from pyannote.core.annotation import Annotation

from .audio.sources.audiosource import AudioSource
from .transcript.format.writer import TranscriptWriterConfig

LOG = logging.getLogger(__name__)

def timestr_to_sample(timestr: str, sample_rate: int = 16000) -> int:
    """
    Converts a time string in the format hh:mm:ss.ms, mm:ss.ms, or ss.ms
    (milliseconds optional) to the corresponding sample index.

    Args:
        timestr (str): Time string in the format hh:mm:ss.ms, mm:ss.ms, or ss.ms.
        sample_rate (int): Sampling rate in Hz (default is 16000).

    Returns:
        int: The corresponding sample index.
    """
    # Define regex patterns for specific formats
    hh_mm_ss_ms_pattern = re.compile(
        r"^(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)(?:\.(?P<milliseconds>\d+))?$"
    )
    mm_ss_ms_pattern = re.compile(
        r"^(?P<minutes>\d+):(?P<seconds>\d+)(?:\.(?P<milliseconds>\d+))?$"
    )
    ss_ms_pattern = re.compile(
        r"^(?P<seconds>\d+)(?:\.(?P<milliseconds>\d+))?$"
    )

    # Match the input against patterns
    if match := hh_mm_ss_ms_pattern.match(timestr.strip()):
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        milliseconds = int(match.group("milliseconds")) if match.group("milliseconds") else 0
    elif match := mm_ss_ms_pattern.match(timestr.strip()):
        hours = 0
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        milliseconds = int(match.group("milliseconds")) if match.group("milliseconds") else 0
    elif match := ss_ms_pattern.match(timestr.strip()):
        hours = 0
        minutes = 0
        seconds = int(match.group("seconds"))
        milliseconds = int(match.group("milliseconds")) if match.group("milliseconds") else 0
    else:
        raise ValueError(f"Invalid time string format: {timestr}")

    # Calculate total time in seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    # Convert to sample index
    return int(total_seconds * sample_rate)


@dataclass
class Config:
    sampling_rate: int = 16000
    transcribe_latency: int = 16000
    frames_per_buffer: int = 1000
    window_duration: int = 30  # seconds
    whisper_prompts: Dict[str, str] = None
    lang: List[str] = field(default_factory=list)
    start_time:int = 0
    stop_time:Union[None,int] = None
    source_stream: AudioSource = None
    input_source:str = None
    isolate:Union[None,bool] = None
    diarization: Annotation = None
    diarization_file: str = None
    whisper_model_size: str = "large-v3"
    device: str = "cuda"
    stream: bool = False
    diarize:Union[int,None] = None
    whisper_beam_size: int = 9
    whisper_best_of: int = 9
    whisper_patience: float = 2.0
    whisper_temperatures: List[float] = None
    debug: bool = False
    working_dir: str = "."
    output_dir: str = "."
    enable_ass: bool = False
    enable_docx: bool = False
    enable_txt: bool = False
    enable_md: bool = False
    enable_json: bool = False
    enable_stdout: bool = True
    enable_stdout_nocolor: bool = False
    write_config:TranscriptWriterConfig = field(default=TranscriptWriterConfig)
    output_prefix_no_ext:str = "out"
    working_prefix_no_ext:str = "out"

    def __init__(
            self, *,
            input_source:Union[None,str],
            outdir:Union[None,str] = ".", workdir:Union[None,str] = None,
            use_cpu:Union[None, bool] = None,
            stream:Union[None,bool] = False, isolate:Union[None,bool]=None, diarize:Union[None,int] = None,
            start_time:Union[None,str] = "0", stop_time:Union[None,str] = None
            ):

        self.chunk_table = [
            (0.75, 0.20),
            (0.50, 0.15),
            (0.25, 0.10),
            (0.10, 0.05),
            (0.00, 0.025),
        ]
        self.lang = ["en"]
        self.whisper_prompts = {
            "en": "This is a sentence.",
            "zh": "这是一个句子。",
            "de": "Dies ist ein Satz.",
            "es": "Esta es una oración.",
            "ru": "Это предложение.",
            "ko": "이것은 문장입니다.",
            "fr": "Ceci est une phrase.",
            "ja": "これは文です。",
            "pt": "Esta é uma frase.",
            "tr": "Bu bir cümledir.",
            "pl": "To jest zdanie.",
            "ca": "Això és una oració.",
            "nl": "Dit is een zin.",
            "ar": "هذه جملة.",
            "sv": "Det här är en mening.",
            "it": "Questa è una frase.",
            "id": "Ini adalah kalimat.",
            "hi": "यह एक वाक्य है।",
            "fi": "Tämä on lause.",
            "vi": "Đây là một câu.",
            "he": "זה משפט.",
            "uk": "Це речення.",
            "el": "Αυτή είναι μια πρόταση.",
            "ms": "Ini ialah ayat.",
            "cs": "Toto je věta.",
            "ro": "Aceasta este o propoziție.",
            "da": "Dette er en sætning.",
            "hu": "Ez egy mondat.",
            "ta": "இது ஒரு வாக்கியம்.",
            "no": "Dette er en setning.",
            "th": "นี่คือประโยค.",
            "ur": "یہ ایک جملہ ہے۔",
            "hr": "Ovo je rečenica.",
            "bg": "Това е изречение.",
            "lt": "Tai yra sakinys.",
            "la": "Haec sententia est.",
            "mi": "He rerenga kōrero tēnei.",
            "ml": "ഇത് ഒരു വാക്യം ആണ്.",
            "cy": "Dyma frawddeg.",
            "sk": "Toto je veta.",
            "te": "ఇది ఒక వాక్యం.",
            "fa": "این یک جمله است.",
            "lv": "Šis ir teikums.",
            "bn": "এটি একটি বাক্য।",
            "sr": "Ово је реченица.",
            "az": "Bu bir cümlədir.",
            "sl": "To je stavek.",
            "kn": "ಇದು ಒಂದು ವಾಕ್ಯ.",
            "et": "See on lause.",
            "mk": "Ова е реченица.",
            "br": "Hemañ zo ur frazenn.",
            "eu": "Hau esaldi bat da.",
            "is": "Þetta er setning.",
            "hy": "Սա նախադասություն է։",
            "ne": "यो एउटा वाक्य हो।",
            "mn": "Энэ бол өгүүлбэр юм.",
            "bs": "Ovo je rečenica.",
            "kk": "Бұл сөйлем.",
            "sq": "Kjo është një fjali.",
            "sw": "Hii ni sentensi.",
            "gl": "Esta é unha oración.",
            "mr": "हे एक वाक्य आहे.",
            "pa": "ਇਹ ਇੱਕ ਵਾਕ ਹੈ।",
            "si": "මේ වාක්‍යයක් වේ.",
            "km": "នេះជាប្រយោគមួយ។",
            "sn": "Uyu mutsara.",
            "yo": "Ọ̀rọ̀ yìí ni.",
            "so": "Tani waa jumlad.",
            "af": "Dit is 'n sin.",
            "oc": "Aquò es una frasa.",
            "ka": "ეს წინადადებაა.",
            "be": "Гэта сказ.",
            "tg": "Ин ҷумла аст.",
            "sd": "هيءَ هڪ جملو آهي.",
            "gu": "આ એક વાક્ય છે.",
            "am": "ይህ አንድ ዓረፍተ ነገር ነው።",
            "yi": "דאָס איז אַ זאַץ.",
            "lo": "ນີ້ແມ່ນປະໂຫຍກ.",
            "uz": "Bu bir gap.",
            "fo": "Hetta er ein setning.",
            "ht": "Sa a se yon fraz.",
            "ps": "دا یوه جمله ده.",
            "tk": "Bu bir sözlemdir.",
            "nn": "Dette er ei setning.",
            "mt": "Din hija sentenza.",
            "sa": "एषा एकं वाक्यम् अस्ति।",
            "lb": "Dëst ass e Saz.",
            "my": "ဒါက စာကြောင်းတစ်ခုပါ။",
            "bo": "འདི་ནི་ཚིག་གྲུབ་ཞིག་རེད།",
            "tl": "Ito ay isang pangungusap.",
            "mg": "Io dia fehezanteny.",
            "as": "এইটো এটা বাক্য।",
            "tt": "Бу җөмлә.",
            "haw": "He hopunaʻōlelo kēia.",
            "ln": "Oyo ezali lokasa.",
            "ha": "Wannan jumla ce.",
            "ba": "Был һөйләм.",
            "jw": "Iki ukara.",
            "su": "Ieu mangrupikeun kalimah."
        }
        self.whisper_temperatures = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.working_dir = os.getenv("TMPDIR", os.getenv("TEMP", os.getenv("TMP", ".")))

        # pylint: disable=import-outside-toplevel
        import torch
        if use_cpu or not torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Set CUDA_VISIBLE_DEVICES to -1 to use CPU
            LOG.info("Using CPU")
            self.device = "cpu"
        else:
            LOG.info("Using GPU")
            self.device = "cuda"

        if stream:
            self.configure_for_low_latency_streaming()

        self.isolate = isolate
        self.diarize = diarize
        if self.diarize == '':
            self.diarize = 0
        elif self.diarize is not None:
            self.diarize = int(self.diarize)

        self.configure_output_directory(outdir=outdir, workdir=workdir)
        if input_source is not None:
            self.configure_audio_source(input_source=input_source, start_time=start_time, stop_time=stop_time)

    def configure_output_directory(self, outdir:str, workdir:str):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        self.output_dir = outdir
        LOG.info(f"Output directory set to {self.output_dir}")

        # Set the working directory
        if workdir is None:
            self.working_dir = os.getenv("TMPDIR", os.getenv("TEMP", os.getenv("TMP", ".")))
        elif workdir == "":
            self.working_dir = self.output_dir
        else:
            if not os.path.isdir(workdir):
                os.makedirs(workdir)
            self.working_dir = workdir
        LOG.info(f"Working directory set to {self.working_dir}")


    def configure_audio_source(self, input_source:str, start_time:str, stop_time:str):
        self.input_source = input_source
        if os.path.exists(input_source) is False:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_source)

        input_name_no_ext = os.path.splitext(os.path.split(self.input_source)[-1])[0]
        self.output_prefix_no_ext = os.path.join(self.output_dir, input_name_no_ext)
        self.working_prefix_no_ext = os.path.join(self.working_dir, input_name_no_ext)

        self.diarization_file = self.diarization
        if self.diarization_file == "" or (self.diarize is not None and self.diarization_file is None):
            self.diarization_file = self.output_prefix_no_ext + ".rttm"

        self.start_time = timestr_to_sample(start_time) if start_time else None
        self.stop_time = timestr_to_sample(stop_time) if stop_time else None

        # pylint: disable=import-outside-toplevel
        from .audio.sources.micaudiosource import MicAudioSourcePyAudio as MicAudioSource
        from .audio.sources.fileaudiosource import FileAudioSource
        from .audio.sources.ffmpegfileaudiosource import PyAVAudioSource
        from .audio.sources.pcmaudiosource import PCMInputStreamAudioSource
        import sys
        import numpy as np

        if self.input_source == "-":
            self.source_stream = PCMInputStreamAudioSource(stream=sys.stdin, channels=1, sampling_rate=16000, dtype=np.int16)
            return

        elif self.input_source is None or self.input_source == ">":
            self.source_stream = MicAudioSource()
            return
        else:
            if os.path.splitext(self.input_source)[-1] == ".wav":
                file_audio_source = FileAudioSource(self.input_source, start_sample=self.start_time, end_sample=self.stop_time)
                if not self.stream:
                    if self.isolate is not None:
                        file_audio_source.isolate_voices(out_path_prefix=self.isolate or None)
                    if not self.diarize is None:
                        self.diarization = file_audio_source.compute_diarization(
                            rttm_file=self.diarization_file, device=self.device, nb_speakers=self.diarize)

                if self.diarization_file:
                    from .voices.diarization import Diarization
                    self.diarization = Diarization.load_diarization(rttm_file=self.diarization_file)
            else:
                if not self.stream and (self.isolate is not None or not self.diarize is None):
                    file_audio_source = PyAVAudioSource(file_path=self.input_source)
                    from .audio.sources.wavsink import WavSink
                    self.input_source = self.working_prefix_no_ext + '.wav'
                    WavSink.dump_to_wav(audio_source=file_audio_source, output_path=self.input_source)
                    self.configure_audio_source(input_source=self.input_source, start_time=start_time, stop_time=stop_time)
                    return
                else:
                    file_audio_source = PyAVAudioSource(
                        file_path=self.input_source, start_time=self.start_time/16000, end_time=self.stop_time/16000 if self.stop_time else None)
            self.source_stream = file_audio_source
            return

    def configure_for_low_latency_streaming(self):
        self.stream = True
        if self.stream:
            self.chunk_table = [
                (0, 0.025),
            ]
            self.whisper_best_of = 3
            self.whisper_beam_size = 3
            self.whisper_patience = 3.0
            self.whisper_temperatures = [0, 0.6]

    def configure_output_formats(self, output_formats:List[str]) -> "Config":
        self.enable_ass = "ass" in output_formats
        self.enable_docx = "docx" in output_formats
        self.enable_txt = "txt" in output_formats
        self.enable_md = "md" in output_formats
        self.enable_json = "json" in output_formats
        self.enable_stdout = "json" in output_formats
        self.enable_stdout = "stdout" in output_formats
        self.enable_stdout_nocolor = "stdout-nocolor" in output_formats
        return self

    def configure_languages(self, lang=List[str]) -> "Config":
        self.lang = lang
        return self
