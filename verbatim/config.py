import logging
import os
import platform
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import List, Mapping, Optional, Tuple

from verbatim.cache import ArtifactCache
from verbatim.logging_utils import get_status_logger
from verbatim_audio.sources.audiosource import AudioSource

LOG = logging.getLogger(__name__)
STATUS_LOG = get_status_logger()

DEFAULT_MULTILANG_PROMPTS: Mapping[str, str] = MappingProxyType(
    {
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
        "su": "Ieu mangrupikeun kalimah.",
    }
)

DEFAULT_HIGHLATENCY_CHUNKTABLE: List[Tuple[float, float]] = [
    (0.75, 0.20),
    (0.50, 0.15),
    (0.25, 0.10),
    (0.10, 0.05),
    (0.00, 0.025),
]

DEFAULT_HIGHLATENCY_WHISPERTEMPERATURES: List[float] = [
    0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

DEFAULT_LOWLATENCY_WHISPERTEMPERATURES: List[float] = [0, 0.6]
DEFAULT_LOWLATENCY_CHUNKTABLE: List[Tuple[float, float]] = [(0, 0.025)]
DEFAULT_LANGUAGES = ["en"]


def get_default_working_directory() -> Optional[str]:
    return os.getenv("TMPDIR", os.getenv("TEMP", os.getenv("TMP")))


@dataclass
class Config:
    sampling_rate: int = 16000
    transcribe_latency: int = 16000
    frames_per_buffer: int = 1000
    window_duration: int = 30  # seconds
    device: str = "auto"
    stream: bool = False
    debug: bool = False
    log_file: Optional[str] = None
    log_colours: bool = True
    # Caching and connectivity
    model_cache_dir: Optional[str] = None
    offline: bool = False

    # TRANSCRIPTION
    lang: List[str] = field(default_factory=lambda: DEFAULT_LANGUAGES)
    code_switching: bool = True
    transcriber_backend: str = "auto"
    language_identifier_backend: str = "transcriber"
    language_detection_initial_seconds: float = 2.0
    language_detection_increment_seconds: float = 0.0
    language_detection_factor: float = 2.0
    mms_lid_model_size: str = "facebook/mms-lid-126"
    non_speech_backend: str = "energy"
    ast_audio_model_size: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    whisper_prompts: Mapping[str, str] = field(default_factory=lambda: DEFAULT_MULTILANG_PROMPTS)
    chunk_table: List[Tuple[float, float]] = field(default_factory=list)

    whisper_beam_size: int = -1
    whisper_best_of: int = -1
    whisper_patience: float = -1
    whisper_temperatures: List[float] = field(default_factory=list)
    # Whisper model size/path used by transcribers; default matches production
    whisper_model_size: str = "large-v3"
    voxtral_model_size: str = "mistralai/Voxtral-Mini-3B-2507"
    voxtral_dtype: str = "auto"
    voxtral_max_new_tokens: int = 256
    qwen_asr_model_size: str = "Qwen/Qwen3-ASR-1.7B"
    qwen_aligner_model_size: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    qwen_dtype: str = "auto"
    qwen_max_inference_batch_size: int = 1
    qwen_max_new_tokens: int = 256

    # OUTPUT
    working_dir: Optional[str] = field(default_factory=get_default_working_directory)

    output_dir: str = "."
    cache: ArtifactCache = field(init=False)

    # INPUT
    source_stream: Optional[AudioSource] = None

    def __post_init__(self):
        self.configure_device(device=self.device)

        self.configure_latency(stream=self.stream)

        self.configure_output_directory(output_dir=self.output_dir, working_dir=self.working_dir)
        self.configure_artifact_cache()

        # Configure model cache and offline mode last so env vars are ready
        self.configure_cache(model_cache_dir=self.model_cache_dir, offline=self.offline)

    def configure_languages(self, lang: List[str]) -> "Config":
        self.lang = lang
        return self

    @staticmethod
    def detect_device() -> str:
        # pylint: disable=import-outside-toplevel
        import torch

        if platform.processor() == "arm" and platform.system() == "Darwin" and torch.backends.mps.is_available():
            # Check for Apple Silicon and MPS availability
            return "mps"

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    @staticmethod
    def is_device_supported(device: str) -> bool:
        # pylint: disable=import-outside-toplevel
        import torch

        if device == "cpu":
            return True
        if device == "cuda":
            return bool(torch.cuda.is_available())
        if device == "mps":
            return bool(platform.processor() == "arm" and platform.system() == "Darwin" and torch.backends.mps.is_available())
        return False

    def configure_device(self, device: str) -> "Config":
        if device == "auto":
            self.device = Config.detect_device()
        else:
            if device not in ("cpu", "cuda", "mps"):
                raise RuntimeError(f"Unsupported device '{device}'. Expected one of: auto, cpu, cuda, mps.")
            if not Config.is_device_supported(device):
                raise RuntimeError(f"Requested device '{device}' is not available on this system.")
            self.device = device

        if self.device != "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Set CUDA_VISIBLE_DEVICES to -1 to force CPU
        elif os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
            del os.environ["CUDA_VISIBLE_DEVICES"]

        if self.device == "cuda":
            STATUS_LOG.info("Using GPU (CUDA)")
        elif self.device == "mps":
            STATUS_LOG.info("Using MPS (Apple Silicon)")
        else:
            if device == "auto":
                STATUS_LOG.info("No hardware acceleration detected, defaulting to CPU hardware")
            else:
                STATUS_LOG.info("Using CPU")

        return self

    def configure_latency(self, stream: bool) -> "Config":
        self.stream = stream
        if len(self.chunk_table) == 0:
            self.chunk_table = DEFAULT_LOWLATENCY_CHUNKTABLE if stream else DEFAULT_HIGHLATENCY_CHUNKTABLE
        if len(self.whisper_temperatures) == 0:
            self.whisper_temperatures = DEFAULT_LOWLATENCY_WHISPERTEMPERATURES if stream else DEFAULT_HIGHLATENCY_WHISPERTEMPERATURES
        if self.whisper_best_of < 0:
            self.whisper_best_of = 3 if stream else 9
        if self.whisper_beam_size < 0:
            self.whisper_beam_size = 3 if stream else 9
        if self.whisper_patience < 0:
            self.whisper_patience = 1.0 if stream else 2.0
        return self

    def configure_output_directory(self, output_dir: str, working_dir: Optional[str] = ""):
        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        STATUS_LOG.info("Output directory set to %s", self.output_dir)

        # Set the working directory
        self.working_dir = working_dir
        if self.working_dir is None:
            STATUS_LOG.info("Working directory disabled; intermediate artifacts will use in-memory cache.")
            return

        if not os.path.isdir(self.working_dir):
            os.makedirs(self.working_dir)
        STATUS_LOG.info("Working directory set to %s", self.working_dir)

    def configure_artifact_cache(self) -> None:
        # pylint: disable=import-outside-toplevel
        if getattr(self, "cache", None) is None:
            if self.working_dir is None:
                from verbatim.cache import InMemoryArtifactCache

                self.cache = InMemoryArtifactCache()
            else:
                from verbatim.cache import FileBackedArtifactCache

                self.cache = FileBackedArtifactCache(base_dir=self.working_dir)

    def configure_cache(self, model_cache_dir: Optional[str], offline: bool) -> "Config":
        """Configure deterministic cache directories and offline mode.

        - Sets environment variables that major libs honor:
          HF_HOME/HUGGINGFACE_HUB_CACHE for Hugging Face, and a project-specific
          VERBATIM_MODELDIR and VERBATIM_OFFLINE.
        - Creates directories if they do not exist.
        """
        # Offline toggle
        if offline:
            os.environ["VERBATIM_OFFLINE"] = "1"
            # Hugging Face/Transformers offline toggles
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            # Do not forcibly unset, but ensure default is empty if not set
            os.environ.setdefault("VERBATIM_OFFLINE", "0")

        # Default to a local project cache if unspecified
        if not model_cache_dir:
            model_cache_dir = os.path.join(os.getcwd(), ".verbatim")

        if model_cache_dir:
            created_ok = True
            try:
                os.makedirs(model_cache_dir, exist_ok=True)
            except OSError:
                # If we cannot create the cache directory, log and continue with defaults
                LOG.warning(f"Could not create model cache dir: {model_cache_dir}")
                created_ok = False

            # If exists but not writable, also degrade gracefully
            if created_ok and (not os.access(model_cache_dir, os.W_OK)):
                LOG.warning(f"Model cache dir not writable: {model_cache_dir}")
                created_ok = False

            if not created_ok:
                # Do not set subdirs or cache env; keep defaults
                return self

            # Root is usable; set env and create subdirs guardedly
            os.environ["VERBATIM_MODELDIR"] = model_cache_dir

            # XDG cache root influences many libs (incl. whisper default cache path)
            try:
                xdg_cache = os.path.join(model_cache_dir, "xdg")
                os.makedirs(xdg_cache, exist_ok=True)
                os.environ.setdefault("XDG_CACHE_HOME", xdg_cache)
            except OSError:
                LOG.warning(f"Could not prepare XDG cache under {model_cache_dir}")

            # Hugging Face cache
            try:
                hf_home = os.path.join(model_cache_dir, "hf")
                os.makedirs(hf_home, exist_ok=True)
                os.environ.setdefault("HF_HOME", hf_home)
                os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
            except OSError:
                LOG.warning(f"Could not prepare Hugging Face cache under {model_cache_dir}")

            try:
                pyannote_cache = os.path.join(model_cache_dir, "pyannote")
                os.makedirs(pyannote_cache, exist_ok=True)
                os.environ["PYANNOTE_CACHE"] = pyannote_cache
            except OSError:
                LOG.warning(f"Could not prepare pyannote cache under {model_cache_dir}")

        return self

    def read_to_cache(
        self,
        *,
        input_path: Optional[str],
        vttm_path: Optional[str] = None,
        rttm_path: Optional[str] = None,
    ) -> None:
        """Populate the artifact cache from disk paths provided by the CLI."""
        if self.cache is None:
            return

        def _read_bytes(path: str) -> None:
            if not os.path.isfile(path):
                LOG.warning("Cache preload skipped; file not found: %s", path)
                return
            try:
                with open(path, "rb") as fh:
                    self.cache.set_bytes(path, fh.read())
                STATUS_LOG.info("Cache preloaded bytes: %s", path)
            except OSError as exc:
                LOG.warning("Cache preload failed for %s: %s", path, exc)

        def _read_text(path: str) -> None:
            if not os.path.isfile(path):
                LOG.warning("Cache preload skipped; file not found: %s", path)
                return
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self.cache.set_text(path, fh.read())
                STATUS_LOG.info("Cache preloaded text: %s", path)
            except OSError as exc:
                LOG.warning("Cache preload failed for %s: %s", path, exc)

        if input_path not in (None, "", "-", ">"):
            _read_bytes(input_path)

        if vttm_path not in (None, ""):
            _read_text(vttm_path)

        if rttm_path not in (None, ""):
            _read_text(rttm_path)
