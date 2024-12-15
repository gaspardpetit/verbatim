from dataclasses import dataclass, field
from typing import Dict, List

from pyannote.core.annotation import Annotation

from .audio.sources.audiosource import AudioSource
from .transcript.format.writer import TranscriptWriterConfig


@dataclass
class Config:
    sampling_rate: int = 16000
    transcribe_latency: int = 16000
    frames_per_buffer: int = 1000
    window_duration: int = 30  # seconds
    whisper_prompts: Dict[str, str] = None
    lang: List[str] = field(default_factory=list)
    source: AudioSource = None
    diarization: Annotation = None
    whisper_model_size: str = "large-v3"
    device: str = "cuda"
    stream: bool = False
    diarize: bool = False
    whisper_beam_size: int = 9
    whisper_best_of: int = 9
    whisper_patience: float = 2.0
    whisper_temperatures: List[float] = None
    debug: bool = True
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

    def __init__(self):
        self.chunk_table = [
            (20, 5),
            (15, 4),
            (10, 3),
            (5, 2),
            (0, 1),
        ]
        self.lang = ["fr", "en"]
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
