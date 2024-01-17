"""
pipeline
"""
#pylint: disable=unused-import
from numpy import ndarray
from pyannote.core import Annotation

from .context import Context
from .wav_conversion import ConvertToWav, ConvertToWavFFMpeg
from .voice_isolation import IsolateVoices, IsolateVoicesNone, IsolateVoicesFile, IsolateVoicesDemucs, IsolateVoicesMDX
from .speaker_diarization import DiarizeSpeakers, DiarizeSpeakersNone, DiarizeSpeakersPyannote
from .speaker_diarization import DiarizeSpeakersFile, DiarizeSpeakersSpeechBrain
from .language_detection import DetectLanguage, DetectLanguageFasterWhisper, DetectLanguageWhisper, DetectLanguageFile
from .speech_transcription import TranscribeSpeechWhisper, TranscribeSpeechFile
from .speech_transcription import TranscribeSpeech, TranscribeSpeechFasterWhisper
from .transcript_writing import WriteTranscript, WriteTranscriptDocx, WriteTranscriptAss, WriteTranscriptStdout
from .transcription import Transcription

class Pipeline:
    #pylint: disable=dangerous-default-value
    def __init__(self,
                 context: Context,
                 convert_to_wav: ConvertToWav = ConvertToWavFFMpeg(),
                 isolate_voices: IsolateVoices = IsolateVoicesMDX(),
                 diarize_speakers: DiarizeSpeakers = DiarizeSpeakersPyannote(),
                 detect_languages: DetectLanguage = DetectLanguageFasterWhisper(),
                 speech_transcription: TranscribeSpeech = TranscribeSpeechFasterWhisper(),
                 transcripte_writing: [WriteTranscript] = [
                     WriteTranscriptDocx(), WriteTranscriptAss(), WriteTranscriptStdout()
                     ],
                 ):
        self.context: Context = context
        self.convert_to_wav: ConvertToWav = convert_to_wav
        self.isolate_voices: IsolateVoices = isolate_voices
        self.diarize_speakers: DiarizeSpeakers = diarize_speakers
        self.detect_languages: DetectLanguage = detect_languages
        self.transcript_speech: TranscribeSpeech = speech_transcription
        self.transcripte_writing: [WriteTranscript] = transcripte_writing

    def execute(self):

        filters: [] = [
            self.convert_to_wav,
            self.isolate_voices,
            self.diarize_speakers,
            self.detect_languages,
            self.transcript_speech
        ] + self.transcripte_writing

        for f in filters:
            f.load(**self.context.to_dict())
            f.execute(**self.context.to_dict())
            f.unload(**self.context.to_dict())
