"""
pipeline
"""
#pylint: disable=unused-import
import os
import logging
from typing import List

from .context import Context
from .wav_conversion import ConvertToWav, ConvertToWavFFMpeg, ConvertToWavSoundfile
from .voice_isolation import IsolateVoices, IsolateVoicesNone, IsolateVoicesFile, IsolateVoicesDemucs, IsolateVoicesMDX
from .speaker_diarization import DiarizeSpeakers, DiarizeSpeakersNone, DiarizeSpeakersPyannote
from .speaker_diarization import DiarizeSpeakersFile, DiarizeSpeakersSpeechBrain
from .language_detection import DetectLanguage, DetectLanguageFasterWhisper, DetectLanguageWhisper, DetectLanguageFile
from .speech_transcription import TranscribeSpeechWhisper, TranscribeSpeechFile
from .speech_transcription import TranscribeSpeech, TranscribeSpeechFasterWhisper
from .transcript_writing import WriteTranscript, WriteTranscriptDocx, WriteTranscriptAss, WriteTranscriptStdout, WriteTranscriptMulti
from .transcription import Transcription
from .filter import Filter

LOG = logging.getLogger(__name__)

class Pipeline:
    def __init__(self,
                 context: Context,
                 convert_to_wav: ConvertToWav = None,
                 isolate_voices: IsolateVoices = None,
                 diarize_speakers: DiarizeSpeakers = None,
                 detect_languages: DetectLanguage = None,
                 speech_transcription: TranscribeSpeech = None,
                 transcript_writing: WriteTranscript = None,
                 ):

        if convert_to_wav is None:
            if context.has_ffmpeg:
                convert_to_wav = ConvertToWavFFMpeg()
            else:
                LOG.warning("ffmpeg was not detected, will only handle .wav audio files")
                convert_to_wav = ConvertToWavSoundfile()

        if isolate_voices is None:
            isolate_voices = IsolateVoicesMDX()
        if diarize_speakers is None:
            diarize_speakers = DiarizeSpeakersPyannote()
        if detect_languages is None:
            detect_languages = DetectLanguageFasterWhisper()
        if speech_transcription is None:
            speech_transcription = TranscribeSpeechFasterWhisper()
        if transcript_writing is None:
            transcript_writing = WriteTranscriptMulti([ WriteTranscriptDocx(), WriteTranscriptAss(), WriteTranscriptStdout() ])

        self.context: Context = context
        self.convert_to_wav: ConvertToWav = convert_to_wav
        self.isolate_voices: IsolateVoices = isolate_voices
        self.diarize_speakers: DiarizeSpeakers = diarize_speakers
        self.detect_languages: DetectLanguage = detect_languages
        self.speech_transcription: TranscribeSpeech = speech_transcription
        self.transcript_writing: WriteTranscript = transcript_writing

    def execute(self):
        os.makedirs(self.context.work_directory_path, exist_ok=True)

        if self.context.transcribe_only:
            self.context.audio_file_path = self.context.source_file_path
            self.context.voice_file_path = self.context.audio_file_path
            self.context.language_file = None
            self.context.diarization_file = None
            filters: List[Filter] = [
                self.speech_transcription,
                self.transcript_writing,
            ]
        else:
            filters: List[Filter] = [
                self.convert_to_wav,
                self.isolate_voices,
                self.diarize_speakers,
                self.detect_languages,
                self.speech_transcription,
                self.transcript_writing,
            ]

        for f in filters:
            f.load(**self.context.to_dict())
            f.execute(**self.context.to_dict())
            f.unload(**self.context.to_dict())
