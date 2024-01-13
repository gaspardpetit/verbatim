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
from .transcript_writing import WriteTranscript, WriteTranscriptDocx, WriteTranscriptAss
from .transcription import Transcription

class Pipeline:
    #pylint: disable=dangerous-default-value
    def __init__(self,
                 context: Context,
                 convert_to_wav: ConvertToWav = ConvertToWavFFMpeg(),
                 isolate_voices: IsolateVoices = IsolateVoicesMDX(),
                 diarize_speakers: DiarizeSpeakers = DiarizeSpeakersPyannote(),
                 detect_languages: DetectLanguage = DetectLanguageFasterWhisper(),
                 speech_transcription: TranscribeSpeech = TranscribeSpeechWhisper(),
                 transcripte_writing: [WriteTranscript] = [ WriteTranscriptDocx(), WriteTranscriptAss() ],
                 ):
        self.context: Context = context
        self.convert_to_wav: ConvertToWav = convert_to_wav
        self.isolate_voices: IsolateVoices = isolate_voices
        self.diarize_speakers: DiarizeSpeakers = diarize_speakers
        self.detect_languages: DetectLanguage = detect_languages
        self.transcript_speech: TranscribeSpeech = speech_transcription
        self.transcripte_writing: [WriteTranscript] = transcripte_writing

    def execute(self):

        self.convert_to_wav.execute(
            input_file=self.context.source_file_path,
            output_file=self.context.audio_file_path,
            kwargs=self.context.to_dict())

        waveform: ndarray = self.isolate_voices.execute(
            source_path=self.context.audio_file_path,
            destination_path=self.context.voice_file_path,
            kwargs=self.context.to_dict())

        diarization: Annotation = self.diarize_speakers.execute(
            audio_file=self.context.voice_file_path,
            rttm_file=self.context.diarization_file,
            min_speakers=self.context.nb_speakers,
            max_speakers=self.context.nb_speakers,
            kwargs=self.context.to_dict())

        detected_languages:Transcription = self.detect_languages.execute(
            diarization=diarization,
            speech_segment_float32_16khz=waveform,
            language_file=self.context.language_file,
            languages=self.context.languages,
            kwargs=self.context.to_dict())

        transcript = self.transcript_speech.execute(
            detected_languages=detected_languages,
            speech_segment_float32_16khz=waveform,
            languages=self.context.languages,
            transcription_file=self.context.transcription_path,
            diarization=diarization,
            kwargs=self.context.to_dict())

        for writer in self.transcripte_writing:
            writer.execute(transcript=transcript, output_file=self.context.output_file, kwargs=self.context.to_dict())
