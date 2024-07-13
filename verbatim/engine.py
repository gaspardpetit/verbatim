from .speech_transcription import TranscribeSpeech, TranscribeSpeechFasterWhisper
from .transcript_writing import WriteTranscript, WriteTranscriptDocx, WriteTranscriptAss, WriteTranscriptStdout, WriteTranscriptMulti
from .wav_conversion import ConvertToWav, ConvertToWavSoundfile

class Engine:
    def __init__(self,
                 wav_converter:ConvertToWav = None,
                 speech_transcription: TranscribeSpeech = None,
                 transcript_writing: WriteTranscript = None,
                 ) -> None:
        self.wav_converter = wav_converter or ConvertToWavSoundfile()
        self.speech_transcription = speech_transcription or TranscribeSpeechFasterWhisper()
        self.transcript_writing = transcript_writing or WriteTranscriptMulti([ WriteTranscriptDocx(), WriteTranscriptAss(), WriteTranscriptStdout() ])
