from .speech_transcription import TranscribeSpeech, TranscribeSpeechFasterWhisper
from .transcript_writing import WriteTranscript, WriteTranscriptDocx, WriteTranscriptAss, WriteTranscriptStdout, WriteTranscriptMulti

class Engine:
    def __init__(self,
                 speech_transcription: TranscribeSpeech = None,
                 transcript_writing: WriteTranscript = None,
                 ) -> None:
        self.speech_transcription = speech_transcription or TranscribeSpeechFasterWhisper()
        self.transcript_writing = transcript_writing or WriteTranscriptMulti([ WriteTranscriptDocx(), WriteTranscriptAss(), WriteTranscriptStdout() ])
