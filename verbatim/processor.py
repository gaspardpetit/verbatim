from abc import ABC, abstractmethod
from .context import Context

class Processor:
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def execute(self) -> None:
        ...

import os
from typing import List
from .context import Context
from .engine import Engine
from .filter import Filter

class ProcessorTranscribePipeline:
    def __init__(self,
                 context: Context,
                 engine: Engine,
                 ):
        self.context: Context = context
        self.engine: Engine = engine

    def execute(self):
        os.makedirs(self.context.work_directory_path, exist_ok=True)

        self.context.audio_file_path = self.context.source_file_path
        self.context.voice_file_path = self.context.audio_file_path
        self.context.language_file = None
        self.context.diarization_file = None
        filters: List[Filter] = [
            self.engine.speech_transcription,
            self.engine.transcript_writing,
        ]

        for f in filters:
            f.load(**self.context.to_dict())
            f.execute(**self.context.to_dict())
            f.unload(**self.context.to_dict())

class ProcessorTranscribe(Processor):
    def __init__(self, context:Context, engine:Engine) -> None:
        super().__init__()
        self.context = context
        self.engine =engine

    def execute(self) -> None:
        pipeline: ProcessorTranscribePipeline = ProcessorTranscribePipeline(context=self.context, engine=self.engine)
        pipeline.execute()
