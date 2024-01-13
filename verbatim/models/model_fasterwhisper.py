"""
model_fasterwhisper.py
"""

from faster_whisper import WhisperModel

class FasterWhisperModel:
    """
    A singleton class for managing an instance of the FasterWhisper model.

    Attributes:
        model (WhisperModel): The FasterWhisper model instance.
    """
    _instance = None
    device = "cuda"

    def __new__(cls):
        """
        Create a new instance of the class if it doesn't exist, otherwise, return the existing 
        instance.

        Returns:
            FasterWhisperModel: The FasterWhisperModel instance.
        """
        if not cls._instance:
            cls._instance = super(FasterWhisperModel, cls).__new__(cls)
            cls._instance._init_once(FasterWhisperModel.device)
        return cls._instance

    # pylint: disable=attribute-defined-outside-init
    def _init_once(self, device:str = False):
        """
        Initialize the FasterWhisperModel instance.
        """
        if device == "cpu":
            self.model = WhisperModel("large-v3", device=device, compute_type="float32")
        else:
            self.model = WhisperModel("large-v3", device=device, compute_type="float16")

    def unload(self):
        """
        Unload the model by setting the model attribute to None and clearing the instance.
        """
        self.model = None
        FasterWhisperModel._instance = None
