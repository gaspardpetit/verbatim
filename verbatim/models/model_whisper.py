import whisper

class WhisperModel:
    """
    A singleton class for managing an instance of the OpenAI Whisper model.

    Attributes:
        model (whisper.Whisper): The OpenAI Whisper model instance.
    """
    _instance = None

    def __new__(cls):
        """
        Create a new instance of the class if it doesn't exist, otherwise, return the existing instance.

        Returns:
            WhisperModel: The WhisperModel instance.
        """
        if not cls._instance:
            cls._instance = super(WhisperModel, cls).__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        """
        Initialize the WhisperModel instance.
        """
        self.model = whisper.load_model("large")

    def unload(self):
        """
        Unload the model by setting the model attribute to None and clearing the instance.
        """
        self.model = None
        WhisperModel._instance = None
