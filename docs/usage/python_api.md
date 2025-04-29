# Python API

Verbatim can be used as a Python library for integration into your own applications.

## Modularity

The project is organized to be modular, so individual components can be used outside the full pipeline,
and the pipeline can be customized with custom stages.

### Example with Custom Diarization

```python
from verbatim.audio.sources.sourceconfig import SourceConfig
from verbatim.audio.sources.factory import create_audio_source

# Create and configure the audio source
source = create_audio_source(
    input_source="audio/1ch_2spk_en-fr_AirFrance_00h03m54s.wav",
    device="cuda",
    source_config=SourceConfig(diarize=2)
)

from verbatim.config import Config
from verbatim.verbatim import Verbatim

# Create and configure Verbatim
verbatim = Verbatim(config=Config(lang=["en", "fr"], output_dir="out"))

# Process the audio
with source.open() as stream:
    for utterance, _unack_utterance, _unconfirmed_word in verbatim.transcribe(audio_stream=stream):
        print(utterance.text)
```

For a complete API reference, please see the [API Reference](../api/verbatim) section.
