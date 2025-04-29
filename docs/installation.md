# Installation

## Prerequisites

- [Python](https://www.python.org/) version 3.9 to 3.11
- Astral's [`uv`](https://github.com/astral-sh/uv) for development

### Portaudio

Portaudio is used on macOS and Linux for accessing the microphone when doing live transcription. To install:

````{tab-set}
```{tab-item} Linux

On Linux, you can install Portaudio using your package manager, for example:

`sudo apt install portaudio19-dev`
```

```{tab-item} macOS
On macOS, you can install Portaudio using Homebrew:

`brew install portaudio`
```
````

## Installing `verbatim`

### From PyPI

```bash
pip install verbatim
```

### From Git (Latest Version)

```bash
pip install git+https://github.com/gaspardpetit/verbatim.git
```

### Torch with CUDA Support

If the tool falls back to CPU instead of GPU, you may need to reinstall the torch dependency with CUDA support.
Refer to the instructions at [PyTorch's installation page](https://pytorch.org/get-started/locally/).

## Hugging Face Token

For diarization, this project requires access to the pyannote models which are gated:

1. Create an account on [Hugging Face](https://huggingface.co/)
2. Request access to the model at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Request access to the model at https://huggingface.co/pyannote/segmentation-3.0
4. From your **Settings** > **Access Tokens**, generate an access token
5. When running verbatim for the first time, set the `HUGGINGFACE_TOKEN` environment variable to your Hugging Face token.
   Once the model is downloaded, this is no longer necessary.

Instead of setting the `HUGGINGFACE_TOKEN` environment variable, you may prefer to set the value
using a `.env` file in the current directory:

```bash
HUGGINGFACE_TOKEN=hf_******
```
