"""
setuptools module for uniscripts.
"""
from pathlib import Path
import re
from setuptools import setup, find_packages

PACKAGE_NAME = "verbatim"
INIT_FILE = Path(f"{PACKAGE_NAME}/__init__.py").absolute()
REQUIREMENTS = """
    wheel
    onnxruntime-gpu==1.16.0
    onnxruntime
    numpy

    pyannote.core
    pyannote.audio

    torch
    torchaudio

    faster_whisper
    openai-whisper

    soundfile
    demucs

    dataclasses_json

    python-ffmpeg
    pydub

    audio_separator

    python-docx
    langcodes
""".split()


# Load version from __init__.py
print(INIT_FILE)
with open(INIT_FILE, encoding="utf-8") as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        f.read(), re.MULTILINE).group(1)

# Load load description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()



setup(
    name=PACKAGE_NAME,
    version=version,
    description='high quality multi-lingual speech to text',
    entry_points={
            'console_scripts': [
                'verbatim = verbatim.__main__:main',
            ],
        },
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=REQUIREMENTS,

    url='https://github.com/gaspardpetit/verbatim',

    # Author details
    author='Gaspard Petit',
    author_email='gaspardpetit@gmail.com',

    # Choose your license
    license='MIT',

    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        "Intended Audience :: Developers",
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',

        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
        "Operating System :: OS Independent",

        'Programming Language :: Python :: 3',
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",

    ],

    keywords=[
        'speech-to-text',
        'audio processing',
        'multilingual',
        'natural language processing',
        'automatic speech recognition',
        'ASR',
        'text transcription',
        'language support',
        'machine learning',
        'deep learning',
        'NLP',
        'linguistics',
        'voice recognition',
        'PyTorch',
        'TensorFlow',
        'audio analysis',
        'speech analytics',
        'spoken language processing',
        'i18n',
        'internationalization',
    ],

)
