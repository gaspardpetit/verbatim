FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y \
    software-properties-common -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    portaudio19-dev \
    ffmpeg

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN pip install cython wheel
RUN pip install --upgrade pip

ADD requirements-gpu.txt requirements-gpu.txt
ADD pyproject.toml pyproject.toml
ADD verbatim/__init__.py verbatim/__init__.py
ADD README.md README.md

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-gpu.txt
RUN pip install pyclean

ADD verbatim verbatim
RUN pyclean verbatim

RUN python -m pip install --upgrade build
RUN python -m build
RUN pip install .
RUN mkdir out

ARG HUGGINGFACE_TOKEN
ADD tests/data/init.mp3 init.mp3
RUN HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN verbatim init.mp3 --diarize --isolate -vv
