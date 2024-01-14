FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y \
    software-properties-common -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    ffmpeg

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN pip install cython wheel
RUN pip install --upgrade pip

ADD requirements.txt requirements.txt
ADD setup.py setup.py
ADD verbatim verbatim
ADD README.md README.md
RUN pip install -r requirements.txt

RUN pip install pyclean
RUN pyclean verbatim

ADD tests/data/init.mp3 init.mp3

RUN python -m pip install --upgrade build
RUN python -m build
RUN pip install .
RUN mkdir out

ARG TOKEN_HUGGINGFACE
RUN TOKEN_HUGGINGFACE=$TOKEN_HUGGINGFACE verbatim init.mp3 -vv
