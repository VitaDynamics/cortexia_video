FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml setup.py uv.lock ./
COPY cortexia_video ./cortexia_video

RUN pip3 install -e .

COPY config ./config

CMD ["cortexia-video"]
