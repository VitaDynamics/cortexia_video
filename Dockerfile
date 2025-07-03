FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# install Python, pip, and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml setup.py uv.lock ./
COPY cortexia_video ./cortexia_video

RUN uv pip install --no-cache-dir -e .

COPY config ./config

CMD ["cortexia-video"]
