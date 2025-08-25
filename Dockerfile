FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# install Python, pip, and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml setup.py uv.lock ./
COPY cortexia ./cortexia

RUN cd ./cortexia 

RUN uv venv --system --python 3.11

RUN uv pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --no-cache-dir -e .

COPY config ./config

CMD ["cortexia-video"]
