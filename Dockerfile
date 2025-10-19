# Легкий образ с Python. Можно сменить на runpod/serverless:latest,
# но официальный способ — просто иметь handler.py и вызвать его как команду.
FROM python:3.10-slim

# Системные зависимости для чтения аудио (flac/ogg/mp3) и для soundfile/libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git ca-certificates \
    build-essential cmake ninja-build pkg-config \
    libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/

# Ускоряем сборку колёс, когда возможно
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY handler.py /app/

# Для локального теста "python handler.py" запустит встроенный dev-server SDK
CMD ["python", "handler.py"]
