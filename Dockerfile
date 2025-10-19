# Легкий образ с Python. Можно сменить на runpod/serverless:latest,
# но официальный способ — просто иметь handler.py и вызвать его как команду.
FROM python:3.10-slim

# Системные зависимости для чтения аудио (flac/ogg/mp3) и для soundfile/libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/

# Ускоряем сборку колёс, когда возможно
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY handler.py /app/

# Параметры по умолчанию: не требуем KenLM; greedy-декодер
ENV TONE_DECODER=greedy
# ENV TONE_HF_REVISION=main   # опционально зафиксировать ревизию модели

# Для локального теста "python handler.py" запустит встроенный dev-server SDK
CMD ["python", "handler.py"]
