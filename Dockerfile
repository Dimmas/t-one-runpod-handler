FROM runpod/serverless:gpu

# Системные либы для аудио + git
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
ENV PIP_NO_CACHE_DIR=1
# Устанавливаем torch с CUDA (подбери индекс под свою CUDA: cu118/cu121 и т.д.)
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch && \
    pip install -r requirements.txt

COPY handler_gpu.py /app/handler.py

# ❗ Переносим HF-кэш в сетевой том (тоже можно задать в Endpoint Env Vars)
ENV HF_HOME=/runpod-volume/hf_cache
ENV HF_HUB_CACHE=/runpod-volume/hf_cache/hub
ENV HF_XET_CACHE=/runpod-volume/hf_cache/xet
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache/transformers

RUN mkdir -p /runpod-volume/hf_cache/hub /runpod-volume/hf_cache/xet /runpod-volume/hf_cache/transformers || true

CMD ["python", "handler.py"]
