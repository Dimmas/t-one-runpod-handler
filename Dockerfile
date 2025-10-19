# ВАРИАНТ 1 (рекомендуемый для прод-стабильности)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# ВАРИАНТ 2 (если нужна более новая CUDA/cuDNN)
# FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Системные зависимости для аудио и https
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git ca-certificates \
    build-essential cmake pkg-config \
    libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости питона (torch уже внутри образа)
COPY requirements.txt /app/
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip && pip install -r requirements.txt

# Хэндлер
COPY handler.py /app/


# (опц.) покажем в логах доступность CUDA
ENV PYTHONUNBUFFERED=1
CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count());" && python /app/handler.py
