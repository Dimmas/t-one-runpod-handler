import os
import json
import tempfile
import base64
import logging
from typing import Optional, Dict

import requests
import runpod

# Переставим HF-кэш в сетевой том (на случай, если не задан в Endpoint ENV)
os.environ.setdefault("HF_HOME", "/runpod-volume/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/runpod-volume/hf_cache/hub")
os.environ.setdefault("HF_XET_CACHE", "/runpod-volume/hf_cache/xet")
os.environ.setdefault("TRANSFORMERS_CACHE", "/runpod-volume/hf_cache/transformers")
for p in (os.getenv("HF_HOME"), os.getenv("HF_HUB_CACHE"),
          os.getenv("HF_XET_CACHE"), os.getenv("TRANSFORMERS_CACHE")):
    if p:
        os.makedirs(p, exist_ok=True)

# --- T-one ---
from tone import StreamingCTCPipeline, read_audio  # https://github.com/voicekit-team/T-one
import torch

logger = logging.getLogger("tone_runpod_gpu")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

PIPELINE: Optional[StreamingCTCPipeline] = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_pipeline() -> StreamingCTCPipeline:
    global PIPELINE
    if PIPELINE is None:
        logger.info("Loading T-one pipeline on device=%s ...", DEVICE)
        PIPELINE = StreamingCTCPipeline.from_hugging_face()
        # Если у пайплайна есть внутренние torch-модули, они, как правило, сами переходят на CUDA.
        # На всякий случай можно попытаться выставить device, если у объекта есть метод .to/device:
        for attr in ("model", "acoustic_model"):
            if hasattr(PIPELINE, attr):
                mod = getattr(PIPELINE, attr)
                try:
                    mod.to(DEVICE)  # no-op на CPU
                except Exception:
                    pass
    return PIPELINE


def _load_audio_from_source(src: str):
    if not isinstance(src, str) or not src:
        raise ValueError("input.audio_file must be a non-empty string (URL or 'base64:<...>').")
    if src.startswith("base64:"):
        b64 = src[len("base64:") :]
        audio_bytes = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        return read_audio(temp_path)

    resp = requests.get(src, timeout=180)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(resp.content)
        temp_path = f.name
    return read_audio(temp_path)


def _validate(job_input: Dict):
    if not isinstance(job_input, dict):
        return {"error": "Input must be a JSON object under key 'input'."}
    if "audio_file" not in job_input:
        return {"error": "Missing required field 'audio_file' (URL or 'base64:<...>')."}
    return None


def handler(job):
    """
    Payload:
    {
      "input": {
        "audio_file": "https://.../file.wav"  // or "base64:<...>"
      }
    }
    Returns: { "text": "<full transcription>" }
    """
    job_input = job.get("input") or {}
    err = _validate(job_input)
    if err:
        return err

    runpod.serverless.progress_update(job, "init")
    pipeline = init_pipeline()

    runpod.serverless.progress_update(job, "loading_audio")
    try:
        audio = _load_audio_from_source(job_input["audio_file"])
    except Exception as e:
        logger.exception("Failed to load audio.")
        return {"error": f"Failed to load audio: {e}"}

    runpod.serverless.progress_update(job, "transcribing")
    try:
        text = pipeline.forward_offline(audio)  # полный ответ, без стрима
    except Exception as e:
        logger.exception("ASR failed.")
        return {"error": f"ASR failed: {e}"}

    return {"text": text}


runpod.serverless.start({"handler": handler})
