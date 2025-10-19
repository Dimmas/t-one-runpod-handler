import os
import json
import tempfile
import base64
import logging
from typing import Optional, Dict

import requests
import runpod

# T-one
from tone import StreamingCTCPipeline, read_audio  # https://github.com/voicekit-team/T-one

logger = logging.getLogger("tone_runpod")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

PIPELINE: Optional[StreamingCTCPipeline] = None


def init_pipeline() -> StreamingCTCPipeline:
    """Lazy, single init per worker. No decoder kwargs here."""
    global PIPELINE
    if PIPELINE is None:
        logger.info("Loading T-one pipeline...")
        PIPELINE = StreamingCTCPipeline.from_hugging_face()
    return PIPELINE


def _load_audio_from_source(src: str):
    """
    Accepts either:
      - URL (http/https)
      - base64 string prefixed with 'base64:' (optional convenience)

    Returns audio array via T-one's read_audio.
    """
    if not isinstance(src, str) or not src:
        raise ValueError("input.audio_file must be a non-empty string (URL or 'base64:<...>').")

    # base64 shortcut, if ever needed
    if src.startswith("base64:"):
        b64 = src[len("base64:") :]
        audio_bytes = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        return read_audio(temp_path)

    # treat as URL
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
    Standard (non-streaming) RunPod handler.
    Expects payload:
    {
      "input": {
        "audio_file": "https://.../file.wav"  // or "base64:<...>"
      }
    }
    Returns:
    { "text": "<full transcription>" }
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
        # Full, one-shot recognition (no streaming)
        text = pipeline.forward_offline(audio)
    except Exception as e:
        logger.exception("ASR failed.")
        return {"error": f"ASR failed: {e}"}

    return {"text": text}


# Required by RunPod
runpod.serverless.start({"handler": handler})