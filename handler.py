import os
import tempfile
import base64
import logging
from typing import Optional, Dict

import runpod
import requests

# --- T-one ---
from tone import StreamingCTCPipeline, read_audio  # from voicekit-team/T-one

logger = logging.getLogger("tone_runpod")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

# ---- Global, single init (Best Practice per RunPod docs) ----
# Decoder: "greedy" по умолчанию (без KenLM). Можно переключить через ENV.
DECODER = os.getenv("TONE_DECODER", "greedy")  # greedy | beam
HF_REVISION = os.getenv("TONE_HF_REVISION")    # опционально, если захочешь фиксировать ревизию
PIPELINE: Optional[StreamingCTCPipeline] = None


def init_pipeline() -> StreamingCTCPipeline:
    global PIPELINE
    if PIPELINE is None:
        logger.info("Loading T-one pipeline (decoder=%s)...", DECODER)
        # В README используется краткая инициализация из HF
        # forward_offline дает полноценную транскрибацию за один вызов.
        if HF_REVISION:
            PIPELINE = StreamingCTCPipeline.from_hugging_face(
                decoder=DECODER, revision=HF_REVISION
            )
        else:
            PIPELINE = StreamingCTCPipeline.from_hugging_face(decoder=DECODER)
    return PIPELINE


def _load_audio_from_b64(b64: str):
    audio_bytes = base64.b64decode(b64)
    # read_audio в T-one принимает путь или файловый объект? В README показан путь.
    # Сохраним во временный файл, чтобы быть совместимыми со всеми форматами.
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    audio = read_audio(temp_path)
    return audio


def _load_audio_from_url(url: str, timeout: float = 120.0):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    # Аналогично — пишем во временный файл
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(resp.content)
        temp_path = f.name
    audio = read_audio(temp_path)
    return audio


def validate_input(job_input: Dict):
    if not isinstance(job_input, dict):
        return {"error": "Input must be a JSON object under key 'input'."}

    has_b64 = "audio_b64" in job_input
    has_url = "audio_url" in job_input
    if not (has_b64 or has_url):
        return {"error": "Provide either 'audio_b64' (base64) or 'audio_url'."}

    # Доп. опции
    decoder = job_input.get("decoder")
    if decoder and decoder not in {"greedy", "beam"}:
        return {"error": "Unsupported decoder. Use 'greedy' or 'beam'."}

    return None


def handler(job):
    """
    RunPod standard (non-streaming) handler.
    Input schema:
    {
      "input": {
        "audio_b64": "<base64-encoded audio>" | "audio_url": "https://...",
        "decoder": "greedy" | "beam",     # optional (overrides env)
        "return_phrases": false           # optional: include phrases from streaming splitter
      }
    }
    """
    job_input = job.get("input", {}) or {}

    # Validation
    err = validate_input(job_input)
    if err:
        return err

    # Optional progress updates (visible in job polling)
    runpod.serverless.progress_update(job, "init")

    # Init model (global)
    pipeline = init_pipeline()

    # Decoder override per-request (если надо)
    req_decoder = job_input.get("decoder")
    if req_decoder and req_decoder != pipeline.decoder_type:
        # Простой путь: переинициализация пайплайна под другой декодер
        # (чтобы не держать сразу две копии в памяти)
        # В высоконагруженных сценариях лучше выделить отдельный воркер/эндпойнт.
        logger.info("Reinitializing pipeline with decoder=%s", req_decoder)
        # NOTE: from_hugging_face кэширует веса; будет быстро после первой загрузки
        globals()["PIPELINE"] = StreamingCTCPipeline.from_hugging_face(decoder=req_decoder)
        pipeline = PIPELINE

    runpod.serverless.progress_update(job, "loading_audio")

    # Load audio
    try:
        if "audio_b64" in job_input:
            audio = _load_audio_from_b64(job_input["audio_b64"])
        else:
            audio = _load_audio_from_url(job_input["audio_url"])
    except Exception as e:
        logger.exception("Failed to load audio.")
        return {"error": f"Failed to load audio: {e}"}

    runpod.serverless.progress_update(job, "transcribing")

    # Offline full transcription (one-shot, non-streaming)
    try:
        text = pipeline.forward_offline(audio)
    except Exception as e:
        logger.exception("ASR failed.")
        return {"error": f"ASR failed: {e}"}

    result = {"text": text}

    # По желанию — еще и фразы (если хочется разбивки на «фразы» тем же сплиттером)
    if job_input.get("return_phrases"):
        try:
            phrases = pipeline.forward_offline(audio, return_phrases=True)
            # Если библиотека не поддерживает параметр, можно fallback-нуть к пустому.
            if phrases is not None:
                result["phrases"] = phrases
        except TypeError:
            # Версии без return_phrases: пропускаем.
            pass

    # Можно попросить освежить воркер (очистка состояния/логов), см. docs
    # return {"refresh_worker": True, "job_results": result}

    return result


# --- Required by RunPod ---
runpod.serverless.start({"handler": handler})
