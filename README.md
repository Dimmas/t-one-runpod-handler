# T-one RunPod Serverless Handler (non-stream, full result)

Serverless-воркер для распознавания русской речи на базе [voicekit-team/T-one] с полным (нестриминговым) ответом.

## Как работает

- Хэндлер типа **Standard** (синхронный) — возвращает готовый текст одной строкой.
- Внутри используется `StreamingCTCPipeline.forward_offline(audio)` из T-one.
- Вход: `audio_b64` **или** `audio_url`. Опционально `decoder: greedy|beam`.

## Сборка

```bash
docker build -t borzovich/t-one-runpod-handler:v1 .