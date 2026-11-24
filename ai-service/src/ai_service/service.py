import asyncio
import logging

from common.broker import Publisher, RabbitConnection
from common.lifecycle import run_consumer
from common.messages import AlertMessage, AnalyzeMessage, Face
from common.storage import S3Storage

from ai_service.analyzer import analyze_emotions, expected_crop_mode, _recognizer
from ai_service.config import settings

logger = logging.getLogger(__name__)
# Пары (режим кропа cv-service, режим модели), о несовпадении которых уже предупредили.
_warned_modes: set[tuple[str | None, str]] = set()


class AnalyzeHandler:
    """analyze_msg {frame, faces} -> GET кропов (или кадра) из S3 -> анализ эмоций -> create_alert."""

    _storage: S3Storage
    _publisher: Publisher

    def __init__(self, storage: S3Storage, publisher: Publisher) -> None:
        self._storage = storage
        self._publisher = publisher

    async def _load_crop(self, face: Face, mode: str) -> bytes | None:
        if face.crop_s3_key is None:
            return None
        if face.crop_mode != mode:
            if (face.crop_mode, mode) not in _warned_modes:
                _warned_modes.add((face.crop_mode, mode))
                logger.warning(
                    "cv-service crops are %r, model expects %r: cropping from frames instead; set CV_FACE_CROP to the model's mode",
                    face.crop_mode, mode,
                )
            return None
        return await self._storage.get_async(face.crop_s3_key)

    async def __call__(self, message: AnalyzeMessage) -> None:
        frame = message.frame
        mode = expected_crop_mode()
        crops = await asyncio.gather(*(self._load_crop(face, mode) for face in message.faces))
        # Целый кадр нужен, только если хотя бы для одного лица нет подходящего кропа.
        jpeg = None
        if any(crop is None for crop in crops):
            jpeg = await self._storage.get_async(frame.s3_key)
        alert_text = await asyncio.to_thread(analyze_emotions, jpeg, message.faces, crops)
        if alert_text is None:
            return

        await self._publisher.publish(AlertMessage(frame=frame, text=alert_text))
        logger.info("alert on %s: %s", frame.frame_id, alert_text)


async def run() -> None:
    # Fail clearly on missing/invalid weights before accepting queue jobs.
    await asyncio.to_thread(_recognizer)
    connection = RabbitConnection(settings.rabbit)
    handler = AnalyzeHandler(S3Storage(settings.s3), Publisher(connection))
    await run_consumer(connection, AnalyzeMessage, handler)
