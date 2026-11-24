import asyncio
import logging
from collections.abc import Awaitable

from common.broker import Publisher, RabbitConnection
from common.lifecycle import run_consumer
from common.messages import AnalyzeMessage, Face, FaceCaptureMessage, FrameRef
from common.storage import S3Storage

from cv_service.config import settings
from cv_service.detector import capture_faces
from model.yunet import get_yunet

logger = logging.getLogger(__name__)


def crop_key(frame: FrameRef, index: int) -> str:
    """<video_id>/000012_f0.png — рядом с кадром <video_id>/000012.jpg."""
    return f"{frame.frame_id}_f{index}.png"


class FaceCaptureHandler:
    """do_face_capture {frame} -> GET кадра из S3 -> поиск лиц -> PUT кропов в S3 -> analyze_msg."""

    _storage: S3Storage
    _publisher: Publisher
    _crop_mode: str | None

    def __init__(
        self, storage: S3Storage, publisher: Publisher, crop_mode: str | None = None
    ) -> None:
        self._storage = storage
        self._publisher = publisher
        self._crop_mode = crop_mode
        self._model = get_yunet()

    async def __call__(self, message: FaceCaptureMessage) -> None:
        frame = message.frame
        jpeg = await self._storage.get_async(frame.s3_key)
        captures = await asyncio.to_thread(capture_faces, jpeg, self._model, self._crop_mode)
        if not captures:
            logger.debug("no faces on %s", frame.frame_id)
            return

        faces: list[Face] = []
        uploads: list[Awaitable[None]] = []
        for i, capture in enumerate(captures):
            if capture.crop_png is None:
                faces.append(capture.face)
                continue
            key = crop_key(frame, i)
            uploads.append(self._storage.put_async(key, capture.crop_png, "image/png"))
            faces.append(
                capture.face.model_copy(update={"crop_s3_key": key, "crop_mode": self._crop_mode})
            )
        # Сообщение уходит только после записи всех кропов: ai-service не должен получить
        # ссылку на объект, которого ещё нет. Ошибка записи -> сообщение уходит в DLQ.
        _ = await asyncio.gather(*uploads)

        await self._publisher.publish(AnalyzeMessage(frame=frame, faces=faces))
        logger.info("%d faces (%d crops) on %s", len(faces), len(uploads), frame.frame_id)


async def run() -> None:
    connection = RabbitConnection(settings.rabbit)
    handler = FaceCaptureHandler(
        S3Storage(settings.s3),
        Publisher(connection),
        settings.face_crop if settings.save_crops else None,
    )
    await run_consumer(connection, FaceCaptureMessage, handler)
