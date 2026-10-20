import asyncio
import logging

from common.broker import Publisher, RabbitConnection
from common.lifecycle import run_consumer
from common.messages import AnalyzeMessage, Face, FaceCaptureMessage
from common.storage import S3Storage

from cv_service.config import settings
from cv_service.detector import detect_faces

logger = logging.getLogger(__name__)


class FaceCaptureHandler:
    """do_face_capture {frame} -> GET кадра из S3 -> поиск лиц -> analyze_msg."""

    _storage: S3Storage
    _publisher: Publisher

    def __init__(self, storage: S3Storage, publisher: Publisher) -> None:
        self._storage = storage
        self._publisher = publisher

    async def __call__(self, message: FaceCaptureMessage) -> None:
        frame = message.frame
        jpeg = await self._storage.get_async(frame.s3_key)
        boxes = await asyncio.to_thread(detect_faces, jpeg)
        if not boxes:
            logger.debug("no faces on %s", frame.frame_id)
            return

        await self._publisher.publish(
            AnalyzeMessage(frame=frame, faces=[Face(bbox=box) for box in boxes])
        )
        logger.info("%d faces on %s", len(boxes), frame.frame_id)


async def run() -> None:
    connection = RabbitConnection(settings.rabbit)
    handler = FaceCaptureHandler(S3Storage(settings.s3), Publisher(connection))
    await run_consumer(connection, FaceCaptureMessage, handler)
