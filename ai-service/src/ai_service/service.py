import asyncio
import logging

from common.broker import Publisher, RabbitConnection
from common.lifecycle import run_consumer
from common.messages import AlertMessage, AnalyzeMessage
from common.storage import S3Storage

from ai_service.analyzer import analyze_emotions
from ai_service.config import settings

logger = logging.getLogger(__name__)


class AnalyzeHandler:
    """analyze_msg {frame, faces} -> GET кадра из S3 -> анализ эмоций -> create_alert."""

    _storage: S3Storage
    _publisher: Publisher

    def __init__(self, storage: S3Storage, publisher: Publisher) -> None:
        self._storage = storage
        self._publisher = publisher

    async def __call__(self, message: AnalyzeMessage) -> None:
        frame = message.frame
        jpeg = await self._storage.get_async(frame.s3_key)
        alert_text = await asyncio.to_thread(analyze_emotions, jpeg, message.faces)
        if alert_text is None:
            return

        await self._publisher.publish(AlertMessage(frame=frame, text=alert_text))
        logger.info("alert on %s: %s", frame.frame_id, alert_text)


async def run() -> None:
    connection = RabbitConnection(settings.rabbit)
    handler = AnalyzeHandler(S3Storage(settings.s3), Publisher(connection))
    await run_consumer(connection, AnalyzeMessage, handler)
