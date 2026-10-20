import logging

from common.broker import RabbitConnection
from common.lifecycle import run_consumer
from common.messages import AlertMessage

from notification_service.config import settings

logger = logging.getLogger(__name__)


async def handle_alert(message: AlertMessage) -> None:
    """create_alert {frame, text} -> уведомление.

    TODO: отправлять в Telegram. Пока алерты только пишутся в лог.
    """
    logger.warning("ALERT [%s]: %s", message.frame.frame_id, message.text)


async def run() -> None:
    await run_consumer(RabbitConnection(settings.rabbit), AlertMessage, handle_alert)
