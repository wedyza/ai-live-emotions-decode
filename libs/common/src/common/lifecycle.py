import asyncio
import logging
import signal

from common.broker import Consumer, Handler, RabbitConnection
from common.messages import Message


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


async def wait_for_shutdown() -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        _ = await stop.wait()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            _ = loop.remove_signal_handler(sig)


async def run_consumer[M: Message](
    connection: RabbitConnection, model: type[M], handler: Handler[M]
) -> None:
    """Обрабатывает `model.QUEUE` через `handler` до SIGINT/SIGTERM, затем корректно останавливается."""
    await connection.connect()
    consumer = Consumer(connection, model)
    try:
        await consumer.start(handler)
        await wait_for_shutdown()
    finally:
        await consumer.stop()
        await connection.close()
