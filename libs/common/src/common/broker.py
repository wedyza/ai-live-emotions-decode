import asyncio
import logging
from collections.abc import Awaitable, Callable

import aio_pika
from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustQueue,
)
from pydantic import ValidationError

from common.messages import Message
from common.settings import RabbitSettings

logger = logging.getLogger(__name__)


class BrokerError(Exception):
    pass


class BrokerNotConnectedError(BrokerError):
    pass


def dead_letter_queue(queue: str) -> str:
    return f"{queue}.dlq"


class RabbitConnection:
    """Одно robust-соединение и канал на процесс, общие для publisher'ов и consumer'ов."""

    _settings: RabbitSettings
    _connection: AbstractRobustConnection | None
    _channel: AbstractRobustChannel | None
    _queues: dict[str, AbstractRobustQueue]

    def __init__(self, settings: RabbitSettings) -> None:
        self._settings = settings
        self._connection = None
        self._channel = None
        self._queues = {}

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(self._settings.url.get_secret_value())
        self._channel = await self._connection.channel()
        _ = await self._channel.set_qos(prefetch_count=self._settings.prefetch_count)

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()
        self._connection = None
        self._channel = None
        self._queues.clear()

    @property
    def channel(self) -> AbstractRobustChannel:
        if self._channel is None:
            raise BrokerNotConnectedError("call connect() first")
        return self._channel

    async def declare_queue(self, name: str) -> AbstractRobustQueue:
        """Объявляет durable-очередь вместе с её dead-letter очередью.

        Producer и consumer должны объявлять очередь с одинаковыми аргументами,
        поэтому все сервисы объявляют очереди только через этот метод.
        """
        if name not in self._queues:
            _ = await self.channel.declare_queue(dead_letter_queue(name), durable=True)
            self._queues[name] = await self.channel.declare_queue(
                name,
                durable=True,
                arguments={
                    # Отклонённые сообщения уходят в "<name>.dlq" через default exchange.
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": dead_letter_queue(name),
                },
            )
        return self._queues[name]


class Publisher:
    _connection: RabbitConnection

    def __init__(self, connection: RabbitConnection) -> None:
        self._connection = connection

    async def publish(self, message: Message) -> None:
        _ = await self._connection.declare_queue(message.QUEUE)
        _ = await self._connection.channel.default_exchange.publish(
            aio_pika.Message(
                body=message.model_dump_json().encode(),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=message.QUEUE,
        )

# generic типы для сообщения, чтобы переиспользовать их в других сервисах
type Handler[M: Message] = Callable[[M], Awaitable[None]]


class Consumer[M: Message]:
    _connection: RabbitConnection
    _model: type[M]
    _consumer_tag: str | None
    _queue: AbstractRobustQueue | None
    _in_flight: int
    _idle: asyncio.Event

    def __init__(self, connection: RabbitConnection, model: type[M]) -> None:
        self._connection = connection
        self._model = model
        self._consumer_tag = None
        self._queue = None
        self._in_flight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    async def start(self, handler: Handler[M]) -> None:
        async def on_message(raw: AbstractIncomingMessage) -> None:
            self._in_flight += 1
            self._idle.clear()
            try:
                await self._process(raw, handler)
            finally:
                self._in_flight -= 1
                if self._in_flight == 0:
                    self._idle.set()

        self._queue = await self._connection.declare_queue(self._model.QUEUE)
        self._consumer_tag = await self._queue.consume(on_message)
        logger.info("consuming %s", self._model.QUEUE)

    async def stop(self, timeout: float = 30.0) -> None:
        """Перестаёт брать новые сообщения и ждёт те, что уже обрабатываются."""
        if self._queue and self._consumer_tag:
            _ = await self._queue.cancel(self._consumer_tag)
            self._consumer_tag = None
        try:
            _ = await asyncio.wait_for(self._idle.wait(), timeout)
        except TimeoutError:
            # Неподтверждённые сообщения вернутся в очередь после закрытия соединения.
            logger.warning("%d messages still in flight after %.0fs", self._in_flight, timeout)

    async def _process(self, raw: AbstractIncomingMessage, handler: Handler[M]) -> None:
        # Отмену (остановку сервиса) не ловим: сообщение останется неподтверждённым и будет доставлено повторно.
        try:
            message = self._model.model_validate_json(raw.body)
        except ValidationError:
            logger.exception("invalid %s message, moving to DLQ", self._model.__name__)
            await raw.reject(requeue=False)
            return
        try:
            await handler(message)
        except Exception:
            logger.exception("handler failed on %s, moving to DLQ", message)
            await raw.reject(requeue=False)
            return
        await raw.ack()
