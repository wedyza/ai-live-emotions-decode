import json

import aio_pika
from aio_pika.abc import AbstractRobustChannel, AbstractRobustConnection

from backend.config import settings


class Broker:
    def __init__(self) -> None:
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractRobustChannel | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        self._channel = await self._connection.channel()
        await self._channel.declare_queue(settings.face_capture_queue, durable=True)

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()

    async def publish(self, queue: str, payload: dict) -> None:
        assert self._channel, "broker is not connected"
        await self._channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload).encode(),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=queue,
        )
