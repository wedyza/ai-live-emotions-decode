from common.broker import Publisher, RabbitConnection

from backend.config import settings

rabbit = RabbitConnection(settings.rabbit)
publisher = Publisher(rabbit)
