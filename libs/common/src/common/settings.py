from pydantic import BaseModel, SecretStr


class RabbitSettings(BaseModel):
    url: SecretStr = SecretStr("amqp://guest:guest@localhost:5672/")
    # Сколько неподтверждённых сообщений RabbitMQ может одновременно выдать одному consumer'у.
    prefetch_count: int = 10


class S3Settings(BaseModel):
    endpoint_url: str = "http://localhost:9000"
    access_key: str = "minioadmin"
    secret_key: SecretStr = SecretStr("minioadmin")
    bucket: str = "frames"
