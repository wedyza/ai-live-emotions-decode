from typing import ClassVar

from common.settings import RabbitSettings, S3Settings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # CV_RABBIT__URL=amqp://..., CV_RABBIT__PREFETCH_COUNT=4, CV_S3__BUCKET=frames
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="CV_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    # Детекция лиц нагружает CPU: не даём RabbitMQ присылать слишком много кадров разом.
    rabbit: RabbitSettings = Field(default_factory=lambda: RabbitSettings(prefetch_count=4))
    s3: S3Settings = Field(default_factory=S3Settings)


settings = Settings()
