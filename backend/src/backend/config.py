from typing import ClassVar

from common.settings import RabbitSettings, S3Settings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # BACKEND_TARGET_FPS=10, BACKEND_S3__BUCKET=frames, BACKEND_RABBIT__URL=amqp://...
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="BACKEND_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    rabbit: RabbitSettings = Field(default_factory=RabbitSettings)
    s3: S3Settings = Field(default_factory=S3Settings)

    # Обработка видео
    target_fps: float = 10.0
    max_frame_height: int = 480
    jpeg_quality: int = 85


settings = Settings()
