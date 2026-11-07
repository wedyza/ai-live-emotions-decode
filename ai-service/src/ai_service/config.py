from common.paths import project_root
from typing import ClassVar, Literal

from common.settings import RabbitSettings, S3Settings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # AI_RABBIT__URL=amqp://..., AI_RABBIT__PREFETCH_COUNT=2, AI_S3__BUCKET=frames
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="AI_",
        env_nested_delimiter="__",
        env_file=project_root() / ".env",
        extra="ignore",
    )

    # Bundled experimental model; select pretrained explicitly for comparison.
    emotion_model: Literal["experimental", "pretrained"] = "experimental"
    emotion_model_path: str = "models/emotion_experiment/model.onnx"

    # Инференс модели — самый медленный шаг конвейера.
    rabbit: RabbitSettings = Field(default_factory=lambda: RabbitSettings(prefetch_count=2))
    s3: S3Settings = Field(default_factory=S3Settings)


settings = Settings()
