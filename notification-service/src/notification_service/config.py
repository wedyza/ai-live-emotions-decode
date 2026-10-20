from typing import ClassVar

from common.settings import RabbitSettings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # NOTIFY_RABBIT__URL=amqp://...
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="NOTIFY_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    rabbit: RabbitSettings = Field(default_factory=RabbitSettings)
    # TODO: настройки Telegram, например `telegram_bot_token: SecretStr` (обязательная, без значения по умолчанию).


settings = Settings()
