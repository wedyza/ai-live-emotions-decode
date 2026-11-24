from common.face_preprocessing import ALIGNED
from common.paths import project_root
from typing import ClassVar, Literal

from common.settings import RabbitSettings, S3Settings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # CV_RABBIT__URL=amqp://..., CV_RABBIT__PREFETCH_COUNT=4, CV_S3__BUCKET=frames
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="CV_",
        env_nested_delimiter="__",
        env_file=project_root() / ".env",
        extra="ignore",
    )

    # Детекция лиц нагружает CPU: не даём RabbitMQ присылать слишком много кадров разом.
    rabbit: RabbitSettings = Field(
        default_factory=lambda: RabbitSettings(prefetch_count=1)
    )
    s3: S3Settings = Field(default_factory=S3Settings)
    model_path: str = "cv-service/src/model/weights/face_detection_yunet_2023mar.onnx"

    # Кропы лиц в S3 (<frame_id>_f<i>.png). Режим должен совпадать с crop у модели ai-service
    # (models/*/model.onnx.json): ALIGNED для reliable/variant1/aligned, BBOX для experimental/pretrained.
    # При несовпадении ai-service не ломается, а сам режет лицо из кадра — просто без выигрыша.
    save_crops: bool = True
    face_crop: Literal["yunet5_similarity_v1", "bbox without margin or alignment"] = ALIGNED


settings = Settings()
