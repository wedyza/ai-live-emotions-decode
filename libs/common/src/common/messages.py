from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    QUEUE: ClassVar[str]


class FrameRef(BaseModel):
    video_id: str
    frame_id: str
    s3_key: str
    timestamp_ms: int


class FaceCaptureMessage(Message):
    """backend -> cv-service: в S3 появился новый кадр, найти на нём лица."""

    QUEUE: ClassVar[str] = "do_face_capture"

    frame: FrameRef


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Face(BaseModel):
    bbox: BoundingBox
    # Вырезанное лицо, если cv-service сохраняет кропы.
    crop_s3_key: str | None = None


class AnalyzeMessage(Message):
    """cv-service -> ai-service: на кадре найдены лица, проанализировать их эмоции."""

    QUEUE: ClassVar[str] = "analyze_msg"

    frame: FrameRef
    faces: list[Face]


class AlertMessage(Message):
    """ai-service -> notification-service: есть о чём уведомить."""

    QUEUE: ClassVar[str] = "create_alert"

    frame: FrameRef
    text: str
