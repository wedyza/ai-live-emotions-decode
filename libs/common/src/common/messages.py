from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


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


class FaceLandmarks(BaseModel):
    # Full-frame pixel coordinates, in YuNet order; never crop-relative.
    points: list[tuple[FiniteFloat, FiniteFloat]] = Field(min_length=5, max_length=5)


class Face(BaseModel):
    bbox: BoundingBox
    landmarks: FaceLandmarks | None = None
    det_score: float | None = Field(default=None, ge=0, le=1)
    # Готовый RGB-кроп (PNG без потерь), который cv-service подготовил через prepare_face.
    # crop_mode — режим common.face_preprocessing (BBOX / ALIGNED), которым он сделан:
    # ai-service берёт кроп, только если режим совпадает с режимом его модели,
    # иначе сам вырезает лицо из кадра по bbox и landmarks.
    crop_s3_key: str | None = None
    crop_mode: str | None = None


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
