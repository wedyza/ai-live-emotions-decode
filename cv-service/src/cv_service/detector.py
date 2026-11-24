import logging
import threading
from dataclasses import dataclass
from typing import cast

import cv2 as cv
import numpy as np
from common.face_preprocessing import prepare_face
from common.messages import BoundingBox, Face, FaceLandmarks

from model.yunet import YuNet

logger = logging.getLogger(__name__)
_model_lock = threading.Lock()


@dataclass(frozen=True)
class FaceCapture:
    face: Face
    # PNG без потерь: ai-service должен получить ровно те пиксели, что дал бы prepare_face
    # на кадре. None — лицо нельзя подготовить (например, вырожденные landmarks).
    crop_png: bytes | None


def _decode(jpeg: bytes) -> np.ndarray | None:
    image = cv.imdecode(np.frombuffer(jpeg, np.uint8), cv.IMREAD_COLOR)
    if image is None:
        logger.warning("failed to decode JPEG (%d bytes)", len(jpeg))
    return image


def detect_captures(jpeg: bytes, model: YuNet) -> list[Face]:
    """Ищет лица на JPEG-кадре. Блокирующая (нагружает CPU), вызывать в отдельном потоке."""
    image = _decode(jpeg)
    return [] if image is None else _detect(image, model)


def capture_faces(jpeg: bytes, model: YuNet, crop_mode: str | None) -> list[FaceCapture]:
    """Ищет лица и готовит их кропы тем же prepare_face, что и ai-service. Блокирующая.

    crop_mode=None — только детекция, без кропов.
    """
    image = _decode(jpeg)
    if image is None:
        return []
    faces = _detect(image, model)
    if crop_mode is None:
        return [FaceCapture(face, None) for face in faces]

    # prepare_face работает с RGB; ai-service декодирует кадр через PIL, а OpenCV и PIL
    # декодируют JPEG одинаково (проверяется в tests/test_face_crops.py).
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    captures: list[FaceCapture] = []
    for i, face in enumerate(faces):
        box = face.bbox
        points = face.landmarks.points if face.landmarks is not None else None
        try:
            crop = prepare_face(rgb, (box.x, box.y, box.width, box.height), points, crop_mode)
        except ValueError as exc:
            logger.warning("no crop for face #%d: %s", i, exc)
            captures.append(FaceCapture(face, None))
            continue
        ok, png = cv.imencode(".png", cv.cvtColor(crop, cv.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("failed to encode face crop as PNG")
        captures.append(FaceCapture(face, png.tobytes()))
    return captures


def _detect(image: np.ndarray, model: YuNet) -> list[Face]:
    height, width = cast(tuple[int, int], image.shape[:2])
    # YuNet требует, чтобы входной размер совпадал с размером кадра.
    # Input-size mutation and forward must be one atomic operation.
    with _model_lock:
        model.setInputSize([width, height])  # pyright: ignore[reportUnknownMemberType]
        # Строка на лицо: x, y, w, h, 5 ключевых точек (x, y), score.
        faces = cast(list[list[float]], model.infer(image).tolist())  # pyright: ignore[reportUnknownMemberType]
    logger.info("YuNet: %d faces on %dx%d frame", len(faces), width, height)

    captures: list[Face] = []
    for i, face in enumerate(faces):
        if len(face) != 15 or not np.isfinite(face).all() or not 0 <= face[14] <= 1:
            continue
        x1, y1 = max(int(face[0]), 0), max(int(face[1]), 0)
        x2 = min(int(face[0] + face[2]), width)
        y2 = min(int(face[1] + face[3]), height)
        if x2 <= x1 or y2 <= y1:
            continue
        box = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        logger.info(
            "  face #%d: x=%d y=%d w=%d h=%d score=%.3f",
            i, box.x, box.y, box.width, box.height, face[14],
        )
        captures.append(Face(
            bbox=box, det_score=float(face[14]),
            landmarks=FaceLandmarks(points=[(float(face[k]),float(face[k+1])) for k in range(4,14,2)]),
        ))

    return captures


def detect_faces(jpeg: bytes, model: YuNet) -> list[BoundingBox]:
    """Backward-compatible bbox-only API used by existing callers/tests."""
    return [capture.bbox for capture in detect_captures(jpeg, model)]
