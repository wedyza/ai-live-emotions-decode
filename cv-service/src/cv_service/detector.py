import logging
import threading
from typing import cast

import cv2 as cv
import numpy as np
from common.messages import BoundingBox, Face, FaceLandmarks

from model.yunet import YuNet

logger = logging.getLogger(__name__)
_model_lock = threading.Lock()


def detect_captures(jpeg: bytes, model: YuNet) -> list[Face]:
    """Ищет лица на JPEG-кадре. Блокирующая (нагружает CPU), вызывать в отдельном потоке."""
    image = cv.imdecode(np.frombuffer(jpeg, np.uint8), cv.IMREAD_COLOR)
    if image is None:
        logger.warning("failed to decode JPEG (%d bytes)", len(jpeg))
        return []

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
