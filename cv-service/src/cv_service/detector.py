import logging
from typing import cast

import cv2 as cv
import numpy as np
from common.messages import BoundingBox

from model.yunet import YuNet

logger = logging.getLogger(__name__)


def detect_faces(jpeg: bytes, model: YuNet) -> list[BoundingBox]:
    """Ищет лица на JPEG-кадре. Блокирующая (нагружает CPU), вызывать в отдельном потоке."""
    image = cv.imdecode(np.frombuffer(jpeg, np.uint8), cv.IMREAD_COLOR)
    if image is None:
        logger.warning("failed to decode JPEG (%d bytes)", len(jpeg))
        return []

    height, width = cast(tuple[int, int], image.shape[:2])
    # YuNet требует, чтобы входной размер совпадал с размером кадра.
    model.setInputSize([width, height])  # pyright: ignore[reportUnknownMemberType]
    # Строка на лицо: x, y, w, h, 5 ключевых точек (x, y), score.
    faces = cast(list[list[float]], model.infer(image).tolist())  # pyright: ignore[reportUnknownMemberType]
    logger.info("YuNet: %d faces on %dx%d frame", len(faces), width, height)

    boxes: list[BoundingBox] = []
    for i, face in enumerate(faces):
        x1, y1 = max(int(face[0]), 0), max(int(face[1]), 0)
        x2 = min(int(face[0] + face[2]), width)
        y2 = min(int(face[1] + face[3]), height)
        box = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        logger.info(
            "  face #%d: x=%d y=%d w=%d h=%d score=%.3f",
            i, box.x, box.y, box.width, box.height, face[14],
        )
        boxes.append(box)

    return boxes
