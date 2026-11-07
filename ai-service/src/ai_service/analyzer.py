"""CPU inference for the existing AnalyzeMessage -> AlertMessage contract."""

import io
import logging
import threading
import urllib.request  # hsemotion-onnx 0.3.1 accesses urllib.request without importing it
from functools import lru_cache

import numpy as np
from PIL import Image, UnidentifiedImageError
from common.messages import Face
from common.paths import project_path
from ai_service.config import settings

logger = logging.getLogger(__name__)
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _recognizer():
    if settings.emotion_model == "experimental":
        path = project_path(settings.emotion_model_path)
        from ai_service.local_emotion_model import LocalEmotionModel

        logger.info("Loading experimental emotion model: %s", path)
        return LocalEmotionModel(path)

    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

    logger.info("Loading original pretrained emotion model")
    return HSEmotionRecognizer(model_name="enet_b0_8_best_afew")


def analyze_emotions(jpeg: bytes, faces: list[Face]) -> str | None:
    """Return per-face expression predictions; no faces means no notification.

    HSEmotion receives RGB face crops, not the full frame. Percentages are model
    softmax scores, not calibrated confidence or a diagnosis of a person's mood.
    Model initialization and inference are serialized because consumers use threads.
    """
    if not faces:
        return None
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Cannot decode frame image") from exc
    h, w = rgb.shape[:2]
    results = []
    with _lock:
        for i, face in enumerate(faces, 1):
            box = face.bbox
            if box.width <= 0 or box.height <= 0:
                logger.warning("Skipping invalid face bbox: %s", box)
                continue
            x1, y1 = max(0, box.x), max(0, box.y)
            x2, y2 = min(w, box.x + box.width), min(h, box.y + box.height)
            if x2 <= x1 or y2 <= y1:
                logger.warning("Skipping bbox outside image: %s", box)
                continue
            crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
            emotion, scores = _recognizer().predict_emotions(crop, logits=False)
            values = np.asarray(scores, dtype=float).reshape(-1)
            if values.size != 8 or not np.isfinite(values).all():
                raise RuntimeError(
                    "Unexpected HSEmotion output (expected 8 finite scores)"
                )
            score = float(values.max())
            results.append(f"face #{i}: {emotion} (model score {score:.1%})")
    return "; ".join(results) or None
