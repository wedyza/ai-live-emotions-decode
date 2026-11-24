"""CPU inference for the existing AnalyzeMessage -> AlertMessage contract."""

import io
import logging
import threading
import urllib.request  # hsemotion-onnx 0.3.1 accesses urllib.request without importing it
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from PIL import Image, UnidentifiedImageError
from common.messages import Face
from common.paths import project_path
from common.face_preprocessing import prepare_face, BBOX
from ai_service.config import settings

logger = logging.getLogger(__name__)
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _recognizer():
    if settings.emotion_model == "reliable":
        from ai_service.reliable_emotion_model import ReliableEmotionModel
        return ReliableEmotionModel(project_path(settings.reliability_path))

    if settings.emotion_model == "experimental":
        path = project_path(settings.emotion_model_path)
        from ai_service.local_emotion_model import LocalEmotionModel

        logger.info("Loading experimental emotion model: %s", path)
        return LocalEmotionModel(path)

    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

    logger.info("Loading original pretrained emotion model")
    return HSEmotionRecognizer(model_name="enet_b0_8_best_afew")


def expected_crop_mode() -> str:
    """Режим common.face_preprocessing, на котором обучена выбранная модель."""
    return getattr(_recognizer(), "crop_mode", BBOX)


def _decode_rgb(data: bytes, what: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB")).copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot decode {what}") from exc


def analyze_emotions(
    jpeg: bytes | None, faces: list[Face], crops: Sequence[bytes | None] | None = None
) -> str | None:
    """Return per-face expression predictions; no faces means no notification.

    HSEmotion receives RGB face crops, not the full frame. Percentages are model
    softmax scores, not calibrated confidence or a diagnosis of a person's mood.
    Model initialization and inference are serialized because consumers use threads.

    crops[i] — готовый кроп лица faces[i] от cv-service (PNG), уже подготовленный в режиме
    expected_crop_mode(). Для лиц без кропа лицо вырезается из кадра jpeg; кадр нужен,
    только если такие лица есть.
    """
    if not faces:
        return None
    if crops is not None and len(crops) != len(faces):
        raise ValueError("crops must match faces one-to-one")
    face_crops: Sequence[bytes | None] = crops if crops is not None else [None] * len(faces)
    rgb: np.ndarray | None = None
    if any(crop is None for crop in face_crops):
        if jpeg is None:
            raise ValueError("Frame image is required for faces without crops")
        rgb = _decode_rgb(jpeg, "frame image")
    results = []
    with _lock:
        for i, (face, crop_png) in enumerate(zip(faces, face_crops), 1):
            recognizer = _recognizer()
            points = face.landmarks.points if face.landmarks is not None else None
            if crop_png is not None:
                crop = _decode_rgb(crop_png, f"crop of face #{i}")
            else:
                assert rgb is not None
                h, w = rgb.shape[:2]
                box = face.bbox
                if box.width <= 0 or box.height <= 0:
                    logger.warning("Skipping invalid face bbox: %s", box)
                    continue
                x1, y1 = max(0, box.x), max(0, box.y)
                x2, y2 = min(w, box.x + box.width), min(h, box.y + box.height)
                if x2 <= x1 or y2 <= y1:
                    logger.warning("Skipping bbox outside image: %s", box)
                    continue
                mode = getattr(recognizer, "crop_mode", BBOX)
                try:
                    crop = prepare_face(rgb, (box.x, box.y, box.width, box.height), points, mode)
                except ValueError as exc:
                    logger.warning("Skipping face #%d: %s", i, exc)
                    if hasattr(recognizer, "predict_decision"):
                        results.append(f"face #{i}: uncertain (invalid_face_geometry)")
                    continue
            if hasattr(recognizer, "predict_decision"):
                decision = recognizer.predict_decision(crop)
                if not decision.accepted:
                    results.append(f"face #{i}: uncertain ({decision.reason})")
                    continue
                emotion, scores = decision.label, decision.scores
            else:
                emotion, scores = recognizer.predict_emotions(crop, logits=False)
            values = np.asarray(scores, dtype=float).reshape(-1)
            if values.size != 8 or not np.isfinite(values).all():
                raise RuntimeError(
                    "Unexpected HSEmotion output (expected 8 finite scores)"
                )
            score = float(values.max())
            results.append(f"face #{i}: {emotion} (model score {score:.1%})")
    return "; ".join(results) or None
