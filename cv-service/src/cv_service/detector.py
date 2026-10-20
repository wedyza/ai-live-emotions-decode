from common.messages import BoundingBox


def detect_faces(jpeg: bytes) -> list[BoundingBox]:
    """Ищет лица на JPEG-кадре. Блокирующая (нагружает CPU), вызывать в отдельном потоке.

    TODO: подключить настоящий детектор (OpenCV / MediaPipe / RetinaFace). Пока лица
    не находятся, и дальше по конвейеру ничего не уходит.
    """
    _ = jpeg
    return []
