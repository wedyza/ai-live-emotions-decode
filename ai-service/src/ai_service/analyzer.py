from common.messages import Face


def analyze_emotions(jpeg: bytes, faces: list[Face]) -> str | None:
    """Анализирует эмоции найденных лиц на JPEG-кадре. Блокирующая, вызывать в отдельном потоке.

    Возвращает текст алерта, если есть о чём уведомить, иначе None.

    TODO: подключить настоящую модель эмоций. Пока алерты не создаются.
    """
    _ = jpeg, faces
    return None
