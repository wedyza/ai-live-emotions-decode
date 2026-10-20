import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
from av.error import FFmpegError
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ms: int
    jpeg: bytes


class InvalidVideoError(Exception):
    pass


class InvalidImageError(Exception):
    pass


def _scaled_size(width: int, height: int, max_height: int) -> tuple[int, int]:
    if height <= max_height:
        return width, height
    new_width = round(width * max_height / height)
    # Чётные размеры нужны кодекам и моделям.
    return new_width - new_width % 2, max_height


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def extract_frames(
    path: Path, target_fps: float, max_height: int, jpeg_quality: int
) -> Iterator[Frame]:
    """Декодирует видео, оставляет ~target_fps кадров в секунду, уменьшает их и кодирует в JPEG."""
    try:
        container = av.open(str(path))
    except FFmpegError as e:
        raise InvalidVideoError(f"cannot open video: {e}") from e

    with container:
        if not container.streams.video:
            raise InvalidVideoError("file has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        step = 1.0 / target_fps
        next_time: float | None = None
        index = 0
        try:
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                if next_time is None:
                    next_time = frame.time
                if frame.time + 1e-6 < next_time:
                    continue
                # Если в исходнике есть пропуски, пропускаем целые шаги, чтобы выход оставался на сетке 10 к/с.
                while next_time <= frame.time + 1e-6:
                    next_time += step

                width, height = _scaled_size(frame.width, frame.height, max_height)
                rgb = frame.reformat(width=width, height=height, format="rgb24")
                plane = rgb.planes[0]
                image = Image.frombuffer(
                    "RGB", (width, height), bytes(plane), "raw", "RGB", plane.line_size, 1
                )

                yield Frame(
                    index=index,
                    timestamp_ms=round(frame.time * 1000),
                    jpeg=_encode_jpeg(image, jpeg_quality),
                )
                index += 1
        except FFmpegError as e:
            raise InvalidVideoError(f"cannot decode video: {e}") from e


def normalize_jpeg(data: bytes, max_height: int, jpeg_quality: int) -> bytes:
    """Проверяет входящую картинку и уменьшает её, если она выше max_height."""
    try:
        image = Image.open(io.BytesIO(data))
        _ = image.load()
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError("frame is not a valid image") from e

    width, height = _scaled_size(image.width, image.height, max_height)
    if image.format == "JPEG" and (width, height) == image.size:
        return data
    resized = ImageOps.fit(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    return _encode_jpeg(resized, jpeg_quality)
