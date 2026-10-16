import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ms: int
    jpeg: bytes


class InvalidVideoError(Exception):
    pass


def _scaled_size(width: int, height: int, max_height: int) -> tuple[int, int]:
    if height <= max_height:
        return width, height
    new_width = round(width * max_height / height)
    # Even dimensions keep codecs and models happy.
    return new_width - new_width % 2, max_height


def extract_frames(
    path: Path, target_fps: float, max_height: int, jpeg_quality: int
) -> Iterator[Frame]:
    """Decode a video, keep ~target_fps frames, downscale them and encode as JPEG."""
    try:
        container = av.open(str(path))
    except av.FFmpegError as e:
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
                if frame.time is None:
                    continue
                if next_time is None:
                    next_time = frame.time
                if frame.time + 1e-6 < next_time:
                    continue
                # Skip whole steps if the source has gaps, so output stays on a 10 fps grid.
                while next_time <= frame.time + 1e-6:
                    next_time += step

                width, height = _scaled_size(frame.width, frame.height, max_height)
                image = frame.to_image(width=width, height=height)
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=jpeg_quality)

                yield Frame(
                    index=index,
                    timestamp_ms=round(frame.time * 1000),
                    jpeg=buf.getvalue(),
                )
                index += 1
        except av.FFmpegError as e:
            raise InvalidVideoError(f"cannot decode video: {e}") from e
